use anyhow::Result;
use tch::{Device, Kind, Tensor};

use crate::{
    config::{Model, Sampling},
    device::{Profile, Timings},
    layers::{Block, Kv, Linear, Norm, Rope, embedding},
    neural_tokenizer::NeuralTokenizer,
    weights::Weights,
};

struct Level {
    projection: Linear,
    inverse_temperature: Tensor,
    scale: Tensor,
    rope: Rope,
}

pub struct Generator {
    source: Linear,
    level_embedding: Tensor,
    blocks: Vec<Block>,
    norm: Norm,
    query: Tensor,
    levels: Vec<Level>,
    widths: Vec<i64>,
    rope: Rope,
    pad: i64,
    device: Device,
}

impl Generator {
    pub fn load(
        weights: &Weights,
        config: &Model,
        vq: &NeuralTokenizer,
        widths: &[i64],
        device: Device,
    ) -> Result<Self> {
        let head_dim = config.n_embd / config.n_head;
        let levels = widths
            .iter()
            .enumerate()
            .map(|(i, &width)| {
                Ok(Level {
                    projection: Linear::load(weights, &format!("lm_heads.{i}.proj"))?,
                    inverse_temperature: weights.get(&format!("lm_heads.{i}.log_inv_T"))?,
                    scale: vq.squared[i].mean(Kind::Float).clamp_min(1e-12),
                    rope: Rope::new(width, head_dim, config.rope_base, device),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            source: Linear::load(weights, "transformer.wte")?,
            level_embedding: weights.get("transformer.wle.weight")?,
            blocks: (0..config.n_layer)
                .map(|i| Block::load(weights, &format!("transformer.blocks.{i}"), config.n_head))
                .collect::<Result<Vec<_>>>()?,
            norm: Norm::load(weights, "transformer.ln_f")?,
            query: weights.get("l0_queries")?,
            levels,
            widths: widths.to_vec(),
            rope: Rope::new(
                config.prefix_len.max(*widths.last().unwrap()),
                head_dim,
                config.rope_base,
                device,
            ),
            pad: config.prefix_pad_token_id,
            device,
        })
    }

    pub fn int8(&mut self) -> Result<()> {
        for block in &mut self.blocks {
            block.int8()?;
        }
        for level in &mut self.levels {
            level.projection.int8()?;
        }
        Ok(())
    }

    pub fn logits(&self, hidden: &Tensor, vq: &NeuralTokenizer, index: usize) -> Tensor {
        let level = &self.levels[index];
        let distance = level
            .projection
            .forward(hidden)
            .matmul(&vq.tables[index].transpose(0, 1))
            * 2.
            - &vq.squared[index];
        distance / &level.scale * level.inverse_temperature.exp()
    }

    fn stack(
        &self,
        x: &Tensor,
        mask: &Tensor,
        rope: &Rope,
        cache: Option<&[Kv]>,
        dropout: f64,
        mut timing: Option<&mut Timings>,
    ) -> Tensor {
        let mut x = x.shallow_clone();
        for (i, block) in self.blocks.iter().enumerate() {
            x = block
                .forward(
                    &x,
                    Some(mask),
                    Some(rope),
                    cache.map(|cache| &cache[i]),
                    dropout,
                    timing.as_deref_mut(),
                )
                .0;
        }
        x
    }

    fn prefill(
        &self,
        mut hidden: Tensor,
        mask: &Tensor,
        rope: &Rope,
        prefix_width: i64,
        timing: &mut Timings,
    ) -> (Tensor, Vec<Kv>) {
        let mut cache = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (next, kv) = block.forward(&hidden, Some(mask), Some(rope), None, 0., Some(timing));
            hidden = next;
            cache.push(kv.prefix(prefix_width));
        }
        (hidden, cache)
    }

    pub fn infer(
        &self,
        vq: &NeuralTokenizer,
        ids: &Tensor,
        sampling: Sampling,
        profile: Profile,
        timing: &mut Timings,
    ) -> (Tensor, Tensor) {
        let shape = ids.size();
        let (batch, prefix_width) = (shape[0], shape[1]);
        let keep = ids.ne(self.pad);
        let prefix = timing.stage("input_embedding", |_| {
            self.source.forward(&embedding(&vq.source, ids))
        });
        // CPU INT8 chooses activation scales dynamically. Keep the two
        // inputs separate there to preserve the existing quantization.
        let separate_prefill = profile == Profile::Ultra && self.device == Device::Cpu;
        let mut cache = if separate_prefill {
            timing.stage("transformer_prefix", |timing| {
                let position = Tensor::arange(prefix_width, (Kind::Int64, self.device));
                let rope = self.rope.select(&position);
                let mask = keep.unsqueeze(1).unsqueeze(1);
                self.prefill(prefix.shallow_clone(), &mask, &rope, prefix_width, timing)
                    .1
            })
        } else {
            Vec::new()
        };
        let mut features = self.query.expand([batch, -1, -1], false);
        let mut latent = Tensor::zeros(
            [batch, *self.widths.last().unwrap(), vq.source.size()[1]],
            (Kind::Float, self.device),
        );
        let mut predictions = Vec::new();

        for (i, &width) in self.widths.iter().enumerate() {
            let logits = timing.stage(format!("transformer_level_{i}"), |timing| {
                let target = self.source.forward(&features) + self.level_embedding.get(i as i64);
                let mask = Tensor::cat(
                    &[
                        &keep,
                        &Tensor::ones([batch, width], (Kind::Bool, self.device)),
                    ],
                    1,
                )
                .unsqueeze(1)
                .unsqueeze(1);
                let hidden = if profile == Profile::Quality || (i == 0 && !separate_prefill) {
                    let x = Tensor::cat(&[&prefix, &target], 1);
                    let positions = Tensor::cat(
                        &[
                            Tensor::arange(prefix_width, (Kind::Int64, self.device)),
                            Tensor::arange(width, (Kind::Int64, self.device)),
                        ],
                        0,
                    );
                    let rope = self.rope.select(&positions);
                    let source_rows =
                        Tensor::arange(prefix_width + width, (Kind::Int64, self.device))
                            .lt(prefix_width);
                    let allowed = source_rows
                        .unsqueeze(1)
                        .logical_not()
                        .logical_or(&source_rows.unsqueeze(0));
                    let mask = mask.logical_and(&allowed);
                    let hidden = if profile == Profile::Quality {
                        self.stack(&x, &mask, &rope, None, 0., Some(timing))
                    } else {
                        // Prefix rows cannot attend to targets, so their K/V
                        // remain valid when the first scale runs alongside them.
                        let (hidden, prefix_cache) =
                            self.prefill(x, &mask, &rope, prefix_width, timing);
                        cache = prefix_cache;
                        hidden
                    };
                    hidden.narrow(1, prefix_width, width)
                } else {
                    self.stack(
                        &target,
                        &mask,
                        &self.levels[i].rope,
                        Some(&cache),
                        0.,
                        Some(timing),
                    )
                };
                self.logits(&self.norm.forward(&hidden), vq, i)
            });
            let predicted = timing.stage("concept_sampling", |_| {
                let logits = if batch == 2 {
                    let conditional = logits.narrow(0, 0, 1);
                    if i == 0 {
                        conditional
                    } else {
                        let strength =
                            sampling.cfg_scale * i as f64 / (self.widths.len() - 1) as f64;
                        conditional * (1. + strength) - logits.narrow(0, 1, 1) * strength
                    }
                } else {
                    logits
                };
                sample(&logits, sampling)
            });
            let applied = if batch == 2 {
                predicted.repeat([2, 1])
            } else {
                predicted.shallow_clone()
            };
            timing.stage("rvq_generation", |_| {
                latent = &latent + vq.contribution(&applied, i);
                if i + 1 < self.widths.len() {
                    features = vq.resize(&latent, i + 1);
                }
            });
            predictions.push(predicted);
        }
        (Tensor::cat(&predictions, 1), latent.narrow(0, 0, 1))
    }

    pub fn teacher_hidden(
        &self,
        vq: &NeuralTokenizer,
        prefix: &Tensor,
        codes: &[Tensor],
        dropout: f64,
    ) -> Vec<Tensor> {
        let prefix_width = prefix.size()[1];
        let mut target = Vec::new();
        let mut full = Tensor::zeros_like(&vq.position).unsqueeze(0);
        for (i, ids) in codes.iter().enumerate() {
            let features = if i == 0 {
                self.query.shallow_clone()
            } else {
                vq.resize(&full, i)
            };
            target.push(self.source.forward(&features) + self.level_embedding.get(i as i64));
            if i + 1 < codes.len() {
                full = &full + vq.contribution(ids, i);
            }
        }
        let source = self.source.forward(&embedding(&vq.source, prefix));
        let mut chunks = vec![source];
        chunks.extend(target);
        let x = Tensor::cat(&chunks, 1);
        let mut level_ids = vec![-1_i64; prefix_width as usize];
        let mut positions: Vec<i64> = (0..prefix_width).collect();
        for (i, &width) in self.widths.iter().enumerate() {
            level_ids.extend(std::iter::repeat_n(i as i64, width as usize));
            positions.extend(0..width);
        }
        let l = Tensor::from_slice(&level_ids).to_device(self.device);
        let mask = l
            .unsqueeze(0)
            .lt(0)
            .logical_or(&l.unsqueeze(0).eq_tensor(&l.unsqueeze(1)));
        let valid = Tensor::cat(
            &[
                prefix.ne(self.pad),
                Tensor::ones([1, x.size()[1] - prefix_width], (Kind::Bool, self.device)),
            ],
            1,
        );
        let mask = mask.logical_and(&valid.unsqueeze(1));
        let rope = self
            .rope
            .select(&Tensor::from_slice(&positions).to_device(self.device));
        let result = self.stack(&x, &mask, &rope, None, dropout, None);
        self.norm
            .forward(&result.narrow(1, prefix_width, x.size()[1] - prefix_width))
            .split_with_sizes(&self.widths, 1)
    }
}

fn sample(logits: &Tensor, cfg: Sampling) -> Tensor {
    if cfg.temperature == 0. || cfg.top_p == 0. {
        return logits.argmax(-1, false);
    }
    let shape = logits.size();
    let vocab = shape[2];
    let mut flat = logits.view([-1, vocab]) / cfg.temperature.max(1e-10);
    if cfg.top_k > 0 {
        let kth = flat
            .topk(cfg.top_k, -1, true, true)
            .0
            .narrow(-1, cfg.top_k - 1, 1);
        flat = flat.masked_fill(&flat.lt_tensor(&kth), f64::NEG_INFINITY);
    }
    let (sorted, order) = flat.sort(-1, true);
    let probs = sorted.softmax(-1, Kind::Float);
    let remove = (probs.cumsum(-1, Kind::Float) - &probs).gt(cfg.top_p);
    let sorted = sorted.masked_fill(&remove, f64::NEG_INFINITY);
    let filtered = Tensor::zeros_like(&flat).scatter(-1, &order, &sorted);
    filtered
        .softmax(-1, Kind::Float)
        .multinomial(1, false)
        .view([shape[0], shape[1]])
}

use anyhow::Result;
use tch::{Device, Kind, Tensor};

use crate::{
    config::Vqvae,
    layers::{Block, Conv, Linear, Norm, embedding, group_norm},
    weights::Weights,
};

struct Area {
    target: i64,
    gather: Option<(Tensor, Tensor, Tensor)>,
}

impl Area {
    fn new(length: i64, target: i64, device: Device) -> Self {
        let gather = if device == Device::Mps && length % target != 0 {
            let width = (length + target - 1) / target + 1;
            let mut ids = Vec::new();
            let mut padding = Vec::new();
            let mut count = Vec::new();
            for t in 0..target {
                let start = t * length / target;
                let stop = ((t + 1) * length + target - 1) / target;
                count.push((stop - start) as f32);
                for offset in 0..width {
                    ids.push((start + offset).min(length - 1));
                    padding.push(start + offset >= stop);
                }
            }
            Some((
                Tensor::from_slice(&ids).to_device(device),
                Tensor::from_slice(&padding)
                    .view([target, width])
                    .to_device(device),
                Tensor::from_slice(&count).to_device(device),
            ))
        } else {
            None
        };
        Self { target, gather }
    }

    fn resize(&self, x: &Tensor) -> Tensor {
        if x.size()[1] == self.target {
            return x.shallow_clone();
        }
        let x = x.transpose(1, 2);
        let output = if let Some((ids, pad, count)) = &self.gather {
            let width = pad.size()[1];
            x.index_select(-1, ids)
                .unflatten(-1, [self.target, width])
                .masked_fill(pad, 0)
                .sum_dim_intlist([-1].as_slice(), false, Kind::Float)
                / count
        } else {
            x.adaptive_avg_pool1d([self.target])
        };
        output.transpose(1, 2).contiguous()
    }
}

pub struct NeuralTokenizer {
    pub source: Tensor,
    pub position: Tensor,
    pub tables: Vec<Tensor>,
    pub squared: Vec<Tensor>,
    residuals: Vec<Conv>,
    up: [Conv; 2],
    down: [Conv; 2],
    area: Vec<Area>,
    ratio: f64,
    groups: i64,
    decoder: Block,
    decoder_norm: Norm,
    projection: Linear,
}

impl NeuralTokenizer {
    pub fn load(weights: &Weights, config: &Vqvae, device: Device, ultra: bool) -> Result<Self> {
        let levels = &config.vector_quantizer_config.levels.manual_levels;
        let tables = (0..levels.len())
            .map(|i| weights.get(&format!("vq.embeddings.{i}.weight")))
            .collect::<Result<Vec<_>>>()?;
        let squared = tables
            .iter()
            .map(|x| {
                x.square()
                    .sum_dim_intlist([-1].as_slice(), false, Kind::Float)
            })
            .collect();
        let mut projection = Linear::load(weights, "decoder.output_layer")?;
        let mut decoder = Block::load(weights, "decoder.middle_block", config.n_layers)?;
        if ultra {
            if device == Device::Mps {
                projection.half();
            } else {
                decoder.int8()?;
                projection.int8()?;
            }
        }
        Ok(Self {
            source: weights.get("encoder.wte.weight")?,
            position: weights.get("encoder.wpe.weight")?,
            tables,
            squared,
            residuals: (0..levels.len())
                .map(|i| Conv::load(weights, &format!("vq.quant_resi.{i}")))
                .collect::<Result<Vec<_>>>()?,
            up: [
                Conv::load(weights, "vq.learned_up_l1.0")?,
                Conv::load(weights, "vq.learned_up_l1.2")?,
            ],
            down: [
                Conv::load(weights, "vq.learned_down_l1.0")?,
                Conv::load(weights, "vq.learned_down_l1.2")?,
            ],
            area: levels
                .iter()
                .map(|&scale| Area::new(config.context_length, scale, device))
                .collect(),
            ratio: config.vector_quantizer_config.quant_resi.ratio.abs(),
            groups: config.pre_quant_groupnorm,
            decoder,
            decoder_norm: Norm::load(weights, "decoder.ln_f")?,
            projection,
        })
    }

    pub fn resize(&self, latent: &Tensor, level: usize) -> Tensor {
        self.area[level].resize(latent)
    }

    pub fn contribution(&self, codes: &Tensor, level: usize) -> Tensor {
        let quantized = embedding(&self.tables[level], codes).transpose(1, 2);
        let full = if level == 0 {
            self.up[1].transpose(&group_norm(&self.up[0].transpose(&quantized, 3), 8), 4)
        } else if level + 1 == self.area.len() {
            quantized
        } else {
            quantized.upsample_linear1d([self.position.size()[0]], false, None)
        };
        let mixed =
            &full * (1. - self.ratio) + self.residuals[level].forward(&full, 1, 1) * self.ratio;
        mixed.transpose(1, 2).contiguous()
    }

    pub fn encode(&self, ids: &Tensor) -> Tensor {
        let x = embedding(&self.source, ids) + &self.position;
        group_norm(&x.transpose(1, 2), self.groups).transpose(1, 2)
    }

    pub fn search(&self, encoded: &Tensor, balanced: bool) -> Vec<Tensor> {
        let mut residual = encoded.shallow_clone();
        let mut codes = Vec::new();
        for (i, table) in self.tables.iter().enumerate() {
            let reduced = if i == 0 {
                let x = self.down[0].forward(&residual.transpose(1, 2), 4, 0);
                group_norm(&self.down[1].forward(&group_norm(&x, 8), 3, 0), 8).transpose(1, 2)
            } else {
                self.resize(&residual, i)
            };
            let norm = if balanced {
                self.squared[i].shallow_clone()
            } else {
                table
                    .square()
                    .sum_dim_intlist([-1].as_slice(), false, Kind::Float)
            };
            let dist = reduced
                .square()
                .sum_dim_intlist([-1].as_slice(), true, Kind::Float)
                + norm
                - reduced.matmul(&table.transpose(0, 1)) * 2.;
            let ids = dist.argmin(-1, false);
            if i + 1 != self.tables.len() {
                residual = &residual - self.contribution(&ids, i);
            }
            codes.push(ids);
        }
        codes
    }

    pub fn decode(&self, latent: &Tensor) -> Tensor {
        let (hidden, _) = self.decoder.forward(latent, None, None, None, 0., None);
        self.projection.forward(&self.decoder_norm.forward(&hidden))
    }
}

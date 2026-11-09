use anyhow::{Context, Result, ensure};
use serde::Serialize;
use std::{collections::BTreeMap, path::Path, time::Instant};
use tch::{Device, Tensor, no_grad};

use crate::{
    config::{Config, Sampling},
    device::{
        Backend, Profile, Timings, backend_name, device, mps_memory, peak_rss_bytes, synchronize,
    },
    generator::Generator,
    neural_tokenizer::NeuralTokenizer,
    text::{TextPolicy, Vocabulary},
    weights::Weights,
};

#[derive(Clone, Debug, Serialize)]
pub struct Stats {
    pub backend: &'static str,
    pub profile: Profile,
    pub model_load_ms: f64,
    pub checkpoint_bytes: u64,
    pub rss_bytes: u64,
    pub milliseconds: BTreeMap<String, f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accelerator_allocated_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accelerator_driver_bytes: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Output {
    pub text: String,
    pub concept_codes: Vec<i64>,
    pub token_ids: Vec<i64>,
    pub stats: Stats,
}

pub struct Engine {
    pub config: Config,
    pub device: Device,
    pub profile: Profile,
    pub vocabulary: Vocabulary,
    pub neural: NeuralTokenizer,
    pub generator: Generator,
    pub generator_weights: Option<Weights>,
    policy: TextPolicy,
    checkpoint_bytes: u64,
    load_ms: f64,
}

impl Engine {
    pub fn load(
        directory: &Path,
        vocabulary: &Path,
        backend: Backend,
        profile: Profile,
    ) -> Result<Self> {
        let start = Instant::now();
        let device = device(backend, profile)?;
        ensure!(
            profile != Profile::Ultra || !matches!(device, Device::Cuda(_)),
            "Ultra-light supports CPU and MPS"
        );
        let config = Config::read(directory)?;
        let vocabulary = Vocabulary::load(vocabulary, config.model.prefix_pad_token_id)?;
        let neural_path = directory.join(&config.checkpoints.tokenizer);
        let generator_path = directory.join(&config.checkpoints.generator);
        let checkpoint_bytes = neural_path
            .metadata()
            .context("tokenizer checkpoint")?
            .len()
            + generator_path
                .metadata()
                .context("generator checkpoint")?
                .len();

        let neural = NeuralTokenizer::load(
            &Weights::read(&neural_path, "model", device)?,
            &config.tokenizer.vqvae,
            device,
            profile == Profile::Ultra,
        )?;
        let generator_weights = Weights::read(&generator_path, "model", device)?;
        let mut generator = Generator::load(
            &generator_weights,
            &config.model,
            &neural,
            config.levels(),
            device,
        )?;
        if profile == Profile::Ultra && device == Device::Cpu {
            generator.int8()?;
        }
        let generator_weights = (profile == Profile::Quality).then_some(generator_weights);

        let allowance = if let Some(spec) = &config.generation.frequency_allowance {
            ensure!(
                spec == &format!("{}:inference.token_unigram", config.checkpoints.generator),
                "unknown frequency allowance format"
            );
            Some(Weights::read(&generator_path, "inference", Device::Cpu)?.get("token_unigram")?)
        } else {
            None
        };
        let policy = TextPolicy::load(&config.generation, allowance)?;
        synchronize(device);
        let load_ms = start.elapsed().as_secs_f64() * 1000.;
        Ok(Self {
            config,
            device,
            profile,
            vocabulary,
            neural,
            generator,
            generator_weights,
            policy,
            checkpoint_bytes,
            load_ms,
        })
    }

    pub fn generate(
        &self,
        text: &str,
        sampling: Sampling,
        context: Option<usize>,
        detailed: bool,
    ) -> Result<Output> {
        let start = Instant::now();
        let max = self.config.model.prefix_len as usize;
        let limit = context.unwrap_or(max);
        ensure!(
            (1..=max).contains(&limit),
            "context limit must be between 1 and {max}"
        );
        sampling.validate(&self.config.model.vocab_size)?;
        let mut timing = Timings::new(self.device, detailed);
        let (text, codes, tokens) = no_grad(|| {
            let ids = timing.stage("preprocessing", |_| {
                self.vocabulary
                    .prefix(text, limit, self.config.task == "dialogue")
            })?;
            let prefix = timing.stage("input_transfer", |_| {
                let mut ids = ids;
                if self.profile == Profile::Quality {
                    ids.resize(max, self.vocabulary.pad);
                }
                let width = ids.len() as i64;
                let batch = if sampling.cfg_scale > 0. {
                    ids.push(self.vocabulary.eos);
                    ids.resize(width as usize * 2, self.vocabulary.pad);
                    2
                } else {
                    1
                };
                Tensor::from_slice(&ids)
                    .view([batch, width])
                    .to_device(self.device)
            });
            let (codes, latent) = timing.stage("concept_generation", |timing| {
                self.generator
                    .infer(&self.neural, &prefix, sampling, self.profile, timing)
            });
            let logits = timing.stage("neural_decoder", |_| self.neural.decode(&latent));
            let tokens = timing.stage("decoder_selection", |_| self.policy.select(&logits))?;
            let codes = timing.stage("output_transfer", |_| tensor_ids(&codes))?;
            let text = timing.stage("text_reconstruction", |_| {
                self.vocabulary.decode(&tokens, self.policy.cut_at_eos)
            })?;
            Ok::<_, anyhow::Error>((text, codes, tokens))
        })?;
        Ok(self.output(text, codes, tokens, timing, start))
    }

    pub fn reconstruct(&self, text: &str, detailed: bool) -> Result<Output> {
        let start = Instant::now();
        let mut timing = Timings::new(self.device, detailed);
        let (text, codes, tokens) = no_grad(|| {
            let width = self.config.tokenizer.vqvae.context_length as usize;
            let ids = timing.stage("preprocessing", |_| self.vocabulary.target(text, width))?;
            let ids = timing.stage("input_transfer", |_| {
                Tensor::from_slice(&ids).unsqueeze(0).to_device(self.device)
            });
            let encoded = timing.stage("neural_encoder", |_| self.neural.encode(&ids));
            let levels = timing.stage("rvq_search", |_| {
                self.neural
                    .search(&encoded, self.profile != Profile::Quality)
            });
            let latent = timing.stage("rvq_reconstruction", |_| {
                let mut sum = Tensor::zeros_like(&encoded);
                for (i, ids) in levels.iter().enumerate() {
                    sum += self.neural.contribution(ids, i);
                }
                sum
            });
            let tokens = timing.stage("neural_decoder", |_| {
                tensor_ids(&self.neural.decode(&latent).argmax(-1, false))
            })?;
            let codes = tensor_ids(&Tensor::cat(&levels, 1))?;
            let text = timing.stage("text_reconstruction", |_| {
                self.vocabulary.decode(&tokens, self.policy.cut_at_eos)
            })?;
            Ok::<_, anyhow::Error>((text, codes, tokens))
        })?;
        Ok(self.output(text, codes, tokens, timing, start))
    }

    fn output(
        &self,
        text: String,
        concept_codes: Vec<i64>,
        token_ids: Vec<i64>,
        mut timing: Timings,
        start: Instant,
    ) -> Output {
        synchronize(self.device);
        timing
            .ms
            .insert("total".into(), start.elapsed().as_secs_f64() * 1000.);
        let (allocated, driver) = mps_memory(self.device).unzip();
        Output {
            text,
            concept_codes,
            token_ids,
            stats: Stats {
                backend: backend_name(self.device),
                profile: self.profile,
                model_load_ms: self.load_ms,
                checkpoint_bytes: self.checkpoint_bytes,
                rss_bytes: peak_rss_bytes(),
                milliseconds: timing.ms,
                accelerator_allocated_bytes: allocated,
                accelerator_driver_bytes: driver,
            },
        }
    }
}

fn tensor_ids(x: &Tensor) -> Result<Vec<i64>> {
    Ok(Vec::<i64>::try_from(&x.to_device(Device::Cpu).view([-1]))?)
}

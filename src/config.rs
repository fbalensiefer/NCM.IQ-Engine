use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Config {
    pub task: String,
    pub checkpoints: Checkpoints,
    pub model: Model,
    pub tokenizer: Tokenizer,
    pub generation: Generation,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Checkpoints {
    pub generator: String,
    pub tokenizer: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Model {
    pub n_layer: usize,
    pub n_head: i64,
    pub n_embd: i64,
    pub prefix_len: i64,
    pub prefix_pad_token_id: i64,
    pub vocab_size: Vec<i64>,
    pub raw_token_prefix: bool,
    pub query_level0: bool,
    pub predict_level0: bool,
    pub attn_pattern: String,
    pub use_rope: bool,
    pub use_qk_norm: bool,
    pub rope_base: f64,
    pub use_level_encoding: bool,
    pub tie_heads_to_codebook: bool,
    #[serde(default)]
    pub l0_autoregressive: bool,
    #[serde(default)]
    pub contextual_prefix: bool,
    #[serde(default)]
    pub ctx_encoder: Feature,
    #[serde(default)]
    pub post_upsample_conv: Feature,
    #[serde(default)]
    pub use_swiglu: bool,
    #[serde(default)]
    pub use_geglu: bool,
    #[serde(default)]
    pub use_relu2: bool,
    #[serde(default)]
    pub parallel_block: bool,
    #[serde(default)]
    pub normformer: bool,
    #[serde(default)]
    pub use_layerscale: bool,
    #[serde(default)]
    pub use_rezero: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Feature {
    pub enabled: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Tokenizer {
    pub vqvae: Vqvae,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Vqvae {
    pub n_layers: i64,
    pub embed_dim: i64,
    pub in_vocab_size: i64,
    pub context_length: i64,
    pub compression_factor: i64,
    pub pre_quant_groupnorm: i64,
    pub use_qk_norm: bool,
    pub vqvae_vocab_size: Vec<i64>,
    pub vector_quantizer_config: Quantizer,
    #[serde(default)]
    pub enc_layers: usize,
    #[serde(default)]
    pub enc_conv_layers: usize,
    #[serde(default)]
    pub dec_positional: bool,
    #[serde(default = "one")]
    pub dec_layers: usize,
}

fn one() -> usize {
    1
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Quantizer {
    pub clss: String,
    pub learned_l1_sampling: bool,
    pub learned_l1_cascade: bool,
    pub learned_l1_cascade_base: i64,
    pub learned_l1_cascade_norm: bool,
    pub learned_l1_cascade_norm_final: bool,
    pub learned_all_sampling: bool,
    pub quant_resi: QuantResidual,
    pub levels: Levels,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct QuantResidual {
    pub enabled: bool,
    pub ratio: f64,
    pub share_mode: i64,
    pub learnable_ratio: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Levels {
    pub use_manual_levels: bool,
    pub manual_levels: Vec<i64>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct Sampling {
    pub temperature: f64,
    pub top_k: i64,
    pub top_p: f64,
    pub cfg_scale: f64,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 1.,
            top_k: 1,
            top_p: 0.,
            cfg_scale: 0.,
        }
    }
}

impl Sampling {
    pub fn validate(&self, vocab: &[i64]) -> Result<()> {
        ensure!(
            self.temperature.is_finite()
                && (0.0..=10.0).contains(&self.temperature)
                && (0..=*vocab.iter().min().context("no codebooks")?).contains(&self.top_k)
                && (0.0..=1.0).contains(&self.top_p)
                && (0.0..=8.0).contains(&self.cfg_scale),
            "invalid concept sampling settings"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Generation {
    #[serde(flatten)]
    pub sampling: Sampling,
    #[serde(default)]
    pub frequency_penalty: f64,
    #[serde(default)]
    pub frequency_deficit: f64,
    #[serde(default)]
    pub adjacency_penalty: String,
    #[serde(default)]
    pub frequency_allowance: Option<String>,
    #[serde(default = "enabled")]
    pub cut_at_eos: bool,
}

fn enabled() -> bool {
    true
}

impl Generation {
    pub fn dialogue() -> Self {
        Self {
            sampling: Sampling::default(),
            frequency_penalty: 0.,
            frequency_deficit: 0.,
            adjacency_penalty: String::new(),
            frequency_allowance: None,
            cut_at_eos: true,
        }
    }
}

impl Config {
    pub fn read(dir: &Path) -> Result<Self> {
        let file = dir.join("config.yaml");
        let cfg: Self =
            serde_yaml::from_slice(&fs::read(&file).with_context(|| file.display().to_string())?)
                .with_context(|| format!("invalid {}", file.display()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        let m = &self.model;
        let v = &self.tokenizer.vqvae;
        let q = &v.vector_quantizer_config;
        let scales = &q.levels.manual_levels;
        ensure!(
            matches!(self.task.as_str(), "xsum" | "dialogue"),
            "unsupported checkpoint task"
        );
        ensure!(
            m.raw_token_prefix
                && m.query_level0
                && !m.predict_level0
                && !m.l0_autoregressive
                && !m.contextual_prefix
                && !m.ctx_encoder.enabled
                && !m.post_upsample_conv.enabled
                && !m.use_swiglu
                && !m.use_geglu
                && !m.use_relu2
                && !m.parallel_block
                && !m.normformer
                && !m.use_layerscale
                && !m.use_rezero
                && m.attn_pattern == "block_diagonal"
                && m.use_rope
                && m.use_qk_norm
                && m.use_level_encoding
                && m.tie_heads_to_codebook,
            "checkpoint does not use the released XSum generator architecture"
        );
        ensure!(
            q.clss == "multiscale_residual_vector_quantizer"
                && q.levels.use_manual_levels
                && q.learned_l1_sampling
                && q.learned_l1_cascade
                && q.learned_l1_cascade_base == 4
                && q.learned_l1_cascade_norm
                && q.learned_l1_cascade_norm_final
                && !q.learned_all_sampling
                && q.quant_resi.enabled
                && q.quant_resi.share_mode == 0
                && !q.quant_resi.learnable_ratio
                && v.enc_layers == 0
                && v.enc_conv_layers == 0
                && v.dec_layers == 1
                && !v.dec_positional
                && v.compression_factor == 1
                && v.use_qk_norm,
            "checkpoint does not use the released XSum neural tokenizer architecture"
        );
        ensure!(
            scales.first() == Some(&4)
                && scales.last() == Some(&48)
                && scales.windows(2).all(|p| p[0] < p[1])
                && v.context_length == 48
                && m.prefix_len > 0
                && scales.len() == m.vocab_size.len()
                && m.vocab_size == v.vqvae_vocab_size
                && m.vocab_size.iter().all(|&n| n > 0)
                && m.n_layer > 0
                && m.n_head > 0
                && m.n_embd > 0
                && m.n_embd % (2 * m.n_head) == 0
                && v.n_layers > 0
                && v.embed_dim > 0
                && v.embed_dim % v.n_layers == 0
                && v.pre_quant_groupnorm > 0
                && v.embed_dim % v.pre_quant_groupnorm == 0
                && v.embed_dim % 8 == 0,
            "inconsistent model dimensions"
        );
        for name in [&self.checkpoints.generator, &self.checkpoints.tokenizer] {
            ensure!(
                Path::new(name).file_name().and_then(|x| x.to_str()) == Some(name.as_str()),
                "checkpoint filenames must be relative to the model directory"
            );
        }
        self.generation.sampling.validate(&m.vocab_size)
    }

    pub fn levels(&self) -> &[i64] {
        &self
            .tokenizer
            .vqvae
            .vector_quantizer_config
            .levels
            .manual_levels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verifies_checkpoint_structure_and_sampling() -> Result<()> {
        let config: Config = serde_yaml::from_str(include_str!("../configs/xsum.yaml"))?;
        config.validate()?;
        assert_eq!(config.levels(), [4, 6, 10, 14, 20, 26, 32, 40, 48]);
        assert_eq!(config.generation.sampling.top_k, 1);
        let invalid = Sampling {
            top_k: 4097,
            ..Sampling::default()
        };
        assert!(invalid.validate(&config.model.vocab_size).is_err());
        Ok(())
    }
}

use anyhow::{Context, Result, anyhow, ensure};
use std::{collections::HashMap, path::Path};
use tch::{Device, Kind, Tensor};
use tokenizers::{AddedToken, Tokenizer};

use crate::config::Generation;

pub struct Vocabulary {
    inner: Tokenizer,
    pub eos: i64,
    pub pad: i64,
}

impl Vocabulary {
    pub fn load(path: &Path, expected_pad: i64) -> Result<Self> {
        let mut inner = Tokenizer::from_file(path.join("tokenizer.json"))
            .map_err(|e| anyhow!("reading text vocabulary: {e}"))?;
        inner
            .add_special_tokens([
                AddedToken::from("<|endoftext|>", true),
                AddedToken::from("<|pad|>", true),
            ])
            .map_err(|e| anyhow!("registering special text tokens: {e}"))?;
        let eos = i64::from(
            inner
                .token_to_id("<|endoftext|>")
                .context("missing EOS token")?,
        );
        let pad = i64::from(
            inner
                .token_to_id("<|pad|>")
                .context("missing padding token")?,
        );
        ensure!(
            pad == expected_pad,
            "text vocabulary padding differs from the checkpoint"
        );
        Ok(Self { inner, eos, pad })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let ids = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow!("encoding text: {e}"))?;
        Ok(ids.get_ids().iter().map(|&id| i64::from(id)).collect())
    }

    pub fn prefix(&self, text: &str, limit: usize, recent: bool) -> Result<Vec<i64>> {
        let mut ids = self.encode(text)?;
        if ids.len() > limit {
            if recent {
                ids.drain(..ids.len() - limit);
            } else {
                ids.truncate(limit);
            }
        }
        if ids.is_empty() {
            ids.push(self.eos);
        }
        Ok(ids)
    }

    pub fn target(&self, text: &str, width: usize) -> Result<Vec<i64>> {
        let mut ids = self.encode(text)?;
        ids.truncate(width - 1);
        ids.push(self.eos);
        ids.resize(width, self.pad);
        Ok(ids)
    }

    pub fn decode(&self, ids: &[i64], cut_at_eos: bool) -> Result<String> {
        let count = if cut_at_eos {
            ids.iter()
                .position(|&id| id == self.eos)
                .unwrap_or(ids.len())
        } else {
            ids.len()
        };
        let ids: Vec<u32> = ids[..count]
            .iter()
            .copied()
            .filter(|&id| id != self.pad && id != self.eos)
            .map(|id| id as u32)
            .collect();
        self.inner
            .decode(&ids, true)
            .map_err(|e| anyhow!("decoding text: {e}"))
    }
}

pub struct TextPolicy {
    frequency: f64,
    deficit: f64,
    allowance: Option<Vec<f64>>,
    adjacency: Vec<f64>,
    pub cut_at_eos: bool,
}

impl TextPolicy {
    pub fn load(config: &Generation, allowance: Option<Tensor>) -> Result<Self> {
        let fields: Vec<f64> = if config.adjacency_penalty.is_empty() {
            Vec::new()
        } else {
            config
                .adjacency_penalty
                .split(',')
                .map(str::parse)
                .collect::<std::result::Result<_, _>>()?
        };
        ensure!(
            fields.len() <= 3 && fields.iter().all(|x| x.is_finite()),
            "invalid adjacency penalty"
        );
        let adjacency = fields
            .first()
            .map(|&alpha| {
                let width = fields.get(1).copied().unwrap_or(1.).clamp(0., 48.) as usize;
                let decay = fields.get(2).copied().unwrap_or(1.);
                (0..width).map(|j| alpha * decay.powi(j as i32)).collect()
            })
            .unwrap_or_default();
        let allowance = allowance
            .map(|x| Vec::<f64>::try_from(&x.to_kind(Kind::Double).view([-1])))
            .transpose()?;
        Ok(Self {
            frequency: config.frequency_penalty,
            deficit: config.frequency_deficit,
            allowance,
            adjacency,
            cut_at_eos: config.cut_at_eos,
        })
    }

    pub fn select(&self, logits: &Tensor) -> Result<Vec<i64>> {
        if self.frequency <= 0. && self.adjacency.is_empty() {
            return Ok(Vec::<i64>::try_from(
                &logits.argmax(-1, false).to_device(Device::Cpu).view([-1]),
            )?);
        }
        let shape = logits.size();
        let (steps, width) = (shape[1] as usize, shape[2] as usize);
        let logits = Vec::<f32>::try_from(
            &logits
                .to_device(Device::Cpu)
                .to_kind(Kind::Float)
                .view([-1]),
        )?;
        let allowance = self
            .allowance
            .as_ref()
            .map(|a| a.iter().map(|&v| v * steps as f64).collect::<Vec<_>>());
        ensure!(
            allowance.as_ref().is_none_or(|a| a.len() >= width),
            "short frequency allowance"
        );
        let mut counts = HashMap::<usize, usize>::new();
        let mut history = Vec::<usize>::new();
        let mut scores = vec![0.0_f64; width];
        for (t, row) in logits.chunks_exact(width).enumerate() {
            let coefficient = self.deficit * ((steps - t) as f64 / steps as f64);
            if let Some(allowance) = &allowance {
                for ((score, &raw), &quota) in scores.iter_mut().zip(row).zip(allowance) {
                    *score = f64::from(raw) + coefficient * quota;
                }
            } else {
                for (score, &raw) in scores.iter_mut().zip(row) {
                    *score = f64::from(raw);
                }
            }
            for (&id, &count) in &counts {
                let quota = allowance.as_ref().map_or(0., |a| a[id]);
                scores[id] = f64::from(row[id]) - self.frequency * (count as f64 - quota).max(0.);
                if allowance.is_some() {
                    scores[id] += coefficient * (quota - count as f64).max(0.);
                }
            }
            for (&id, penalty) in history.iter().rev().zip(&self.adjacency) {
                scores[id] -= penalty;
            }
            let mut best = 0;
            for i in 1..width {
                if scores[i] > scores[best] {
                    best = i;
                }
            }
            *counts.entry(best).or_default() += 1;
            history.push(best);
        }
        Ok(history.into_iter().map(|id| id as i64).collect())
    }
}

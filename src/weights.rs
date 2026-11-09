use anyhow::{Context, Result, bail, ensure};
use candle_core::pickle::PthTensors;
use std::{collections::HashMap, path::Path};
use tch::{Device, Kind, Tensor};

pub struct Weights(HashMap<String, Tensor>);

impl Weights {
    pub fn read(path: &Path, key: &str, device: Device) -> Result<Self> {
        let mut tensors = HashMap::new();
        if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            for (name, value) in Tensor::read_safetensors(path)? {
                tensors.insert(name, value.to_device(device));
            }
        } else {
            let archive = PthTensors::new(path, Some(key))
                .with_context(|| format!("reading {} field {key}", path.display()))?;
            for name in archive.tensor_infos().keys() {
                if name.starts_with("vq.ema_")
                    || name.starts_with("vq.codebook_hit_")
                    || matches!(name.as_str(), "lvl_1L" | "attn_mask_for_masking")
                {
                    continue;
                }
                let raw = archive
                    .get(name)?
                    .with_context(|| format!("missing tensor {name}"))?;
                let shape: Vec<i64> = raw.dims().iter().map(|&x| x as i64).collect();
                let raw = raw.flatten_all()?;
                let value = match raw.dtype() {
                    candle_core::DType::F32 => Tensor::from_slice(&raw.to_vec1::<f32>()?),
                    candle_core::DType::F64 => Tensor::from_slice(&raw.to_vec1::<f64>()?),
                    dtype => bail!("unexpected dtype {dtype:?} for {name}"),
                };
                tensors.insert(
                    name.to_owned(),
                    value.view(shape.as_slice()).to_device(device),
                );
            }
        }
        ensure!(!tensors.is_empty(), "empty checkpoint field {key}");
        Ok(Self(tensors))
    }

    pub fn get(&self, name: &str) -> Result<Tensor> {
        self.0
            .get(name)
            .map(Tensor::shallow_clone)
            .with_context(|| format!("missing tensor {name}"))
    }

    pub fn optional(&self, name: &str) -> Option<Tensor> {
        self.0.get(name).map(Tensor::shallow_clone)
    }

    pub fn trainable(&self, full: bool) -> Vec<Tensor> {
        let mut names: Vec<_> = self
            .0
            .keys()
            .filter(|n| full || n.starts_with("lm_heads."))
            .collect();
        names.sort();
        names
            .into_iter()
            .map(|name| self.0[name].set_requires_grad(true))
            .collect()
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let tensors: Vec<_> = self
            .0
            .iter()
            .map(|(name, tensor)| (name, tensor.to_device(Device::Cpu).to_kind(Kind::Float)))
            .collect();
        Tensor::write_safetensors(&tensors, path)?;
        Ok(())
    }
}

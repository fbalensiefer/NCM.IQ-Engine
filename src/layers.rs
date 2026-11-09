use anyhow::Result;
use tch::{Device, Kind, Tensor};

use crate::{device::Timings, int8::Int8, weights::Weights};

pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
    packed: Option<Int8>,
}

impl Linear {
    pub fn load(weights: &Weights, name: &str) -> Result<Self> {
        Ok(Self {
            weight: weights.get(&format!("{name}.weight"))?,
            bias: weights.optional(&format!("{name}.bias")),
            packed: None,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        if let Some(packed) = &self.packed {
            return packed.apply(x);
        }
        if x.kind() == self.weight.kind() {
            x.linear(&self.weight, self.bias.as_ref())
        } else {
            x.to_kind(self.weight.kind())
                .linear(&self.weight, self.bias.as_ref())
        }
    }

    pub fn int8(&mut self) -> Result<()> {
        self.packed = Some(Int8::new(&self.weight, self.bias.as_ref())?);
        self.weight = Tensor::new();
        self.bias = None;
        Ok(())
    }

    pub fn half(&mut self) {
        self.weight = self.weight.to_kind(Kind::Half);
        self.bias = self.bias.as_ref().map(|b| b.to_kind(Kind::Half));
    }
}

pub struct Norm {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Norm {
    pub fn load(weights: &Weights, name: &str) -> Result<Self> {
        Ok(Self {
            weight: weights.get(&format!("{name}.weight"))?,
            bias: weights.optional(&format!("{name}.bias")),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.layer_norm(
            self.weight.size(),
            Some(&self.weight),
            self.bias.as_ref(),
            1e-6,
            true,
        )
    }
}

pub struct Conv {
    weight: Tensor,
    bias: Tensor,
}

impl Conv {
    pub fn load(weights: &Weights, name: &str) -> Result<Self> {
        Ok(Self {
            weight: weights.get(&format!("{name}.weight"))?,
            bias: weights.get(&format!("{name}.bias"))?,
        })
    }

    pub fn forward(&self, x: &Tensor, stride: i64, padding: i64) -> Tensor {
        x.conv1d(&self.weight, Some(&self.bias), [stride], [padding], [1], 1)
    }

    pub fn transpose(&self, x: &Tensor, stride: i64) -> Tensor {
        x.conv_transpose1d(&self.weight, Some(&self.bias), [stride], [0], [0], 1, [1])
    }
}

pub fn group_norm(x: &Tensor, groups: i64) -> Tensor {
    x.group_norm(groups, None::<&Tensor>, None::<&Tensor>, 1e-5, true)
}

pub fn embedding(weight: &Tensor, ids: &Tensor) -> Tensor {
    Tensor::embedding(weight, ids, -1, false, false)
}

fn rms(x: &Tensor, weight: &Tensor) -> Tensor {
    x * (x.square().mean_dim([-1].as_slice(), true, Kind::Float) + 1e-6).rsqrt() * weight
}

pub struct Rope {
    cos: Tensor,
    sin: Tensor,
}

impl Rope {
    pub fn new(width: i64, head_dim: i64, base: f64, device: Device) -> Self {
        let half = head_dim / 2;
        let inv = Tensor::pow_scalar(
            base,
            &(Tensor::arange(half, (Kind::Float, Device::Cpu)) / half as f64),
        )
        .reciprocal();
        let freq =
            Tensor::arange(width, (Kind::Float, Device::Cpu)).unsqueeze(1) * inv.unsqueeze(0);
        let freq = Tensor::cat(&[&freq, &freq], 1);
        Self {
            cos: freq.cos().to_device(device),
            sin: freq.sin().to_device(device),
        }
    }

    pub fn select(&self, positions: &Tensor) -> Self {
        Self {
            cos: self.cos.index_select(0, positions),
            sin: self.sin.index_select(0, positions),
        }
    }

    pub fn apply(&self, x: &Tensor) -> Tensor {
        let half = x.size()[3] / 2;
        let rotated = Tensor::cat(&[x.narrow(-1, half, half).neg(), x.narrow(-1, 0, half)], -1);
        x * &self.cos + rotated * &self.sin
    }
}

pub struct Kv {
    k: Tensor,
    v: Tensor,
}

impl Kv {
    pub fn prefix(self, width: i64) -> Self {
        Self {
            k: self.k.narrow(2, 0, width),
            v: self.v.narrow(2, 0, width),
        }
    }
}

pub struct Block {
    norm_attn: Norm,
    norm_mlp: Norm,
    qkv: Linear,
    attention_out: Linear,
    up: Linear,
    down: Linear,
    q_norm: Tensor,
    k_norm: Tensor,
    heads: i64,
}

impl Block {
    pub fn load(weights: &Weights, name: &str, heads: i64) -> Result<Self> {
        Ok(Self {
            norm_attn: Norm::load(weights, &format!("{name}.ln_1"))?,
            norm_mlp: Norm::load(weights, &format!("{name}.ln_2"))?,
            qkv: Linear::load(weights, &format!("{name}.attn.c_attn"))?,
            attention_out: Linear::load(weights, &format!("{name}.attn.c_proj"))?,
            up: Linear::load(weights, &format!("{name}.mlp.c_fc"))?,
            down: Linear::load(weights, &format!("{name}.mlp.c_proj"))?,
            q_norm: weights.get(&format!("{name}.attn.q_norm.weight"))?,
            k_norm: weights.get(&format!("{name}.attn.k_norm.weight"))?,
            heads,
        })
    }

    pub fn int8(&mut self) -> Result<()> {
        for linear in [
            &mut self.qkv,
            &mut self.attention_out,
            &mut self.up,
            &mut self.down,
        ] {
            linear.int8()?;
        }
        Ok(())
    }

    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        rope: Option<&Rope>,
        prefix: Option<&Kv>,
        dropout: f64,
        mut timing: Option<&mut Timings>,
    ) -> (Tensor, Kv) {
        let shape = x.size();
        let (batch, length, width) = (shape[0], shape[1], shape[2]);
        let normalized = Timings::block_stage(&mut timing, "block_normalization", || {
            self.norm_attn.forward(x)
        });
        let qkv = Timings::block_stage(&mut timing, "block_linear", || {
            self.qkv.forward(&normalized)
        })
        .split(width, -1);
        let heads = |x: &Tensor| {
            x.view([batch, length, self.heads, width / self.heads])
                .transpose(1, 2)
        };
        let rotate = |x: Tensor| match rope {
            Some(rope) => rope.apply(&x),
            None => x,
        };
        let (q, mut k) = Timings::block_stage(&mut timing, "block_qk_norm_rope", || {
            (
                rotate(rms(&heads(&qkv[0]), &self.q_norm)),
                rotate(rms(&heads(&qkv[1]), &self.k_norm)),
            )
        });
        let mut v = heads(&qkv[2]);
        if let Some(prefix) = prefix {
            (k, v) = Timings::block_stage(&mut timing, "block_kv_copy", || {
                (
                    Tensor::cat(&[&prefix.k, &k], 2),
                    Tensor::cat(&[&prefix.v, &v], 2),
                )
            });
        }
        let attended = Timings::block_stage(&mut timing, "block_attention", || {
            Tensor::scaled_dot_product_attention(&q, &k, &v, mask, dropout, false, None, false)
                .transpose(1, 2)
                .contiguous()
                .view([batch, length, width])
        });
        let projected = Timings::block_stage(&mut timing, "block_linear", || {
            self.attention_out.forward(&attended)
        });
        let x = x + projected.dropout(dropout, dropout > 0.);
        let normalized = Timings::block_stage(&mut timing, "block_normalization", || {
            self.norm_mlp.forward(&x)
        });
        let up = Timings::block_stage(&mut timing, "block_linear", || self.up.forward(&normalized));
        let activated = Timings::block_stage(&mut timing, "block_activation", || up.gelu("none"));
        let mlp = Timings::block_stage(&mut timing, "block_linear", || {
            self.down.forward(&activated)
        });
        (x + mlp.dropout(dropout, dropout > 0.), Kv { k, v })
    }
}

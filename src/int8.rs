use anyhow::{Context, Result};
use std::{
    ffi::c_void,
    ptr::{NonNull, null},
};
use tch::Tensor;

unsafe extern "C" {
    fn ncm_int8_pack(weight: *const c_void, bias: *const c_void) -> *mut c_void;
    fn ncm_int8_linear(packed: *mut c_void, input: *const c_void) -> *mut c_void;
    fn ncm_int8_free(packed: *mut c_void);
}

pub struct Int8(NonNull<c_void>);

// Owns the packed LibTorch object. The API serializes access to the model.
unsafe impl Send for Int8 {}

impl Int8 {
    pub fn new(weight: &Tensor, bias: Option<&Tensor>) -> Result<Self> {
        let bias = bias.map_or(null(), |b| b.as_ptr().cast());
        let raw = unsafe { ncm_int8_pack(weight.as_ptr().cast(), bias) };
        Ok(Self(NonNull::new(raw).context(
            "LibTorch could not pack an INT8 linear weight",
        )?))
    }

    pub fn apply(&self, input: &Tensor) -> Tensor {
        let raw = unsafe { ncm_int8_linear(self.0.as_ptr(), input.as_ptr().cast()) };
        assert!(!raw.is_null(), "LibTorch dynamic INT8 linear failed");
        unsafe { Tensor::from_ptr(raw.cast()) }
    }
}

impl Drop for Int8 {
    fn drop(&mut self) {
        unsafe { ncm_int8_free(self.0.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_int8_linear_tracks_float_linear() -> Result<()> {
        let weight = Tensor::from_slice(&[1_f32, 2., -3., 2., 1., 0.]).view([2, 3]);
        let bias = Tensor::from_slice(&[0.5_f32, 1.]);
        let input = Tensor::from_slice(&[1_f32, -2., 3., 2., 1., 1.]).view([2, 3]);
        let packed = Int8::new(&weight, Some(&bias))?;
        let error = (packed.apply(&input) - input.linear(&weight, Some(&bias)))
            .abs()
            .max()
            .double_value(&[]);
        assert!(error < 0.2, "INT8 maximum error {error}");
        Ok(())
    }
}

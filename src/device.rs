use anyhow::{Result, bail};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, time::Instant};
use tch::{Cuda, Device};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    #[default]
    Auto,
    Cpu,
    Mps,
    Cuda,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Profile {
    Quality,
    #[default]
    Balanced,
    Ultra,
}

pub fn backends() -> Vec<Backend> {
    let mut result = vec![Backend::Auto, Backend::Cpu];
    if tch::utils::has_mps() {
        result.push(Backend::Mps);
    }
    if Cuda::is_available() {
        result.push(Backend::Cuda);
    }
    result
}

pub fn device(backend: Backend, profile: Profile) -> Result<Device> {
    match backend {
        Backend::Auto if Cuda::is_available() && profile != Profile::Ultra => Ok(Device::Cuda(0)),
        Backend::Auto | Backend::Mps if tch::utils::has_mps() => Ok(Device::Mps),
        Backend::Auto | Backend::Cpu => Ok(Device::Cpu),
        Backend::Cuda if Cuda::is_available() => Ok(Device::Cuda(0)),
        _ => bail!("requested backend is unavailable"),
    }
}

pub fn backend_name(device: Device) -> &'static str {
    match device {
        Device::Mps => "mps",
        Device::Cuda(_) => "cuda",
        _ => "cpu",
    }
}

#[cfg(ncm_mps)]
unsafe extern "C" {
    fn ncm_mps_synchronize();
    fn ncm_mps_allocated() -> usize;
    fn ncm_mps_driver_allocated() -> usize;
}

pub fn synchronize(device: Device) {
    match device {
        Device::Cuda(index) => Cuda::synchronize(index as i64),
        #[cfg(ncm_mps)]
        Device::Mps => unsafe { ncm_mps_synchronize() },
        _ => {}
    }
}

pub fn mps_memory(device: Device) -> Option<(usize, usize)> {
    #[cfg(ncm_mps)]
    if device == Device::Mps {
        return Some(unsafe { (ncm_mps_allocated(), ncm_mps_driver_allocated()) });
    }
    let _ = device;
    None
}

pub fn peak_rss_bytes() -> u64 {
    // getrusage reports bytes on macOS and KiB on Linux.
    let mut usage = std::mem::MaybeUninit::uninit();
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return 0;
    }
    let value = unsafe { usage.assume_init() }.ru_maxrss as u64;
    if cfg!(target_os = "macos") {
        value
    } else {
        value * 1024
    }
}

pub struct Timings {
    device: Device,
    detailed: bool,
    pub ms: BTreeMap<String, f64>,
}

impl Timings {
    pub fn new(device: Device, detailed: bool) -> Self {
        Self {
            device,
            detailed,
            ms: BTreeMap::new(),
        }
    }

    pub fn stage<T>(&mut self, label: impl Into<String>, run: impl FnOnce(&mut Self) -> T) -> T {
        if self.detailed {
            synchronize(self.device);
        }
        let start = Instant::now();
        let result = run(self);
        if self.detailed {
            synchronize(self.device);
        }
        *self.ms.entry(label.into()).or_default() += start.elapsed().as_secs_f64() * 1000.;
        result
    }
}

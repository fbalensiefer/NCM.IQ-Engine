use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::Path};
use tch::manual_seed;

use crate::{
    download::digest,
    engine::{Engine, Output},
};

pub const PROMPTS: &str = include_str!("../benchmarks/prompts.json");

fn stats(values: &mut [f64]) -> Value {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    let median = (values[(n - 1) / 2] + values[n / 2]) * 0.5;
    let p95 = values[(n * 95).div_ceil(100) - 1];
    json!({"median": median, "p95": p95})
}

fn stages(results: &[Output]) -> BTreeMap<&str, Value> {
    let mut all: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for result in results {
        let mut transformer = 0.;
        for (key, value) in &result.stats.milliseconds {
            all.entry(key).or_default().push(*value);
            if key.starts_with("transformer_") {
                transformer += value;
            }
        }
        if transformer > 0. {
            all.entry("transformer").or_default().push(transformer);
        }
    }
    all.into_iter()
        .map(|(key, mut values)| (key, stats(&mut values)))
        .collect()
}

fn signature(result: &Output) -> Result<Value> {
    Ok(json!({
        "concept_sha256": digest(&serde_json::to_vec(&result.concept_codes)?),
        "token_sha256": digest(&serde_json::to_vec(&result.token_ids)?),
        "text": result.text,
    }))
}

pub fn run(
    engine: &Engine,
    prompts: &[u8],
    warmup: usize,
    runs: usize,
    seed: i64,
    threads: i32,
) -> Result<Value> {
    let items: Vec<String> = serde_json::from_slice(prompts)?;
    ensure!(
        !items.is_empty() && items.iter().all(|t| !t.trim().is_empty()),
        "prompts must contain nonempty text"
    );
    ensure!(runs > 0, "runs must be positive");
    let sampling = engine.config.generation.sampling;
    manual_seed(seed);
    let first = engine.generate(&items[0], sampling, None, false)?;
    for i in 0..warmup {
        engine.generate(&items[i % items.len()], sampling, None, false)?;
    }
    let mut latencies = Vec::with_capacity(runs);
    for i in 0..runs {
        manual_seed(seed.wrapping_add(i as i64));
        latencies.push(
            engine
                .generate(&items[i % items.len()], sampling, None, false)?
                .stats
                .milliseconds["total"],
        );
    }
    let mut generation = Vec::with_capacity(runs);
    let mut reconstruction = Vec::with_capacity(runs);
    for i in 0..runs {
        manual_seed(seed.wrapping_add(i as i64));
        let prompt = &items[i % items.len()];
        generation.push(engine.generate(prompt, sampling, None, true)?);
        reconstruction.push(engine.reconstruct(prompt, true)?);
    }
    let metadata = &reconstruction.last().unwrap().stats;
    let rate = runs as f64 * 1000. / latencies.iter().sum::<f64>();
    Ok(json!({
        "runtime": "rust",
        "libtorch": "2.13.0",
        "device": metadata.backend,
        "profile": metadata.profile,
        "task": engine.config.task,
        "threads": threads,
        "seed": seed,
        "warmup": warmup,
        "runs": runs,
        "prompts_sha256": digest(prompts),
        "checkpoint_bytes": metadata.checkpoint_bytes,
        "load_ms": metadata.model_load_ms,
        "first_generation_ms": first.stats.milliseconds["total"],
        "load_and_first_ms": metadata.model_load_ms + first.stats.milliseconds["total"],
        "steady_latency_ms": stats(&mut latencies),
        "requests_per_second": rate,
        "peak_rss_bytes": metadata.rss_bytes,
        "mps_current_allocated_bytes": metadata.accelerator_allocated_bytes,
        "mps_driver_allocated_bytes": metadata.accelerator_driver_bytes,
        "generation_stage_ms": stages(&generation),
        "reconstruction_stage_ms": stages(&reconstruction),
        "generation": generation.iter().map(signature).collect::<Result<Vec<_>>>()?,
        "reconstruction": reconstruction.iter().map(signature).collect::<Result<Vec<_>>>()?,
    }))
}

pub fn compare(a: &Path, b: &Path) -> Result<()> {
    let a: Value = serde_json::from_slice(&fs::read(a)?)?;
    let b: Value = serde_json::from_slice(&fs::read(b)?)?;
    for key in [
        "device",
        "threads",
        "seed",
        "prompts_sha256",
        "task",
        "runs",
        "warmup",
    ] {
        ensure!(
            !a[key].is_null() && a[key] == b[key],
            "incompatible benchmark parameter {key}"
        );
    }
    println!("| Metric | Baseline ms | Candidate ms | Speedup |\n|---|---:|---:|---:|");
    for (label, pointer) in [
        ("Full generation", "/steady_latency_ms/median"),
        ("Transformer", "/generation_stage_ms/transformer/median"),
        (
            "Decoder forward",
            "/generation_stage_ms/neural_decoder/median",
        ),
        (
            "Decoder token selection",
            "/generation_stage_ms/decoder_selection/median",
        ),
        ("Exact RVQ", "/reconstruction_stage_ms/rvq_search/median"),
        (
            "Full reconstruction",
            "/reconstruction_stage_ms/total/median",
        ),
    ] {
        let median = |x: &Value| {
            x.pointer(pointer)
                .and_then(Value::as_f64)
                .context("missing stage median")
        };
        let (base, candidate) = (median(&a)?, median(&b)?);
        println!(
            "| {label} | {base:.3} | {candidate:.3} | {:.2}× |",
            base / candidate
        );
    }
    for kind in ["generation", "reconstruction"] {
        let rows = |x: &Value| x[kind].as_array().cloned().context("missing output hashes");
        let (a, b) = (rows(&a)?, rows(&b)?);
        ensure!(a.len() == b.len(), "different output counts");
        let matches = |key: &str| {
            a.iter()
                .zip(&b)
                .filter(|(a, b)| !a[key].is_null() && a[key] == b[key])
                .count()
        };
        println!(
            "{kind}: {}/{} identical concept arrays, {}/{} identical token arrays",
            matches("concept_sha256"),
            a.len(),
            matches("token_sha256"),
            a.len()
        );
    }
    Ok(())
}

use anyhow::{Context, Result, ensure};
use std::{
    collections::HashMap,
    fs,
    io::{BufRead, BufReader},
    path::Path,
};
use tch::{Device, Kind, Tensor, manual_seed, no_grad};
use uuid::Uuid;

use crate::{
    config::Generation,
    engine::Engine,
    session::{Turn, dialogue_prompt},
};

fn examples(path: &Path) -> Result<Vec<(String, String)>> {
    let mut result = Vec::new();
    for (index, line) in BufReader::new(fs::File::open(path)?).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let conversation: Vec<Turn> =
            serde_json::from_str(&line).with_context(|| format!("line {}", index + 1))?;
        ensure!(!conversation.is_empty(), "line {} has no turns", index + 1);
        for (i, turn) in conversation.iter().enumerate() {
            ensure!(
                !turn.user.trim().is_empty() && !turn.assistant.trim().is_empty(),
                "empty turn in line {}",
                index + 1
            );
            result.push((
                dialogue_prompt(&conversation[..i], &turn.user),
                turn.assistant.clone(),
            ));
        }
    }
    ensure!(!result.is_empty(), "training file has no examples");
    Ok(result)
}

fn encode(engine: &Engine, example: &(String, String)) -> Result<(Tensor, Vec<Tensor>)> {
    let prefix =
        engine
            .vocabulary
            .prefix(&example.0, engine.config.model.prefix_len as usize, true)?;
    let target = engine.vocabulary.target(
        &example.1,
        engine.config.tokenizer.vqvae.context_length as usize,
    )?;
    Ok(no_grad(|| {
        let prefix = Tensor::from_slice(&prefix)
            .unsqueeze(0)
            .to_device(engine.device);
        let target = Tensor::from_slice(&target)
            .unsqueeze(0)
            .to_device(engine.device);
        let codes = engine.neural.search(&engine.neural.encode(&target), true);
        (prefix, codes)
    }))
}

fn objective(engine: &Engine, hidden: &[Tensor], codes: &[Tensor]) -> Tensor {
    let mut terms = Vec::new();
    let mut width = 0;
    for (i, (x, y)) in hidden.iter().zip(codes).enumerate() {
        let logits = engine.generator.logits(x, &engine.neural, i).flatten(0, 1);
        let positions = y.numel();
        terms.push(logits.cross_entropy_for_logits(&y.flatten(0, -1)) * positions as f64);
        width += positions;
    }
    Tensor::stack(&terms, 0).sum(Kind::Float) / width as f64
}

struct AdamW {
    params: Vec<Tensor>,
    first: Vec<Tensor>,
    second: Vec<Tensor>,
    lr: f64,
}

impl AdamW {
    fn new(params: Vec<Tensor>, lr: f64) -> Self {
        let first = params.iter().map(Tensor::zeros_like).collect();
        let second = params.iter().map(Tensor::zeros_like).collect();
        Self {
            params,
            first,
            second,
            lr,
        }
    }

    fn update(&mut self, step: usize) {
        let bias1 = 1. - 0.9_f64.powf(step as f64);
        let bias2 = 1. - 0.999_f64.powf(step as f64);
        no_grad(|| {
            for ((p, first), second) in self
                .params
                .iter_mut()
                .zip(&mut self.first)
                .zip(&mut self.second)
            {
                let grad = p.grad();
                if !grad.defined() {
                    continue;
                }
                *first = &*first * 0.9 + &grad * 0.1;
                *second = &*second * 0.999 + grad.square() * 0.001;
                let delta = &*first / bias1 / ((&*second / bias2).sqrt() + 1e-8);
                let next = &*p * (1. - self.lr * 0.01) - delta * self.lr;
                p.copy_(&next);
                p.zero_grad();
            }
        });
    }
}

pub struct Options<'a> {
    pub source: &'a Path,
    pub data: &'a Path,
    pub output: &'a Path,
    pub steps: usize,
    pub learning_rate: f64,
    pub seed: i64,
    pub full: bool,
}

pub fn train(engine: &Engine, opts: Options<'_>) -> Result<()> {
    ensure!(
        opts.steps > 0 && (0.0..1.0).contains(&opts.learning_rate) && opts.learning_rate > 0.,
        "invalid optimizer options"
    );
    ensure!(!opts.output.exists(), "output directory already exists");
    let examples = examples(opts.data)?;
    manual_seed(opts.seed);
    let weights = engine
        .generator_weights
        .as_ref()
        .context("training requires Quality weights")?;
    let params = weights.trainable(opts.full);
    let mut optimizer = AdamW::new(params, opts.learning_rate);
    let mut cached = HashMap::new();
    let mut order = Vec::new();
    for step in 1..=opts.steps {
        if order.is_empty() {
            order = Vec::<i64>::try_from(&Tensor::randperm(
                examples.len() as i64,
                (Kind::Int64, Device::Cpu),
            ))?;
        }
        let i = order.pop().unwrap() as usize;
        let loss = if opts.full {
            let (prefix, codes) = encode(engine, &examples[i])?;
            let hidden = engine
                .generator
                .teacher_hidden(&engine.neural, &prefix, &codes, 0.3);
            objective(engine, &hidden, &codes)
        } else {
            if cached.len() == 128 && !cached.contains_key(&i) {
                cached.clear();
            }
            let (hidden, codes) = match cached.entry(i) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let (prefix, codes) = encode(engine, &examples[i])?;
                    let hidden = no_grad(|| {
                        engine
                            .generator
                            .teacher_hidden(&engine.neural, &prefix, &codes, 0.)
                    });
                    entry.insert((hidden, codes))
                }
            };
            objective(engine, hidden, codes)
        };
        loss.backward();
        optimizer.update(step);
        if step == 1 || step % 20 == 0 || step == opts.steps {
            let value = loss.double_value(&[]);
            ensure!(value.is_finite(), "training loss is non-finite");
            eprintln!("step={step} concept_ce={value:.4}");
        }
    }

    let parent = opts
        .output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let staging = parent.join(format!(".ncm-adapt-{}", Uuid::new_v4()));
    fs::create_dir(&staging)?;
    let result = (|| {
        let mut config = engine.config.clone();
        config.task = "dialogue".into();
        config.generation = Generation::dialogue();
        config.checkpoints.generator = "generator.safetensors".into();
        let tokenizer = &config.checkpoints.tokenizer;
        fs::copy(opts.source.join(tokenizer), staging.join(tokenizer))?;
        weights.save(&staging.join(&config.checkpoints.generator))?;
        fs::write(staging.join("config.yaml"), serde_yaml::to_string(&config)?)?;
        ensure!(!opts.output.exists(), "output directory already exists");
        fs::rename(&staging, opts.output)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(staging);
    }
    result
}

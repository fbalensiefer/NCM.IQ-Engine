use anyhow::{Result, ensure};
use clap::{Args, Parser, Subcommand};
use ncm_iq::{
    benchmark,
    device::{Backend, Profile},
    download,
    engine::Engine,
    server, train,
};
use std::{fs, path::PathBuf};

#[derive(Parser)]
#[command(about, version)]
struct Cli {
    #[arg(long, default_value_t = 4, global = true)]
    threads: i32,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Serve {
        #[arg(long, default_value = "artifacts")]
        models: PathBuf,
        #[arg(long, default_value = "artifacts/gpt2")]
        vocabulary: PathBuf,
        #[arg(long, default_value = "sessions/chat.sqlite3")]
        sessions: PathBuf,
        #[arg(long, default_value_t = 8000)]
        port: u16,
    },
    Download {
        #[arg(long, default_value = "artifacts")]
        directory: PathBuf,
    },
    Generate {
        #[command(flatten)]
        model: ModelArgs,
        text: String,
    },
    Reconstruct {
        #[command(flatten)]
        model: ModelArgs,
        text: String,
    },
    Benchmark {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long)]
        prompts: Option<PathBuf>,
        #[arg(long, default_value_t = 2)]
        warmup: usize,
        #[arg(long, default_value_t = 12)]
        runs: usize,
        #[arg(long, default_value_t = 2026)]
        seed: i64,
        /// Collect synchronized timings inside the Transformer blocks during the diagnostic pass.
        #[arg(long)]
        transformer_breakdown: bool,
        #[arg(long)]
        output: PathBuf,
    },
    Compare {
        baseline: PathBuf,
        candidate: PathBuf,
    },
    Train {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long)]
        train: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 200)]
        steps: usize,
        #[arg(long, default_value_t = 0.001)]
        learning_rate: f64,
        #[arg(long, default_value_t = 2026)]
        seed: i64,
        #[arg(long)]
        full: bool,
    },
}

#[derive(Args)]
struct ModelArgs {
    #[arg(long, default_value = "artifacts/xsum")]
    checkpoint: PathBuf,
    #[arg(long, default_value = "artifacts/gpt2")]
    vocabulary: PathBuf,
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    backend: Backend,
    #[arg(long, value_enum, default_value_t = Profile::Balanced)]
    profile: Profile,
}

impl ModelArgs {
    fn load(&self) -> Result<Engine> {
        Engine::load(
            &self.checkpoint,
            &self.vocabulary,
            self.backend,
            self.profile,
        )
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    ensure!(cli.threads > 0, "threads must be positive");
    tch::set_num_threads(cli.threads);
    match cli.command {
        Command::Serve {
            models,
            vocabulary,
            sessions,
            port,
        } => server::serve(models, vocabulary, sessions, port).await?,
        Command::Download { directory } => download::download(&directory)?,
        Command::Generate { model, text } => {
            let engine = model.load()?;
            let output = engine.generate(&text, engine.config.generation.sampling, None, true)?;
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        Command::Reconstruct { model, text } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&model.load()?.reconstruct(&text, true)?)?
            );
        }
        Command::Benchmark {
            model,
            prompts,
            warmup,
            runs,
            seed,
            transformer_breakdown,
            output,
        } => {
            let prompts = prompts
                .map(fs::read)
                .transpose()?
                .unwrap_or_else(|| benchmark::PROMPTS.as_bytes().to_vec());
            let report = benchmark::run(
                &model.load()?,
                &prompts,
                warmup,
                runs,
                seed,
                cli.threads,
                transformer_breakdown,
            )?;
            if let Some(dir) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
                fs::create_dir_all(dir)?;
            }
            fs::write(output, serde_json::to_string_pretty(&report)? + "\n")?;
            println!("{}", report["steady_latency_ms"]);
        }
        Command::Compare {
            baseline,
            candidate,
        } => benchmark::compare(&baseline, &candidate)?,
        Command::Train {
            mut model,
            train,
            output,
            steps,
            learning_rate,
            seed,
            full,
        } => {
            model.profile = Profile::Quality;
            let engine = model.load()?;
            train::train(
                &engine,
                train::Options {
                    source: &model.checkpoint,
                    data: &train,
                    output: &output,
                    steps,
                    learning_rate,
                    seed,
                    full,
                },
            )?;
        }
    }
    Ok(())
}

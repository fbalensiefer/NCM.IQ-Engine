use anyhow::Result;
use ncm_chat::{
    benchmark::PROMPTS,
    device::{Backend, Profile},
    download::digest,
    engine::Engine,
};
use std::path::PathBuf;

#[test]
#[ignore = "requires the public XSum checkpoints in artifacts/ or NCM_TEST_ARTIFACTS"]
fn official_xsum_cpu_and_mps() -> Result<()> {
    tch::set_num_threads(4);
    let directory = std::env::var_os("NCM_TEST_ARTIFACTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| "artifacts".into());
    let checkpoint = directory.join("xsum");
    let vocabulary = directory.join("gpt2");
    let prompts: Vec<String> = serde_json::from_str(PROMPTS)?;
    let references: serde_json::Value =
        serde_json::from_str(include_str!("../docs/results/rust/cpu-quality.json"))?;
    let mut backends = vec![Backend::Cpu];
    if tch::utils::has_mps() {
        backends.push(Backend::Mps);
    }
    for backend in backends {
        let engine = Engine::load(&checkpoint, &vocabulary, backend, Profile::Balanced)?;
        for (i, prompt) in prompts.iter().enumerate() {
            let generated =
                engine.generate(prompt, engine.config.generation.sampling, None, true)?;
            let reconstructed = engine.reconstruct(prompt, true)?;
            for (kind, result) in [("generation", generated), ("reconstruction", reconstructed)] {
                assert_eq!(result.concept_codes.len(), 200);
                assert_eq!(result.token_ids.len(), 48);
                assert!(!result.text.is_empty());
                if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                    assert_eq!(
                        digest(&serde_json::to_vec(&result.concept_codes)?),
                        references[kind][i]["concept_sha256"].as_str().unwrap(),
                        "{kind} concept codes for prompt {i} on {backend:?}"
                    );
                    assert_eq!(
                        digest(&serde_json::to_vec(&result.token_ids)?),
                        references[kind][i]["token_sha256"].as_str().unwrap(),
                        "{kind} text IDs for prompt {i} on {backend:?}"
                    );
                    assert_eq!(result.text, references[kind][i]["text"].as_str().unwrap());
                }
                assert!(result.stats.milliseconds["neural_decoder"] > 0.);
            }
        }
    }
    Ok(())
}

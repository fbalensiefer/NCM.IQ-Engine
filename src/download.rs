use anyhow::{Result, ensure};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::Read,
    path::Path,
    process::Command,
};
use uuid::Uuid;

const RELEASE: &str = "https://huggingface.co/hazyresearch/ncm-checkpoints/resolve/d342ce8aaea58830eef3b8ad473664673e41f692";
const VOCAB: &str =
    "https://huggingface.co/openai-community/gpt2/resolve/607a30d783dfa663caf39e06633721c8d4cfcd7e";
const FILES: &[(&str, &str, &str)] = &[
    (
        RELEASE,
        "xsum/config.yaml",
        "5dee6b7339645a22362f695fbadb27df6e5254db2d3be0bff8516138d95da493",
    ),
    (
        RELEASE,
        "xsum/generator.pt",
        "f152f390253de886e46303654f01b9e7a2586e609ba3d6b5e86d5969903b1fd4",
    ),
    (
        RELEASE,
        "xsum/tokenizer.pt",
        "1f806ddaf8e023d1933e95efeabcaaf99f795576c0a05045f1a0f9389ab34fa7",
    ),
    (
        VOCAB,
        "gpt2/tokenizer.json",
        "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
    ),
];

pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn file_hash(path: &Path) -> Result<String> {
    let mut stream = File::open(path)?;
    let mut state = Sha256::new();
    let mut buffer = vec![0; 256 * 1024];
    loop {
        let size = stream.read(&mut buffer)?;
        if size == 0 {
            break;
        }
        state.update(&buffer[..size]);
    }
    Ok(format!("{:x}", state.finalize()))
}

pub fn download(root: &Path) -> Result<()> {
    for &(release, file, hash) in FILES {
        let output = root.join(file);
        if output.is_file() {
            ensure!(
                file_hash(&output)? == hash,
                "existing file has a different checksum: {}",
                output.display()
            );
            continue;
        }
        let parent = output.parent().unwrap();
        fs::create_dir_all(parent)?;
        let temporary = parent.join(format!(".ncm-download-{}", Uuid::new_v4()));
        let url_file = if release == VOCAB {
            "tokenizer.json"
        } else {
            file
        };
        let result = (|| {
            let status = Command::new("curl")
                .args(["-fLsS", "--retry", "2", "--max-time", "900", "--output"])
                .arg(&temporary)
                .arg(format!("{release}/{url_file}"))
                .status()?;
            ensure!(status.success(), "download failed for {file}");
            ensure!(
                file_hash(&temporary)? == hash,
                "checksum mismatch for {file}"
            );
            fs::rename(&temporary, &output)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result?;
    }
    println!("Verified model files in {}", root.display());
    Ok(())
}

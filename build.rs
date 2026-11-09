fn main() {
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-changed=src/mps.cpp");
    println!("cargo:rerun-if-changed=src/int8.cpp");
    println!("cargo:rustc-check-cfg=cfg(ncm_mps)");
    let sdk = std::env::var("LIBTORCH").expect("set LIBTORCH to the native LibTorch SDK directory");
    let sdk = std::path::Path::new(&sdk);
    assert!(sdk.is_absolute(), "LIBTORCH must be an absolute path");
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,{}",
        sdk.join("lib").display()
    );
    let macos = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos");
    let origin = if macos { "@executable_path" } else { "$ORIGIN" };
    println!("cargo:rustc-link-arg=-Wl,-rpath,{origin}/../../.libtorch/lib");
    let abi = std::env::var("LIBTORCH_CXX11_ABI").unwrap_or_else(|_| "1".to_owned());
    println!("cargo:rerun-if-env-changed=LIBTORCH_CXX11_ABI");
    let mut bridge = cc::Build::new();
    bridge
        .cpp(true)
        .std("c++20")
        .define("_GLIBCXX_USE_CXX11_ABI", abi.as_str())
        .include(sdk.join("include"))
        .file("src/int8.cpp");
    if macos {
        bridge.file("src/mps.cpp");
        println!("cargo:rustc-cfg=ncm_mps");
    }
    bridge.compile("ncm_libtorch");
}

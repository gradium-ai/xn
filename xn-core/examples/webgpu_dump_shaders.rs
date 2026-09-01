//! Writes every kernel's composed WGSL to a directory, so a browser's WGSL
//! implementation can validate the whole set.
//!
//! Naga (what wgpu uses natively) is more permissive than the WGSL validators in
//! Chrome and Safari, and the backend compiles kernels lazily on first dispatch,
//! so a kernel that a browser rejects would otherwise only surface when some op
//! is first exercised in the browser.
//!
//! Run with: cargo run --release --features webgpu --example webgpu_dump_shaders -- <outdir>

fn main() -> xn::Result<()> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "/tmp/xn-wgsl".to_string());
    std::fs::create_dir_all(&dir).map_err(xn::Error::wrap)?;
    let sources = xn::webgpu_backend::all_shader_sources();
    let mut index = Vec::new();
    for (name, src) in &sources {
        let path = format!("{dir}/{name}.wgsl");
        std::fs::write(&path, src).map_err(xn::Error::wrap)?;
        index.push(name.clone());
    }
    std::fs::write(format!("{dir}/index.json"), serde_json::to_string(&index).unwrap())
        .map_err(xn::Error::wrap)?;
    println!("wrote {} shaders to {dir}", sources.len());
    Ok(())
}

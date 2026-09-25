pub fn get_num_cpus() -> usize {
    num_cpus::get()
}

static NUM_THREADS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn set_num_threads(num_threads: usize) {
    NUM_THREADS.store(num_threads, std::sync::atomic::Ordering::Relaxed);
    // `set_var` panics on this target, where rayon's width comes from the embedder's pool.
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    unsafe {
        std::env::set_var("RAYON_NUM_THREADS", num_threads.to_string());
    }
}

pub fn get_num_threads() -> usize {
    let n = NUM_THREADS.load(std::sync::atomic::Ordering::Relaxed);
    if n > 0 {
        return n;
    }
    use std::str::FromStr;
    // Respond to the same environment variable as rayon.
    match std::env::var("RAYON_NUM_THREADS").ok().and_then(|s| usize::from_str(&s).ok()) {
        Some(x) if x > 0 => x,
        Some(_) | None => num_cpus::get(),
    }
}

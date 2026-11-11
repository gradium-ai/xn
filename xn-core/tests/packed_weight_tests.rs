//! The CPU gemm keeps model parameters packed between calls: a linear layer's weight and a
//! convolution's kernel. These check that doing so changes no result, bit for bit, and that
//! nothing else is ever served from that cache, least of all a buffer written in place.
//!
//! The cache's budget is process-wide and the thread pool's width is fixed at first use, hence
//! a dedicated test binary, and a mutex to keep the tests out of each other's way inside it.

use std::sync::{Mutex, MutexGuard, Once};
use xn::nn::Linear;
use xn::{CPU, CpuTensor, Result, Tensor};

static EXCLUSIVE: Mutex<()> = Mutex::new(());

/// Four threads, so that a wide product is split into column stripes across the pool.
fn setup() -> MutexGuard<'static, ()> {
    static WIDTH: Once = Once::new();
    WIDTH.call_once(|| xn::set_num_threads(4));
    EXCLUSIVE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Deterministic values in [-1, 1).
fn values(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

/// Whether f32 products reach the gemm, and with it the cache: not with Accelerate, which runs
/// them in BLAS, nor where KleidiAI's kernels take them.
fn products_reach_the_cache() -> bool {
    #[cfg(xn_kai)]
    let kai = xn::quantized::kai::active();
    #[cfg(not(xn_kai))]
    let kai = false;
    !cfg!(feature = "accelerate") && !kai
}

fn tensor(shape: &[usize], seed: u64) -> Result<CpuTensor<f32>> {
    Tensor::from_vec(values(shape.iter().product(), seed), shape.to_vec(), &CPU)
}

#[test]
fn a_parameter_used_again_gives_the_same_result() -> Result<()> {
    let _x = setup();
    let before = gemm::packed_cache::stats().0;

    // The first call packs per call, the second packs into the cache, later ones reuse it.
    // One layer wide enough to be split into stripes across the pool, one too narrow to be.
    for (rows, out, inp) in [(16, 512, 384), (6, 12, 64)] {
        let linear = Linear::new(tensor(&[out, inp], 1)?);
        let x = tensor(&[rows, inp], 2)?;
        let first = linear.forward(&x)?.to_vec()?;
        for call in 1..4 {
            assert_eq!(linear.forward(&x)?.to_vec()?, first, "linear {out}x{inp}, call {call}");
        }
    }

    let kernel = tensor(&[128, 256, 3], 3)?;
    let x = tensor(&[1, 256, 10], 4)?;
    let first = x.conv1d(&kernel, None, 1, 0, 1, 1)?.to_vec()?;
    for call in 1..4 {
        assert_eq!(x.conv1d(&kernel, None, 1, 0, 1, 1)?.to_vec()?, first, "conv1d, call {call}");
    }

    let kernel = tensor(&[256, 128, 10], 5)?;
    let x = tensor(&[1, 256, 4], 6)?;
    let first = x.conv_transpose1d(&kernel, None, 5, 0, 0, 1)?.to_vec()?;
    for call in 1..4 {
        let again = x.conv_transpose1d(&kernel, None, 5, 0, 0, 1)?.to_vec()?;
        assert_eq!(again, first, "conv_transpose1d, call {call}");
    }

    if products_reach_the_cache() {
        assert!(gemm::packed_cache::stats().0 > before, "the parameters were cached");
    }
    Ok(())
}

/// A key/value cache written in place, as a decoder's prompt is prefilled: the same buffers and
/// the same shapes on every call, with new rows where the cache's fingerprint does not look.
#[test]
fn a_buffer_written_in_place_is_never_served_stale() -> Result<()> {
    let _x = setup();
    let (heads, dim, capacity, prefix, t) = (4, 32, 128, 60, 4);
    let len = prefix + t;
    let keys = Tensor::<f32, _>::zeros((1, capacity, heads, dim), &CPU)?;
    let vals = Tensor::<f32, _>::zeros((1, capacity, heads, dim), &CPU)?;
    keys.slice_set(&tensor(&[1, prefix, heads, dim], 7)?, 1, 0)?;
    vals.slice_set(&tensor(&[1, prefix, heads, dim], 8)?, 1, 0)?;
    // Small enough to run on this thread, so a panel would be counted here.
    let panels = gemm::packed_cache::stats().1;

    for prompt in 0..6 {
        keys.slice_set(&tensor(&[1, t, heads, dim], 100 + prompt)?, 1, prefix)?;
        vals.slice_set(&tensor(&[1, t, heads, dim], 200 + prompt)?, 1, prefix)?;
        let q = tensor(&[1, t, heads, dim], 300 + prompt)?;
        let p = tensor(&[1, heads, t, len], 400 + prompt)?;

        let k = keys.narrow(1, ..len)?.transpose(1, 2)?;
        let v = vals.narrow(1, ..len)?.transpose(1, 2)?;
        let scores = q.transpose(1, 2)?.matmul_t(&k)?.to_vec()?;
        let out = p.matmul(&v)?.to_vec()?;

        let (kd, vd, qd, pd) = (keys.to_vec()?, vals.to_vec()?, q.to_vec()?, p.to_vec()?);
        let at = |pos: usize, h: usize, c: usize| (pos * heads + h) * dim + c;
        for h in 0..heads {
            for i in 0..t {
                for j in 0..len {
                    let want: f32 = (0..dim).map(|c| qd[at(i, h, c)] * kd[at(j, h, c)]).sum();
                    let got = scores[(h * t + i) * len + j];
                    assert!((got - want).abs() < 1e-4, "prompt {prompt}: scores[{h},{i},{j}]");
                }
                for c in 0..dim {
                    let want: f32 =
                        (0..len).map(|j| pd[(h * t + i) * len + j] * vd[at(j, h, c)]).sum();
                    let got = out[(h * t + i) * dim + c];
                    assert!((got - want).abs() < 1e-4, "prompt {prompt}: out[{h},{i},{c}]");
                }
            }
        }
    }
    assert_eq!(gemm::packed_cache::stats().1, panels, "nothing but parameters is cached");

    // The count is this thread's. A declared parameter in a product as small as the ones above
    // does move it, so the assertion above cannot pass because the work ran elsewhere.
    if products_reach_the_cache() {
        let kernel = tensor(&[16, 8, 4], 9)?;
        let x = tensor(&[1, 16, 8], 10)?;
        for _ in 0..3 {
            x.conv_transpose1d(&kernel, None, 2, 0, 0, 1)?;
        }
        assert!(gemm::packed_cache::stats().1 > panels, "a parameter is counted on this thread");
    }
    Ok(())
}

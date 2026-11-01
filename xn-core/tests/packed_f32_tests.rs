use xn::cpu_backend::packed::{PackedF32, PackedF32Scratch};

#[test]
fn rejects_invalid_dimensions_before_native_access() {
    assert!(PackedF32::try_new(&[], 0, 1, None).is_err());
    assert!(PackedF32::try_new(&[1.], 1, 2, None).is_err());
    assert!(PackedF32::try_new(&[1.], 1, 1, Some(&[])).is_err());
    assert!(PackedF32::try_new(&[], usize::MAX, 2, None).is_err());
}

#[test]
fn snapshot_bias_tail_rows_and_concurrent_workspaces() -> xn::Result<()> {
    let (n, k) = (19, 37);
    let mut weights: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.13).sin() * 0.2).collect();
    let original = weights.clone();
    let mut bias: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
    let original_bias = bias.clone();
    let packed = PackedF32::try_new(&weights, n, k, Some(&bias))?;
    #[cfg(xn_kai)]
    assert_eq!(packed.is_some(), xn::quantized::kai::active());
    #[cfg(not(xn_kai))]
    assert!(packed.is_none());
    let Some(packed) = packed else {
        return Ok(());
    };
    assert!(packed.packed_bytes() >= n * k * 4);
    weights.fill(99.);
    bias.fill(-99.); // Construction snapshots both inputs.
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|worker| {
                let (packed, original, original_bias) = (&packed, &original, &original_bias);
                scope.spawn(move || -> xn::Result<()> {
                    let mut scratch = PackedF32Scratch::default();
                    for rows in [1, 2, 3, 17, 8, 0, 31, 1] {
                        let input: Vec<f32> =
                            (0..rows * k).map(|i| ((i + worker) as f32 * 0.07).cos()).collect();
                        let mut output = vec![f32::NAN; rows * n];
                        packed.matmul(&input, rows, &mut output, &mut scratch)?;
                        for (i, &v) in output.iter().enumerate() {
                            let (r, c) = (i / n, i % n);
                            let expected = original_bias[c]
                                + (0..k)
                                    .map(|j| input[r * k + j] * original[c * k + j])
                                    .sum::<f32>();
                            assert!(
                                v.is_finite() && (v - expected).abs() < 1e-5,
                                "row={r} col={c} {v} != {expected}"
                            );
                        }
                    }
                    Ok(())
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap()?;
        }
        xn::Result::Ok(())
    })?;
    let mut scratch = PackedF32Scratch::default();
    let mut output = vec![42.; n];
    assert!(packed.matmul(&vec![0.; k - 1], 1, &mut output, &mut scratch).is_err());
    assert_eq!(output, vec![42.; n]);
    assert!(packed.matmul(&vec![0.; k], usize::MAX, &mut output, &mut scratch).is_err());
    Ok(())
}

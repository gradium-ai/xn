#![cfg(feature = "cuda")]

use xn::{Result, Tensor, cuda_backend::Device};

fn get_device() -> Device {
    Device::new(0).expect("Failed to initialize CUDA device")
}

// =============================================================================
// Basic tensor operations
// =============================================================================

#[test]
fn test_from_vec_and_to_vec() -> Result<()> {
    let device = get_device();
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let t: Tensor<f32, Device> = Tensor::from_vec(data.clone(), vec![5], &device)?;
    assert_eq!(t.dims(), &[5]);
    assert_eq!(t.to_vec()?, data);
    Ok(())
}

#[test]
fn test_zeros() -> Result<()> {
    let device = get_device();
    let zeros: Tensor<f32, Device> = Tensor::zeros(vec![3, 4], &device)?;
    assert_eq!(zeros.dims(), &[3, 4]);
    assert!(zeros.to_vec()?.iter().all(|&x| x == 0.0));
    Ok(())
}

#[test]
fn test_full() -> Result<()> {
    let device = get_device();
    let full: Tensor<f32, Device> = Tensor::full(42.0, vec![2, 3], &device)?;
    assert_eq!(full.dims(), &[2, 3]);
    assert!(full.to_vec()?.iter().all(|&x| x == 42.0));
    Ok(())
}

#[test]
fn test_f16_roundtrip() -> Result<()> {
    let device = get_device();
    let data: Vec<half::f16> = vec![1.0, 2.0, 3.0].into_iter().map(half::f16::from_f32).collect();
    let t: Tensor<half::f16, Device> = Tensor::from_vec(data.clone(), vec![3], &device)?;
    assert_eq!(t.to_vec()?, data);
    Ok(())
}

#[test]
fn test_bf16_roundtrip() -> Result<()> {
    let device = get_device();
    let data: Vec<half::bf16> = vec![1.0, 2.0, 3.0].into_iter().map(half::bf16::from_f32).collect();
    let t: Tensor<half::bf16, Device> = Tensor::from_vec(data.clone(), vec![3], &device)?;
    assert_eq!(t.to_vec()?, data);
    Ok(())
}

// =============================================================================
// Binary operations
// =============================================================================

#[test]
fn test_add() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4], &device)?;
    let b: Tensor<f32, Device> = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![4], &device)?;
    let c = a.add(&b)?;
    assert_eq!(c.to_vec()?, vec![6.0, 8.0, 10.0, 12.0]);
    Ok(())
}

#[test]
fn test_sub() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![4], &device)?;
    let b: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4], &device)?;
    let c = a.sub(&b)?;
    assert_eq!(c.to_vec()?, vec![4.0, 4.0, 4.0, 4.0]);
    Ok(())
}

#[test]
fn test_mul() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4], &device)?;
    let b: Tensor<f32, Device> = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![4], &device)?;
    let c = a.mul(&b)?;
    assert_eq!(c.to_vec()?, vec![5.0, 12.0, 21.0, 32.0]);
    Ok(())
}

#[test]
fn test_div() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> = Tensor::from_vec(vec![10.0, 12.0, 21.0, 32.0], vec![4], &device)?;
    let b: Tensor<f32, Device> = Tensor::from_vec(vec![2.0, 3.0, 7.0, 8.0], vec![4], &device)?;
    let c = a.div(&b)?;
    assert_eq!(c.to_vec()?, vec![5.0, 4.0, 3.0, 4.0]);
    Ok(())
}

#[test]
fn test_scale() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4], &device)?;
    let b = a.scale(2.0)?;
    assert_eq!(b.to_vec()?, vec![2.0, 4.0, 6.0, 8.0]);
    Ok(())
}

#[test]
fn test_maximum() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 5.0, 3.0, 4.0], vec![4], &device)?;
    let b: Tensor<f32, Device> = Tensor::from_vec(vec![2.0, 3.0, 4.0, 2.0], vec![4], &device)?;
    let c = a.maximum(&b)?;
    assert_eq!(c.to_vec()?, vec![2.0, 5.0, 4.0, 4.0]);
    Ok(())
}

#[test]
fn test_minimum() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 5.0, 3.0, 4.0], vec![4], &device)?;
    let b: Tensor<f32, Device> = Tensor::from_vec(vec![2.0, 3.0, 4.0, 2.0], vec![4], &device)?;
    let c = a.minimum(&b)?;
    assert_eq!(c.to_vec()?, vec![1.0, 3.0, 3.0, 2.0]);
    Ok(())
}

// =============================================================================
// Unary operations
// =============================================================================

#[test]
fn test_relu() -> Result<()> {
    let device = get_device();
    let x: Tensor<f32, Device> =
        Tensor::from_vec(vec![0.0, 1.0, -1.0, 2.0, -2.0], vec![5], &device)?;
    let y = x.relu()?;
    assert_eq!(y.to_vec()?, vec![0.0, 1.0, 0.0, 2.0, 0.0]);
    Ok(())
}

#[test]
fn test_silu() -> Result<()> {
    let device = get_device();
    let x: Tensor<f32, Device> = Tensor::from_vec(vec![0.0, 1.0, -1.0], vec![3], &device)?;
    let y = x.silu()?;
    let y_data = y.to_vec()?;
    assert!((y_data[0] - 0.0).abs() < 1e-5);
    assert!((y_data[1] - 0.7310586).abs() < 1e-5);
    assert!((y_data[2] - (-0.26894143)).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_sqrt() -> Result<()> {
    let device = get_device();
    let x: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 4.0, 9.0, 16.0], vec![4], &device)?;
    let y = x.sqrt()?;
    assert_eq!(y.to_vec()?, vec![1.0, 2.0, 3.0, 4.0]);
    Ok(())
}

#[test]
fn test_sqr() -> Result<()> {
    let device = get_device();
    let x: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4], &device)?;
    let y = x.sqr()?;
    assert_eq!(y.to_vec()?, vec![1.0, 4.0, 9.0, 16.0]);
    Ok(())
}

#[test]
fn test_tanh() -> Result<()> {
    let device = get_device();
    let x: Tensor<f32, Device> = Tensor::from_vec(vec![0.0, 1.0, -1.0], vec![3], &device)?;
    let y = x.tanh()?;
    let y_data = y.to_vec()?;
    assert!((y_data[0] - 0.0).abs() < 1e-5);
    assert!((y_data[1] - 0.7615942).abs() < 1e-5);
    assert!((y_data[2] - (-0.7615942)).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_sigmoid() -> Result<()> {
    let device = get_device();
    let x: Tensor<f32, Device> = Tensor::from_vec(vec![0.0, 1.0, -1.0], vec![3], &device)?;
    let y = x.sigmoid()?;
    let y_data = y.to_vec()?;
    assert!((y_data[0] - 0.5).abs() < 1e-5);
    assert!((y_data[1] - 0.7310586).abs() < 1e-5);
    assert!((y_data[2] - 0.26894143).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_abs() -> Result<()> {
    let device = get_device();
    let x: Tensor<f32, Device> = Tensor::from_vec(vec![-1.0, 2.0, -3.0, 4.0], vec![4], &device)?;
    let y = x.abs()?;
    assert_eq!(y.to_vec()?, vec![1.0, 2.0, 3.0, 4.0]);
    Ok(())
}

// =============================================================================
// Matrix multiplication
// =============================================================================

#[test]
fn test_matmul_2d() -> Result<()> {
    let device = get_device();
    // A = [[1, 2, 3], [4, 5, 6]]  (2x3)
    // B = [[1, 2], [3, 4], [5, 6]]  (3x2)
    // C = A @ B = [[22, 28], [49, 64]]  (2x2)
    let a: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], &device)?;
    let b: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], &device)?;
    let c = a.matmul(&b)?;
    assert_eq!(c.dims(), &[2, 2]);
    assert_eq!(c.to_vec()?, vec![22.0, 28.0, 49.0, 64.0]);
    Ok(())
}

#[test]
fn test_matmul_f16() -> Result<()> {
    let device = get_device();
    let a: Tensor<half::f16, Device> = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_iter().map(half::f16::from_f32).collect(),
        vec![2, 3],
        &device,
    )?;
    let b: Tensor<half::f16, Device> = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_iter().map(half::f16::from_f32).collect(),
        vec![3, 2],
        &device,
    )?;
    let c = a.matmul(&b)?;
    let c_data: Vec<f32> = c.to_vec()?.iter().map(|x| x.to_f32()).collect();
    assert!((c_data[0] - 22.0).abs() < 0.1);
    assert!((c_data[1] - 28.0).abs() < 0.1);
    assert!((c_data[2] - 49.0).abs() < 0.1);
    assert!((c_data[3] - 64.0).abs() < 0.1);
    Ok(())
}

#[test]
fn test_matmul_batched() -> Result<()> {
    let device = get_device();
    // Batch of 2: each batch is 2x3 @ 3x2
    let a: Tensor<f32, Device> = Tensor::from_vec(
        vec![
            // Batch 0: [[1,2,3], [4,5,6]]
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // Batch 1: all ones
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
        vec![2, 2, 3],
        &device,
    )?;
    let b: Tensor<f32, Device> = Tensor::from_vec(
        vec![
            // Batch 0
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // Batch 1: all ones
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
        vec![2, 3, 2],
        &device,
    )?;
    let c = a.matmul(&b)?;
    assert_eq!(c.dims(), &[2, 2, 2]);
    let c_data = c.to_vec()?;
    // Batch 0: [[22, 28], [49, 64]]
    assert_eq!(c_data[0], 22.0);
    assert_eq!(c_data[1], 28.0);
    assert_eq!(c_data[2], 49.0);
    assert_eq!(c_data[3], 64.0);
    // Batch 1: [[3, 3], [3, 3]]
    assert_eq!(c_data[4], 3.0);
    assert_eq!(c_data[5], 3.0);
    assert_eq!(c_data[6], 3.0);
    assert_eq!(c_data[7], 3.0);
    Ok(())
}

// =============================================================================
// Transpose
// =============================================================================

#[test]
fn test_transpose_2d() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], &device)?;
    let b = a.transpose(0, 1)?.contiguous()?;
    assert_eq!(b.dims(), &[3, 2]);
    // Original: [[1, 2, 3], [4, 5, 6]]
    // Transposed: [[1, 4], [2, 5], [3, 6]]
    assert_eq!(b.to_vec()?, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    Ok(())
}

#[test]
fn test_transpose_3d() -> Result<()> {
    let device = get_device();
    // Shape: [2, 3, 4]
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let a: Tensor<f32, Device> = Tensor::from_vec(data, vec![2, 3, 4], &device)?;

    // Transpose dims 1 and 2: [2, 3, 4] -> [2, 4, 3]
    let b = a.transpose(1, 2)?;
    assert_eq!(b.dims(), &[2, 4, 3]);
    Ok(())
}

// =============================================================================
// Softmax
// =============================================================================

#[test]
fn test_softmax() -> Result<()> {
    let device = get_device();
    let x: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], vec![2, 3], &device)?;
    let y = x.softmax()?;
    let y_data = y.to_vec()?;

    // Each row should sum to 1.0
    let row1_sum: f32 = y_data[0..3].iter().sum();
    let row2_sum: f32 = y_data[3..6].iter().sum();
    assert!((row1_sum - 1.0).abs() < 1e-5);
    assert!((row2_sum - 1.0).abs() < 1e-5);

    // Check expected values: softmax([1,2,3]) ≈ [0.090, 0.245, 0.665]
    assert!((y_data[0] - 0.0900306).abs() < 1e-4);
    assert!((y_data[1] - 0.2447285).abs() < 1e-4);
    assert!((y_data[2] - 0.6652409).abs() < 1e-4);
    Ok(())
}

#[test]
fn test_softmax_single_row() -> Result<()> {
    let device = get_device();
    let x: Tensor<f32, Device> = Tensor::from_vec(vec![0.0, 0.0, 0.0], vec![1, 3], &device)?;
    let y = x.softmax()?;
    let y_data = y.to_vec()?;
    // Uniform distribution
    assert!((y_data[0] - 1.0 / 3.0).abs() < 1e-5);
    assert!((y_data[1] - 1.0 / 3.0).abs() < 1e-5);
    assert!((y_data[2] - 1.0 / 3.0).abs() < 1e-5);
    Ok(())
}

// =============================================================================
// RMS Norm
// =============================================================================

#[test]
fn test_rms_norm() -> Result<()> {
    let device = get_device();
    let x: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], &device)?;
    let alpha: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 1.0, 1.0], vec![3], &device)?;
    let y = x.rms_norm(&alpha, 1e-5)?;
    let y_data = y.to_vec()?;

    // RMS of [1,2,3] = sqrt((1+4+9)/3) = sqrt(14/3) ≈ 2.16
    let rms_row1 = (1.0f32 + 4.0 + 9.0) / 3.0;
    let scale1 = 1.0 / (rms_row1 + 1e-5).sqrt();
    assert!((y_data[0] - 1.0 * scale1).abs() < 1e-4);
    assert!((y_data[1] - 2.0 * scale1).abs() < 1e-4);
    assert!((y_data[2] - 3.0 * scale1).abs() < 1e-4);

    // RMS of [4,5,6] = sqrt((16+25+36)/3) = sqrt(77/3) ≈ 5.07
    let rms_row2 = (16.0f32 + 25.0 + 36.0) / 3.0;
    let scale2 = 1.0 / (rms_row2 + 1e-5).sqrt();
    assert!((y_data[3] - 4.0 * scale2).abs() < 1e-4);
    assert!((y_data[4] - 5.0 * scale2).abs() < 1e-4);
    assert!((y_data[5] - 6.0 * scale2).abs() < 1e-4);
    Ok(())
}

#[test]
fn test_rms_norm_with_scale() -> Result<()> {
    let device = get_device();
    let x: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3], &device)?;
    let alpha: Tensor<f32, Device> = Tensor::from_vec(vec![2.0, 2.0, 2.0], vec![3], &device)?;
    let y = x.rms_norm(&alpha, 1e-5)?;
    let y_data = y.to_vec()?;

    let rms = (1.0f32 + 4.0 + 9.0) / 3.0;
    let scale = 1.0 / (rms + 1e-5).sqrt();
    // Values should be doubled due to alpha=2
    assert!((y_data[0] - 1.0 * scale * 2.0).abs() < 1e-4);
    assert!((y_data[1] - 2.0 * scale * 2.0).abs() < 1e-4);
    assert!((y_data[2] - 3.0 * scale * 2.0).abs() < 1e-4);
    Ok(())
}

// =============================================================================
// Layer Norm
// =============================================================================

#[test]
fn test_layer_norm() -> Result<()> {
    let device = get_device();
    let x: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], &device)?;
    let weight: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 1.0, 1.0], vec![3], &device)?;
    let bias: Tensor<f32, Device> = Tensor::from_vec(vec![0.0, 0.0, 0.0], vec![3], &device)?;
    let y = x.layer_norm(&weight, &bias, 1e-5)?;
    let y_data = y.to_vec()?;

    // For [1,2,3]: mean=2, var=2/3
    let mean1 = 2.0f32;
    let var1 = ((1.0 - mean1).powi(2) + (2.0 - mean1).powi(2) + (3.0 - mean1).powi(2)) / 3.0;
    let inv_std1 = 1.0 / (var1 + 1e-5).sqrt();
    assert!((y_data[0] - (1.0 - mean1) * inv_std1).abs() < 1e-4);
    assert!((y_data[1] - (2.0 - mean1) * inv_std1).abs() < 1e-4);
    assert!((y_data[2] - (3.0 - mean1) * inv_std1).abs() < 1e-4);

    // For [4,5,6]: mean=5, var=2/3
    let mean2 = 5.0f32;
    let var2 = ((4.0 - mean2).powi(2) + (5.0 - mean2).powi(2) + (6.0 - mean2).powi(2)) / 3.0;
    let inv_std2 = 1.0 / (var2 + 1e-5).sqrt();
    assert!((y_data[3] - (4.0 - mean2) * inv_std2).abs() < 1e-4);
    assert!((y_data[4] - (5.0 - mean2) * inv_std2).abs() < 1e-4);
    assert!((y_data[5] - (6.0 - mean2) * inv_std2).abs() < 1e-4);
    Ok(())
}

#[test]
fn test_layer_norm_with_affine() -> Result<()> {
    let device = get_device();
    let x: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3], &device)?;
    let weight: Tensor<f32, Device> = Tensor::from_vec(vec![2.0, 2.0, 2.0], vec![3], &device)?;
    let bias: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 1.0, 1.0], vec![3], &device)?;
    let y = x.layer_norm(&weight, &bias, 1e-5)?;
    let y_data = y.to_vec()?;

    let mean = 2.0f32;
    let var = 2.0f32 / 3.0;
    let inv_std = 1.0 / (var + 1e-5).sqrt();
    // y = (x - mean) * inv_std * weight + bias
    assert!((y_data[0] - ((1.0 - mean) * inv_std * 2.0 + 1.0)).abs() < 1e-4);
    assert!((y_data[1] - ((2.0 - mean) * inv_std * 2.0 + 1.0)).abs() < 1e-4);
    assert!((y_data[2] - ((3.0 - mean) * inv_std * 2.0 + 1.0)).abs() < 1e-4);
    Ok(())
}

// =============================================================================
// Reshape
// =============================================================================

#[test]
fn test_reshape() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> =
        Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], vec![2, 3], &device)?;

    let b = a.reshape((3, 2))?;
    assert_eq!(b.dims(), &[3, 2]);
    assert_eq!(b.to_vec()?, vec![1., 2., 3., 4., 5., 6.]);

    let c = a.reshape((6,))?;
    assert_eq!(c.dims(), &[6]);

    let d = a.reshape((1, 2, 3))?;
    assert_eq!(d.dims(), &[1, 2, 3]);
    Ok(())
}

#[test]
fn test_reshape_with_hole() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> =
        Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], vec![2, 3], &device)?;

    let b = a.reshape((3, ()))?;
    assert_eq!(b.dims(), &[3, 2]);

    let c = a.reshape(((), 2))?;
    assert_eq!(c.dims(), &[3, 2]);
    Ok(())
}

// =============================================================================
// Reduce operations
// =============================================================================

#[test]
fn test_reduce_max_1d() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 5.0, 2.0, 8.0, 3.0], vec![5], &device)?;
    let max_val = a.max(0)?;
    assert_eq!(max_val.dims(), &[1]);
    assert_eq!(max_val.to_vec()?, vec![8.0]);
    Ok(())
}

#[test]
fn test_reduce_min_1d() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> = Tensor::from_vec(vec![3.0, 1.0, 4.0, 1.0, 5.0], vec![5], &device)?;
    let min_val = a.min(0)?;
    assert_eq!(min_val.dims(), &[1]);
    assert_eq!(min_val.to_vec()?, vec![1.0]);
    Ok(())
}

#[test]
fn test_reduce_max_2d_dim0() -> Result<()> {
    let device = get_device();
    // Shape [3, 4]
    let a: Tensor<f32, Device> = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![3, 4],
        &device,
    )?;
    // Max along dim 0 -> [4]
    let max_val = a.max(0)?;
    assert_eq!(max_val.dims(), &[4]);
    assert_eq!(max_val.to_vec()?, vec![9.0, 10.0, 11.0, 12.0]);
    Ok(())
}

#[test]
fn test_reduce_max_2d_dim1() -> Result<()> {
    let device = get_device();
    // Shape [3, 4]
    let a: Tensor<f32, Device> = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![3, 4],
        &device,
    )?;
    // Max along dim 1 -> [3]
    let max_val = a.max(1)?;
    assert_eq!(max_val.dims(), &[3]);
    assert_eq!(max_val.to_vec()?, vec![4.0, 8.0, 12.0]);
    Ok(())
}

#[test]
fn test_reduce_min_2d_dim0() -> Result<()> {
    let device = get_device();
    // Shape [3, 4]
    let a: Tensor<f32, Device> = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![3, 4],
        &device,
    )?;
    // Min along dim 0 -> [4]
    let min_val = a.min(0)?;
    assert_eq!(min_val.dims(), &[4]);
    assert_eq!(min_val.to_vec()?, vec![1.0, 2.0, 3.0, 4.0]);
    Ok(())
}

#[test]
fn test_reduce_min_2d_dim1() -> Result<()> {
    let device = get_device();
    // Shape [3, 4]
    let a: Tensor<f32, Device> = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![3, 4],
        &device,
    )?;
    // Min along dim 1 -> [3]
    let min_val = a.min(1)?;
    assert_eq!(min_val.dims(), &[3]);
    assert_eq!(min_val.to_vec()?, vec![1.0, 5.0, 9.0]);
    Ok(())
}

#[test]
fn test_reduce_max_3d() -> Result<()> {
    let device = get_device();
    // Shape [2, 3, 4]
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let a: Tensor<f32, Device> = Tensor::from_vec(data, vec![2, 3, 4], &device)?;

    // Max along dim 1 (middle dimension) -> [2, 4]
    let max_val = a.max(1)?;
    assert_eq!(max_val.dims(), &[2, 4]);
    // For first batch: max over rows [[1,2,3,4], [5,6,7,8], [9,10,11,12]] = [9,10,11,12]
    // For second batch: max over rows [[13,14,15,16], [17,18,19,20], [21,22,23,24]] = [21,22,23,24]
    assert_eq!(max_val.to_vec()?, vec![9.0, 10.0, 11.0, 12.0, 21.0, 22.0, 23.0, 24.0]);
    Ok(())
}

// =============================================================================
// Index select operations
// =============================================================================

#[test]
fn test_index_select_1d() -> Result<()> {
    let device = get_device();
    // Shape [5]
    let a: Tensor<f32, Device> =
        Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0], vec![5], &device)?;
    let indices = Tensor::from_vec(vec![0i64, 2, 4], 3, &device)?;
    let selected = a.index_select(&indices, 0)?;
    assert_eq!(selected.dims(), &[3]);
    assert_eq!(selected.to_vec()?, vec![10.0, 30.0, 50.0]);
    Ok(())
}

#[test]
fn test_index_select_2d_dim0() -> Result<()> {
    let device = get_device();
    // Shape [4, 3] - select rows
    let a: Tensor<f32, Device> = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![4, 3],
        &device,
    )?;
    let indices = Tensor::from_vec(vec![0i64, 2, 3], 3, &device)?;
    let selected = a.index_select(&indices, 0)?;
    assert_eq!(selected.dims(), &[3, 3]);
    // Rows 0, 2, 3
    assert_eq!(selected.to_vec()?, vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    Ok(())
}

#[test]
fn test_index_select_2d_dim1() -> Result<()> {
    let device = get_device();
    // Shape [3, 4] - select columns
    let a: Tensor<f32, Device> = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![3, 4],
        &device,
    )?;
    let indices = Tensor::from_vec(vec![1i64, 3], 2, &device)?;
    let selected = a.index_select(&indices, 1)?;
    assert_eq!(selected.dims(), &[3, 2]);
    // Columns 1, 3 from each row
    assert_eq!(selected.to_vec()?, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    Ok(())
}

#[test]
fn test_index_select_3d() -> Result<()> {
    let device = get_device();
    // Shape [2, 3, 4] - embedding-style lookup on dim 1
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let a: Tensor<f32, Device> = Tensor::from_vec(data, vec![2, 3, 4], &device)?;

    let indices = Tensor::from_vec(vec![0i64, 2], 2, &device)?;
    let selected = a.index_select(&indices, 1)?;
    assert_eq!(selected.dims(), &[2, 2, 4]);
    // From first batch [1-12]: rows 0 and 2 -> [1,2,3,4] and [9,10,11,12]
    // From second batch [13-24]: rows 0 and 2 -> [13,14,15,16] and [21,22,23,24]
    assert_eq!(
        selected.to_vec()?,
        vec![
            1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 21.0, 22.0, 23.0,
            24.0
        ]
    );
    Ok(())
}

#[test]
fn test_index_select_minus_one_f32() -> Result<()> {
    let device = get_device();
    // Shape [4, 3], rows 1..=12.
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let a: Tensor<f32, Device> = Tensor::from_vec(data, vec![4, 3], &device)?;

    // -1 selects a row of zeros on dim 0.
    let indices = Tensor::from_vec(vec![2i64, -1, 0], 3, &device)?;
    let selected = a.index_select(&indices, 0)?;
    assert_eq!(selected.dims(), &[3, 3]);
    assert_eq!(selected.to_vec()?, vec![7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);

    // -1 selects zeros on dim 1 as well.
    let indices = Tensor::from_vec(vec![-1i64, 1, -1], 3, &device)?;
    let selected = a.index_select(&indices, 1)?;
    assert_eq!(selected.dims(), &[4, 3]);
    assert_eq!(
        selected.to_vec()?,
        vec![0.0, 2.0, 0.0, 0.0, 5.0, 0.0, 0.0, 8.0, 0.0, 0.0, 11.0, 0.0]
    );
    Ok(())
}

#[test]
fn test_index_select_minus_one_f16() -> Result<()> {
    let device = get_device();
    let data: Vec<half::f16> = (1..=12).map(|x| half::f16::from_f32(x as f32)).collect();
    let a: Tensor<half::f16, Device> = Tensor::from_vec(data, vec![4, 3], &device)?;

    let indices = Tensor::from_vec(vec![2i64, -1, 0], 3, &device)?;
    let selected = a.index_select(&indices, 0)?;
    let got: Vec<f32> = selected.to_vec()?.into_iter().map(|v| v.to_f32()).collect();
    assert_eq!(got, vec![7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
    Ok(())
}

#[test]
fn test_index_select_minus_one_bf16() -> Result<()> {
    let device = get_device();
    let data: Vec<half::bf16> = (1..=12).map(|x| half::bf16::from_f32(x as f32)).collect();
    let a: Tensor<half::bf16, Device> = Tensor::from_vec(data, vec![4, 3], &device)?;

    let indices = Tensor::from_vec(vec![2i64, -1, 0], 3, &device)?;
    let selected = a.index_select(&indices, 0)?;
    let got: Vec<f32> = selected.to_vec()?.into_iter().map(|v| v.to_f32()).collect();
    assert_eq!(got, vec![7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
    Ok(())
}

// =============================================================================
// Narrow and Cat operations (use copy2d internally)
// =============================================================================

#[test]
fn test_narrow_1d() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5], &device)?;
    let narrowed = a.narrow(0, 1..4)?.contiguous()?;
    assert_eq!(narrowed.dims(), &[3]);
    assert_eq!(narrowed.to_vec()?, vec![2.0, 3.0, 4.0]);
    Ok(())
}

#[test]
fn test_narrow_2d_dim0() -> Result<()> {
    let device = get_device();
    // Shape [4, 3]
    let a: Tensor<f32, Device> = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![4, 3],
        &device,
    )?;
    // Take rows 1..3 (2 rows)
    let narrowed = a.narrow(0, 1..3)?.contiguous()?;
    assert_eq!(narrowed.dims(), &[2, 3]);
    assert_eq!(narrowed.to_vec()?, vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    Ok(())
}

#[test]
fn test_narrow_2d_dim1() -> Result<()> {
    let device = get_device();
    // Shape [3, 4]
    let a: Tensor<f32, Device> = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![3, 4],
        &device,
    )?;
    // Take columns 1..3 (2 columns)
    let narrowed = a.narrow(1, 1..3)?.contiguous()?;
    assert_eq!(narrowed.dims(), &[3, 2]);
    assert_eq!(narrowed.to_vec()?, vec![2.0, 3.0, 6.0, 7.0, 10.0, 11.0]);
    Ok(())
}

#[test]
fn test_cat_1d() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3], &device)?;
    let b: Tensor<f32, Device> = Tensor::from_vec(vec![4.0, 5.0], vec![2], &device)?;
    let c = Tensor::cat(&[&a, &b], 0)?;
    assert_eq!(c.dims(), &[5]);
    assert_eq!(c.to_vec()?, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    Ok(())
}

#[test]
fn test_cat_2d_dim0() -> Result<()> {
    let device = get_device();
    // Shape [2, 3]
    let a: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], &device)?;
    // Shape [1, 3]
    let b: Tensor<f32, Device> = Tensor::from_vec(vec![7.0, 8.0, 9.0], vec![1, 3], &device)?;
    let c = Tensor::cat(&[&a, &b], 0)?;
    assert_eq!(c.dims(), &[3, 3]);
    assert_eq!(c.to_vec()?, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    Ok(())
}

#[test]
fn test_cat_2d_dim1() -> Result<()> {
    let device = get_device();
    // Shape [2, 2]
    let a: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], &device)?;
    // Shape [2, 3]
    let b: Tensor<f32, Device> =
        Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0], vec![2, 3], &device)?;
    let c = Tensor::cat(&[&a, &b], 1)?;
    assert_eq!(c.dims(), &[2, 5]);
    assert_eq!(c.to_vec()?, vec![1.0, 2.0, 5.0, 6.0, 7.0, 3.0, 4.0, 8.0, 9.0, 10.0]);
    Ok(())
}

// =============================================================================
// Causality mask
// =============================================================================

#[test]
fn test_apply_causality_mask() -> Result<()> {
    let device = get_device();
    // Shape [1, 3, 4] - 1 batch*head, 3 query positions, 4 key positions
    // All ones initially
    let a: Tensor<f32, Device> = Tensor::full(1.0, vec![1, 3, 4], &device)?;

    // With offset=0:
    // Query 0 can attend to keys 0..=0 (mask keys 1,2,3)
    // Query 1 can attend to keys 0..=1 (mask keys 2,3)
    // Query 2 can attend to keys 0..=2 (mask key 3)
    let masked = a.apply_causality_mask(0)?;
    let result = masked.to_vec()?;

    // Expected pattern (1.0 = can attend, -inf = masked):
    // Row 0: [1, -inf, -inf, -inf]
    // Row 1: [1, 1, -inf, -inf]
    // Row 2: [1, 1, 1, -inf]
    assert_eq!(result[0], 1.0);
    assert!(result[1].is_infinite() && result[1] < 0.0);
    assert!(result[2].is_infinite() && result[2] < 0.0);
    assert!(result[3].is_infinite() && result[3] < 0.0);

    assert_eq!(result[4], 1.0);
    assert_eq!(result[5], 1.0);
    assert!(result[6].is_infinite() && result[6] < 0.0);
    assert!(result[7].is_infinite() && result[7] < 0.0);

    assert_eq!(result[8], 1.0);
    assert_eq!(result[9], 1.0);
    assert_eq!(result[10], 1.0);
    assert!(result[11].is_infinite() && result[11] < 0.0);

    Ok(())
}

#[test]
fn test_apply_causality_mask_with_offset() -> Result<()> {
    let device = get_device();
    // Shape [1, 2, 4] - simulating KV cache scenario
    // offset=2 means query tokens start at position 2
    let a: Tensor<f32, Device> = Tensor::full(1.0, vec![1, 2, 4], &device)?;

    // With offset=2:
    // Query 0 (position 2) can attend to keys 0..=2 (mask key 3)
    // Query 1 (position 3) can attend to keys 0..=3 (no mask)
    let masked = a.apply_causality_mask(2)?;
    let result = masked.to_vec()?;

    // Row 0: [1, 1, 1, -inf]
    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 1.0);
    assert_eq!(result[2], 1.0);
    assert!(result[3].is_infinite() && result[3] < 0.0);

    // Row 1: [1, 1, 1, 1]
    assert_eq!(result[4], 1.0);
    assert_eq!(result[5], 1.0);
    assert_eq!(result[6], 1.0);
    assert_eq!(result[7], 1.0);

    Ok(())
}

// =============================================================================
// KV cache style cat tests (4D tensors along dim 2)
// =============================================================================

#[test]
fn test_cat_4d_dim2_kv_cache_shape() -> Result<()> {
    // This replicates the exact shapes from the llama KV cache:
    // prev_k: [1, 4, 5, 64] - cached keys from 5 positions
    // k: [1, 4, 1, 64] - new key for 1 position
    // cat along dim 2 -> [1, 4, 6, 64]
    let device = get_device();

    // Create prev_k with sequential values for easy verification
    // Total elements: 1 * 4 * 5 * 64 = 1280
    let prev_k_data: Vec<f32> = (0..1280).map(|i| i as f32).collect();
    let prev_k: Tensor<f32, Device> =
        Tensor::from_vec(prev_k_data.clone(), vec![1, 4, 5, 64], &device)?;

    // Create k with offset values for easy verification
    // Total elements: 1 * 4 * 1 * 64 = 256
    let k_data: Vec<f32> = (10000..10256).map(|i| i as f32).collect();
    let k: Tensor<f32, Device> = Tensor::from_vec(k_data.clone(), vec![1, 4, 1, 64], &device)?;

    // Cat along dim 2
    let result = Tensor::cat(&[&prev_k, &k], 2)?;
    assert_eq!(result.dims(), &[1, 4, 6, 64]);

    let result_data = result.to_vec()?;

    // Verify layout: result should be laid out as [1, 4, 6, 64]
    // For each of the 4 heads:
    //   - First 5*64=320 elements from prev_k
    //   - Then 1*64=64 elements from k
    for head in 0..4 {
        let result_head_start = head * 6 * 64;
        let prev_k_head_start = head * 5 * 64;
        let k_head_start = head * 64;

        // Check first 320 elements of this head come from prev_k
        for i in 0..(5 * 64) {
            let result_idx = result_head_start + i;
            let expected = prev_k_data[prev_k_head_start + i];
            let actual = result_data[result_idx];
            if (expected - actual).abs() > 1e-6 {
                panic!(
                    "Mismatch at head={}, pos within head={}: expected {} (from prev_k[{}]), got {} (result[{}])",
                    head,
                    i,
                    expected,
                    prev_k_head_start + i,
                    actual,
                    result_idx
                );
            }
        }

        // Check last 64 elements of this head come from k
        for i in 0..64 {
            let result_idx = result_head_start + 5 * 64 + i;
            let expected = k_data[k_head_start + i];
            let actual = result_data[result_idx];
            if (expected - actual).abs() > 1e-6 {
                panic!(
                    "Mismatch at head={}, k element {}: expected {} (from k[{}]), got {} (result[{}])",
                    head,
                    i,
                    expected,
                    k_head_start + i,
                    actual,
                    result_idx
                );
            }
        }
    }

    Ok(())
}

#[test]
fn test_cat_4d_dim2_values_preserved() -> Result<()> {
    // Simpler test: just check first 5 values are preserved after cat
    let device = get_device();

    let prev_k_data: Vec<f32> = (0..1280).map(|i| i as f32).collect();
    let prev_k: Tensor<f32, Device> =
        Tensor::from_vec(prev_k_data.clone(), vec![1, 4, 5, 64], &device)?;

    let k: Tensor<f32, Device> = Tensor::full(9999.0, vec![1, 4, 1, 64], &device)?;

    let result = Tensor::cat(&[&prev_k, &k], 2)?;
    let result_data = result.to_vec()?;

    // First 5 elements should be 0.0, 1.0, 2.0, 3.0, 4.0
    assert_eq!(result_data[0], 0.0, "first element should be 0.0");
    assert_eq!(result_data[1], 1.0, "second element should be 1.0");
    assert_eq!(result_data[2], 2.0, "third element should be 2.0");
    assert_eq!(result_data[3], 3.0, "fourth element should be 3.0");
    assert_eq!(result_data[4], 4.0, "fifth element should be 4.0");

    Ok(())
}

#[test]
fn test_cat_4d_dim2_with_copy() -> Result<()> {
    // Test that simulates the exact KV cache pattern:
    // 1. Create k_cache from k.copy()
    // 2. Cat k_cache with new k
    // This is what happens in the attention layer
    let device = get_device();

    // Step 0: k is created and k_cache = k.copy()
    let k_step0_data: Vec<f32> = (0..1280).map(|i| i as f32).collect();
    let k_step0: Tensor<f32, Device> =
        Tensor::from_vec(k_step0_data.clone(), vec![1, 4, 5, 64], &device)?;
    let k_cache = k_step0.copy()?;

    // Verify k_cache has correct values
    let k_cache_data = k_cache.to_vec()?;
    assert_eq!(
        k_cache_data[0..5],
        [0.0, 1.0, 2.0, 3.0, 4.0],
        "k_cache first 5 should match k_step0"
    );

    // Step 1: new k is created, cat with k_cache
    let k_step1: Tensor<f32, Device> = Tensor::full(9999.0, vec![1, 4, 1, 64], &device)?;
    let k_cat = Tensor::cat(&[&k_cache, &k_step1], 2)?;

    // Verify k_cat preserves k_cache values
    let k_cat_data = k_cat.to_vec()?;
    assert_eq!(k_cat_data[0], 0.0, "after cat, first element should be 0.0");
    assert_eq!(k_cat_data[1], 1.0, "after cat, second element should be 1.0");
    assert_eq!(k_cat_data[2], 2.0, "after cat, third element should be 2.0");
    assert_eq!(k_cat_data[3], 3.0, "after cat, fourth element should be 3.0");
    assert_eq!(k_cat_data[4], 4.0, "after cat, fifth element should be 4.0");

    // Also check the new values at position 5*64 = 320
    assert_eq!(k_cat_data[320], 9999.0, "new value at pos 320 should be 9999.0");

    // Step 2: new k_cache from k_cat.copy(), cat again
    let k_cache_step1 = k_cat.copy()?;
    let k_step2: Tensor<f32, Device> = Tensor::full(8888.0, vec![1, 4, 1, 64], &device)?;
    let k_cat2 = Tensor::cat(&[&k_cache_step1, &k_step2], 2)?;

    // Verify values are still preserved
    let k_cat2_data = k_cat2.to_vec()?;
    assert_eq!(k_cat2_data[0], 0.0, "after second cat, first element should be 0.0");
    assert_eq!(k_cat2_data[1], 1.0, "after second cat, second element should be 1.0");
    assert_eq!(k_cat2_data[320], 9999.0, "after second cat, pos 320 should be 9999.0");
    // New position: 6*64 = 384
    assert_eq!(k_cat2_data[384], 8888.0, "new value at pos 384 should be 8888.0");

    Ok(())
}

#[test]
fn test_cat_after_transpose() -> Result<()> {
    // This test mimics the attention layer pattern more closely:
    // k is created, reshaped, transposed, then cached
    let device = get_device();

    // Shape before reshape: [1, 5, 256] (batch, seq, num_kv_heads * head_dim)
    let k_linear: Vec<f32> = (0..(5 * 256)).map(|i| i as f32).collect();
    let k: Tensor<f32, Device> = Tensor::from_vec(k_linear, vec![1, 5, 256], &device)?;

    // Reshape to [1, 5, 4, 64]
    let k = k.reshape(vec![1, 5, 4, 64])?;

    // Transpose (1, 2) to get [1, 4, 5, 64]
    let k = k.transpose(1, 2)?.contiguous()?;
    assert_eq!(k.dims(), &[1, 4, 5, 64]);

    // Create k_cache as a copy
    let k_cache = k.copy()?;
    let k_cache_data = k_cache.to_vec()?;

    // Verify the first 5 values of k_cache
    let first5: Vec<f32> = k_cache_data[0..5].to_vec();
    eprintln!("After transpose, k_cache first5: {:?}", first5);

    // Now simulate step 1: create a new k and cat
    let k_new_linear: Vec<f32> = (9000..(9000 + 256)).map(|i| i as f32).collect();
    let k_new: Tensor<f32, Device> = Tensor::from_vec(k_new_linear, vec![1, 1, 256], &device)?;
    let k_new = k_new.reshape(vec![1, 1, 4, 64])?;
    let k_new = k_new.transpose(1, 2)?.contiguous()?;
    assert_eq!(k_new.dims(), &[1, 4, 1, 64]);

    // Cat k_cache with k_new
    let k_cat = Tensor::cat(&[&k_cache, &k_new], 2)?;
    assert_eq!(k_cat.dims(), &[1, 4, 6, 64]);

    let k_cat_data = k_cat.to_vec()?;
    let cat_first5: Vec<f32> = k_cat_data[0..5].to_vec();
    eprintln!("After cat, k_cat first5: {:?}", cat_first5);

    // The first 5 values should match k_cache first 5 values
    for i in 0..5 {
        assert_eq!(
            k_cache_data[i], k_cat_data[i],
            "Mismatch at position {}: k_cache has {}, k_cat has {}",
            i, k_cache_data[i], k_cat_data[i]
        );
    }

    Ok(())
}

#[test]
fn test_cat_after_multiple_transpose() -> Result<()> {
    // Even more complex: multiple transposes
    let device = get_device();

    let data: Vec<f32> = (0..1280).map(|i| i as f32).collect();
    let t: Tensor<f32, Device> = Tensor::from_vec(data, vec![1, 4, 5, 64], &device)?;

    // Do a transpose, then transpose back (should be identity)
    let t2 = t.transpose(1, 2)?;
    assert_eq!(t2.dims(), &[1, 5, 4, 64]);

    let t3 = t2.transpose(1, 2)?.contiguous()?;
    assert_eq!(t3.dims(), &[1, 4, 5, 64]);

    // Copy and cat
    let t_cache = t3.copy()?;
    let t_new: Tensor<f32, Device> = Tensor::full(9999.0, vec![1, 4, 1, 64], &device)?;
    let t_cat = Tensor::cat(&[&t_cache, &t_new], 2)?;

    let t_cat_data = t_cat.to_vec()?;

    // First 5 values should be preserved from original
    assert_eq!(t_cat_data[0], 0.0);
    assert_eq!(t_cat_data[1], 1.0);
    assert_eq!(t_cat_data[2], 2.0);
    assert_eq!(t_cat_data[3], 3.0);
    assert_eq!(t_cat_data[4], 4.0);

    Ok(())
}

#[test]
fn test_cat_with_rope() -> Result<()> {
    // Full test including rope, which is what the llama model does
    let device = get_device();

    // Shape: [b, h, t, d] = [1, 4, 5, 64], d_over_2 = 32
    // Step 0: Create k, transpose, rope, copy
    let k_data: Vec<f32> = (0..1280).map(|i| (i as f32) / 100.0).collect();
    let k: Tensor<f32, Device> = Tensor::from_vec(k_data, vec![1, 5, 4, 64], &device)?;
    let k = k.transpose(1, 2)?.contiguous()?; // -> [1, 4, 5, 64]

    // Create cos/sin for rope - shape should be [max_pos, d/2] = [10, 32]
    let cos: Tensor<f32, Device> = Tensor::full(1.0, vec![10, 32], &device)?;
    let sin: Tensor<f32, Device> = Tensor::full(0.0, vec![10, 32], &device)?;

    // Apply rope at pos=0, t=5 -> needs cos/sin from pos 0 to 4
    let k = k.rope(&cos, &sin, 0)?; // With cos=1, sin=0, rope should be identity

    // Create cache
    let k_cache = k.copy()?;
    let k_cache_first5: Vec<f32> = k_cache.to_vec()?[0..5].to_vec();
    eprintln!("Step 0 k_cache first5: {:?}", k_cache_first5);

    // Step 1: Create new k, transpose, rope, cat with cache
    let k_new_data: Vec<f32> = (9000..9256).map(|i| (i as f32) / 100.0).collect();
    let k_new: Tensor<f32, Device> = Tensor::from_vec(k_new_data, vec![1, 1, 4, 64], &device)?;
    let k_new = k_new.transpose(1, 2)?.contiguous()?; // -> [1, 4, 1, 64]

    // Apply rope at pos=5, t=1 -> needs cos/sin at pos 5
    let k_new = k_new.rope(&cos, &sin, 5)?;

    // Cat
    let k_cat = Tensor::cat(&[&k_cache, &k_new], 2)?;
    let k_cat_first5: Vec<f32> = k_cat.to_vec()?[0..5].to_vec();
    eprintln!("Step 1 k_cat first5: {:?}", k_cat_first5);

    // First 5 values should match k_cache
    for i in 0..5 {
        assert!(
            (k_cache_first5[i] - k_cat_first5[i]).abs() < 1e-5,
            "Mismatch at {}: cache={}, cat={}",
            i,
            k_cache_first5[i],
            k_cat_first5[i]
        );
    }

    Ok(())
}

#[test]
fn test_rope_position_offset() -> Result<()> {
    // Test that rope correctly handles non-zero position offsets
    // This tests the bug where CUDA rope ignores the pos parameter
    let device = get_device();

    // Create input tensor: [1, 1, 1, 4] (b=1, h=1, t=1, d=4)
    let x: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 1, 4], &device)?;

    // Create cos/sin with different values at different positions
    // Shape: [10, 2] (max_pos=10, d/2=2)
    // Position 0: cos=[1, 1], sin=[0, 0]  -> identity
    // Position 1: cos=[0, 0], sin=[1, 1]  -> 90 degree rotation
    let cos_data = vec![
        1.0, 1.0, // pos 0: identity
        0.0, 0.0, // pos 1: 90 degree
        -1.0, -1.0, // pos 2: 180 degree
        0.0, 0.0, // pos 3
        1.0, 1.0, // pos 4
        0.5, 0.5, // pos 5
        0.0, 0.0, // pos 6
        0.0, 0.0, // pos 7
        0.0, 0.0, // pos 8
        0.0, 0.0, // pos 9
    ];
    let sin_data = vec![
        0.0, 0.0, // pos 0: identity
        1.0, 1.0, // pos 1: 90 degree
        0.0, 0.0, // pos 2
        0.0, 0.0, // pos 3
        0.0, 0.0, // pos 4
        0.866, 0.866, // pos 5: ~60 degree
        0.0, 0.0, // pos 6
        0.0, 0.0, // pos 7
        0.0, 0.0, // pos 8
        0.0, 0.0, // pos 9
    ];
    let cos: Tensor<f32, Device> = Tensor::from_vec(cos_data, vec![10, 2], &device)?;
    let sin: Tensor<f32, Device> = Tensor::from_vec(sin_data, vec![10, 2], &device)?;

    // Apply rope at position 0 (identity): dst = [x1, x2, x3, x4]
    let y0 = x.rope(&cos, &sin, 0)?;
    let y0_data = y0.to_vec()?;
    eprintln!("rope at pos 0: {:?}", y0_data);
    // With cos=1, sin=0, output should equal input
    assert!((y0_data[0] - 1.0).abs() < 1e-5, "pos 0 element 0");
    assert!((y0_data[1] - 2.0).abs() < 1e-5, "pos 0 element 1");
    assert!((y0_data[2] - 3.0).abs() < 1e-5, "pos 0 element 2");
    assert!((y0_data[3] - 4.0).abs() < 1e-5, "pos 0 element 3");

    // Apply rope at position 1 (90 degree rotation):
    // dst[0] = x1*0 - x3*1 = -3
    // dst[2] = x1*1 + x3*0 = 1
    // dst[1] = x2*0 - x4*1 = -4
    // dst[3] = x2*1 + x4*0 = 2
    let y1 = x.rope(&cos, &sin, 1)?;
    let y1_data = y1.to_vec()?;
    eprintln!("rope at pos 1: {:?}", y1_data);
    // If the pos parameter works correctly, this should NOT equal the identity
    // If CUDA ignores pos, it will still use pos=0 values (identity)
    let is_identity = (y1_data[0] - 1.0).abs() < 1e-5 && (y1_data[1] - 2.0).abs() < 1e-5;
    assert!(!is_identity, "rope at pos=1 should NOT be identity, got {:?}", y1_data);

    Ok(())
}

// =============================================================================
// Argmin operations
// =============================================================================

#[test]
fn test_argmin_1d() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> = Tensor::from_vec(vec![3.0, 1.0, 4.0, 1.0, 5.0], vec![5], &device)?;
    let argmin = a.argmin(0)?;
    assert_eq!(argmin.dims(), &[1]);
    // First occurrence of min value 1.0 is at index 1
    assert_eq!(argmin.to_vec()?, vec![1i64]);
    Ok(())
}

#[test]
fn test_argmin_2d_dim0() -> Result<()> {
    let device = get_device();
    // Shape [3, 4]
    let a: Tensor<f32, Device> = Tensor::from_vec(
        vec![1.0, 5.0, 3.0, 4.0, 8.0, 2.0, 7.0, 6.0, 9.0, 0.0, 1.0, 2.0],
        vec![3, 4],
        &device,
    )?;
    // Argmin along dim 0 -> [4]
    // col 0: argmin(1, 8, 9) = 0
    // col 1: argmin(5, 2, 0) = 2
    // col 2: argmin(3, 7, 1) = 2
    // col 3: argmin(4, 6, 2) = 2
    let argmin = a.argmin(0)?;
    assert_eq!(argmin.dims(), &[4]);
    assert_eq!(argmin.to_vec()?, vec![0i64, 2, 2, 2]);
    Ok(())
}

#[test]
fn test_argmin_2d_dim1() -> Result<()> {
    let device = get_device();
    // Shape [3, 4]
    let a: Tensor<f32, Device> = Tensor::from_vec(
        vec![1.0, 5.0, 3.0, 4.0, 8.0, 2.0, 7.0, 6.0, 9.0, 0.0, 1.0, 2.0],
        vec![3, 4],
        &device,
    )?;
    // Argmin along dim 1 -> [3]
    // row 0: argmin(1, 5, 3, 4) = 0
    // row 1: argmin(8, 2, 7, 6) = 1
    // row 2: argmin(9, 0, 1, 2) = 1
    let argmin = a.argmin(1)?;
    assert_eq!(argmin.dims(), &[3]);
    assert_eq!(argmin.to_vec()?, vec![0i64, 1, 1]);
    Ok(())
}

#[test]
fn test_argmin_3d() -> Result<()> {
    let device = get_device();
    // Shape [2, 3, 4]
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let a: Tensor<f32, Device> = Tensor::from_vec(data, vec![2, 3, 4], &device)?;

    // Argmin along dim 1 (middle dimension) -> [2, 4]
    let argmin = a.argmin(1)?;
    assert_eq!(argmin.dims(), &[2, 4]);
    // For each position, the min is in the first row (index 0)
    assert_eq!(argmin.to_vec()?, vec![0i64, 0, 0, 0, 0, 0, 0, 0]);
    Ok(())
}

// =============================================================================
// Sum operations
// =============================================================================

#[test]
fn test_sum_keepdim_1d() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5], &device)?;
    let sum = a.sum_keepdim(vec![0])?;
    assert_eq!(sum.dims(), &[1]);
    assert_eq!(sum.to_vec()?, vec![15.0]);
    Ok(())
}

#[test]
fn test_sum_keepdim_2d_dim0() -> Result<()> {
    let device = get_device();
    // Shape [3, 4]
    let a: Tensor<f32, Device> = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![3, 4],
        &device,
    )?;
    // Sum along dim 0 -> [1, 4]
    let sum = a.sum_keepdim(vec![0])?;
    assert_eq!(sum.dims(), &[1, 4]);
    // Column sums: 1+5+9=15, 2+6+10=18, 3+7+11=21, 4+8+12=24
    assert_eq!(sum.to_vec()?, vec![15.0, 18.0, 21.0, 24.0]);
    Ok(())
}

#[test]
fn test_sum_keepdim_2d_dim1() -> Result<()> {
    let device = get_device();
    // Shape [3, 4]
    let a: Tensor<f32, Device> = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![3, 4],
        &device,
    )?;
    // Sum along dim 1 -> [3, 1]
    let sum = a.sum_keepdim(vec![1])?;
    assert_eq!(sum.dims(), &[3, 1]);
    // Row sums: 1+2+3+4=10, 5+6+7+8=26, 9+10+11+12=42
    assert_eq!(sum.to_vec()?, vec![10.0, 26.0, 42.0]);
    Ok(())
}

#[test]
fn test_sum_keepdim_3d() -> Result<()> {
    let device = get_device();
    // Shape [2, 3, 4]
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let a: Tensor<f32, Device> = Tensor::from_vec(data, vec![2, 3, 4], &device)?;

    // Sum along dim 1 -> [2, 1, 4]
    let sum = a.sum_keepdim(vec![1])?;
    assert_eq!(sum.dims(), &[2, 1, 4]);
    // Batch 0: [[1,2,3,4], [5,6,7,8], [9,10,11,12]] -> sum = [15, 18, 21, 24]
    // Batch 1: [[13,14,15,16], [17,18,19,20], [21,22,23,24]] -> sum = [51, 54, 57, 60]
    assert_eq!(sum.to_vec()?, vec![15.0, 18.0, 21.0, 24.0, 51.0, 54.0, 57.0, 60.0]);
    Ok(())
}

#[test]
fn test_sum_keepdim_f16() -> Result<()> {
    let device = get_device();
    let data: Vec<half::f16> =
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_iter().map(half::f16::from_f32).collect();
    let a: Tensor<half::f16, Device> = Tensor::from_vec(data, vec![2, 3], &device)?;
    let sum = a.sum_keepdim(vec![1])?;
    assert_eq!(sum.dims(), &[2, 1]);
    let result: Vec<f32> = sum.to_vec()?.iter().map(|x| x.to_f32()).collect();
    // Row 0: 1+2+3=6, Row 1: 4+5+6=15
    assert!((result[0] - 6.0).abs() < 0.1);
    assert!((result[1] - 15.0).abs() < 0.1);
    Ok(())
}

// =============================================================================
// Broadcast binary operations
// =============================================================================

#[test]
fn test_broadcast_add_same_shape() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], &device)?;
    let b: Tensor<f32, Device> =
        Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2], &device)?;
    let c = a.broadcast_add(&b)?;
    assert_eq!(c.dims(), &[2, 2]);
    assert_eq!(c.to_vec()?, vec![11.0, 22.0, 33.0, 44.0]);
    Ok(())
}

#[test]
fn test_broadcast_add_row() -> Result<()> {
    let device = get_device();
    // a: [2, 3], b: [3] -> broadcast b along first dim
    let a: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], &device)?;
    let b: Tensor<f32, Device> = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3], &device)?;
    let c = a.broadcast_add(&b)?;
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(c.to_vec()?, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    Ok(())
}

#[test]
fn test_broadcast_add_col() -> Result<()> {
    let device = get_device();
    // a: [2, 3], b: [2, 1] -> broadcast b along second dim
    let a: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], &device)?;
    let b: Tensor<f32, Device> = Tensor::from_vec(vec![10.0, 20.0], vec![2, 1], &device)?;
    let c = a.broadcast_add(&b)?;
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(c.to_vec()?, vec![11.0, 12.0, 13.0, 24.0, 25.0, 26.0]);
    Ok(())
}

#[test]
fn test_broadcast_mul_row() -> Result<()> {
    let device = get_device();
    // a: [2, 3], b: [3] -> broadcast b along first dim
    let a: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], &device)?;
    let b: Tensor<f32, Device> = Tensor::from_vec(vec![2.0, 3.0, 4.0], vec![3], &device)?;
    let c = a.broadcast_mul(&b)?;
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(c.to_vec()?, vec![2.0, 6.0, 12.0, 8.0, 15.0, 24.0]);
    Ok(())
}

#[test]
fn test_broadcast_mul_col() -> Result<()> {
    let device = get_device();
    // a: [2, 3], b: [2, 1] -> broadcast b along second dim
    let a: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], &device)?;
    let b: Tensor<f32, Device> = Tensor::from_vec(vec![2.0, 3.0], vec![2, 1], &device)?;
    let c = a.broadcast_mul(&b)?;
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(c.to_vec()?, vec![2.0, 4.0, 6.0, 12.0, 15.0, 18.0]);
    Ok(())
}

#[test]
fn test_broadcast_sub() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> =
        Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 3], &device)?;
    let b: Tensor<f32, Device> = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3], &device)?;
    let c = a.broadcast_sub(&b)?;
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(c.to_vec()?, vec![9.0, 18.0, 27.0, 39.0, 48.0, 57.0]);
    Ok(())
}

#[test]
fn test_broadcast_div() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> =
        Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 3], &device)?;
    let b: Tensor<f32, Device> = Tensor::from_vec(vec![2.0, 4.0, 5.0], vec![3], &device)?;
    let c = a.broadcast_div(&b)?;
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(c.to_vec()?, vec![5.0, 5.0, 6.0, 20.0, 12.5, 12.0]);
    Ok(())
}

#[test]
fn test_broadcast_3d() -> Result<()> {
    let device = get_device();
    // a: [2, 2, 3], b: [3] -> broadcast b across batch and row dims
    let a: Tensor<f32, Device> = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![2, 2, 3],
        &device,
    )?;
    let b: Tensor<f32, Device> = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3], &device)?;
    let c = a.broadcast_add(&b)?;
    assert_eq!(c.dims(), &[2, 2, 3]);
    assert_eq!(
        c.to_vec()?,
        vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0, 17.0, 28.0, 39.0, 20.0, 31.0, 42.0]
    );
    Ok(())
}

#[test]
fn test_broadcast_maximum() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 5.0, 3.0, 8.0, 2.0, 6.0], vec![2, 3], &device)?;
    let b: Tensor<f32, Device> = Tensor::from_vec(vec![4.0, 4.0, 4.0], vec![3], &device)?;
    let c = a.broadcast_maximum(&b)?;
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(c.to_vec()?, vec![4.0, 5.0, 4.0, 8.0, 4.0, 6.0]);
    Ok(())
}

#[test]
fn test_broadcast_minimum() -> Result<()> {
    let device = get_device();
    let a: Tensor<f32, Device> =
        Tensor::from_vec(vec![1.0, 5.0, 3.0, 8.0, 2.0, 6.0], vec![2, 3], &device)?;
    let b: Tensor<f32, Device> = Tensor::from_vec(vec![4.0, 4.0, 4.0], vec![3], &device)?;
    let c = a.broadcast_minimum(&b)?;
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(c.to_vec()?, vec![1.0, 4.0, 3.0, 4.0, 2.0, 4.0]);
    Ok(())
}

#[test]
fn test_broadcast_f16() -> Result<()> {
    let device = get_device();
    let a_data: Vec<half::f16> =
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_iter().map(half::f16::from_f32).collect();
    let b_data: Vec<half::f16> =
        vec![10.0, 20.0, 30.0].into_iter().map(half::f16::from_f32).collect();
    let a: Tensor<half::f16, Device> = Tensor::from_vec(a_data, vec![2, 3], &device)?;
    let b: Tensor<half::f16, Device> = Tensor::from_vec(b_data, vec![3], &device)?;
    let c = a.broadcast_add(&b)?;
    assert_eq!(c.dims(), &[2, 3]);
    let result: Vec<f32> = c.to_vec()?.iter().map(|x| x.to_f32()).collect();
    let expected = [11.0, 22.0, 33.0, 14.0, 25.0, 36.0];
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 0.1, "Expected {} but got {}", e, r);
    }
    Ok(())
}

// =============================================================================
// Conv1d operations
// =============================================================================

#[test]
fn test_conv1d_simple() -> Result<()> {
    let device = get_device();
    // Input: (batch=1, in_channels=1, length=5)
    // Kernel: (out_channels=1, in_channels=1, kernel_size=3)
    let input: Tensor<f32, Device> =
        Tensor::from_vec(vec![1., 2., 3., 4., 5.], vec![1, 1, 5], &device)?;
    let kernel: Tensor<f32, Device> = Tensor::from_vec(vec![1., 0., -1.], vec![1, 1, 3], &device)?;

    let output = input.conv1d(&kernel, None, 1, 0, 1, 1)?;
    assert_eq!(output.dims(), &[1, 1, 3]);
    // output[i] = input[i]*1 + input[i+1]*0 + input[i+2]*(-1)
    assert_eq!(output.to_vec()?, vec![-2., -2., -2.]);
    Ok(())
}

#[test]
fn test_conv1d_with_padding() -> Result<()> {
    let device = get_device();
    let input: Tensor<f32, Device> =
        Tensor::from_vec(vec![1., 2., 3., 4.], vec![1, 1, 4], &device)?;
    let kernel: Tensor<f32, Device> = Tensor::from_vec(vec![1., 1., 1.], vec![1, 1, 3], &device)?;

    let output = input.conv1d(&kernel, None, 1, 1, 1, 1)?;
    assert_eq!(output.dims(), &[1, 1, 4]);
    // With padding=1: [0, 1, 2, 3, 4, 0]
    assert_eq!(output.to_vec()?, vec![3., 6., 9., 7.]);
    Ok(())
}

#[test]
fn test_conv1d_with_stride() -> Result<()> {
    let device = get_device();
    let input: Tensor<f32, Device> =
        Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], vec![1, 1, 6], &device)?;
    let kernel: Tensor<f32, Device> = Tensor::from_vec(vec![1., 1.], vec![1, 1, 2], &device)?;

    let output = input.conv1d(&kernel, None, 2, 0, 1, 1)?;
    assert_eq!(output.dims(), &[1, 1, 3]);
    assert_eq!(output.to_vec()?, vec![3., 7., 11.]);
    Ok(())
}

#[test]
fn test_conv1d_multi_channel() -> Result<()> {
    let device = get_device();
    // Input: (batch=1, in_channels=2, length=4)
    // Kernel: (out_channels=1, in_channels=2, kernel_size=2)
    let input: Tensor<f32, Device> =
        Tensor::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![1, 2, 4], &device)?;
    let kernel: Tensor<f32, Device> =
        Tensor::from_vec(vec![1., 1., 1., 1.], vec![1, 2, 2], &device)?;

    let output = input.conv1d(&kernel, None, 1, 0, 1, 1)?;
    assert_eq!(output.dims(), &[1, 1, 3]);
    // out[0] = (1+2) + (5+6) = 14
    // out[1] = (2+3) + (6+7) = 18
    // out[2] = (3+4) + (7+8) = 22
    assert_eq!(output.to_vec()?, vec![14., 18., 22.]);
    Ok(())
}

#[test]
fn test_conv1d_batch() -> Result<()> {
    let device = get_device();
    // Input: (batch=2, in_channels=1, length=4)
    let input: Tensor<f32, Device> =
        Tensor::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 1, 4], &device)?;
    let kernel: Tensor<f32, Device> = Tensor::from_vec(vec![1., 1.], vec![1, 1, 2], &device)?;

    let output = input.conv1d(&kernel, None, 1, 0, 1, 1)?;
    assert_eq!(output.dims(), &[2, 1, 3]);
    // Batch 0: [1+2, 2+3, 3+4] = [3, 5, 7]
    // Batch 1: [5+6, 6+7, 7+8] = [11, 13, 15]
    assert_eq!(output.to_vec()?, vec![3., 5., 7., 11., 13., 15.]);
    Ok(())
}

#[test]
fn test_conv1d_f16() -> Result<()> {
    let device = get_device();
    let input_data: Vec<half::f16> =
        vec![1., 2., 3., 4., 5.].into_iter().map(half::f16::from_f32).collect();
    let kernel_data: Vec<half::f16> =
        vec![1., 0., -1.].into_iter().map(half::f16::from_f32).collect();
    let input: Tensor<half::f16, Device> = Tensor::from_vec(input_data, vec![1, 1, 5], &device)?;
    let kernel: Tensor<half::f16, Device> = Tensor::from_vec(kernel_data, vec![1, 1, 3], &device)?;

    let output = input.conv1d(&kernel, None, 1, 0, 1, 1)?;
    assert_eq!(output.dims(), &[1, 1, 3]);
    let result: Vec<f32> = output.to_vec()?.iter().map(|x| x.to_f32()).collect();
    for (r, e) in result.iter().zip([-2., -2., -2.].iter()) {
        assert!((r - e).abs() < 0.1, "Expected {} but got {}", e, r);
    }
    Ok(())
}

// =============================================================================
// Conv transpose 1d operations
// =============================================================================

#[test]
fn test_conv_transpose1d_simple() -> Result<()> {
    let device = get_device();
    // Input: (batch=1, in_channels=1, length=3)
    // Kernel: (in_channels=1, out_channels=1, kernel_size=3)
    let input: Tensor<f32, Device> = Tensor::from_vec(vec![1., 2., 3.], vec![1, 1, 3], &device)?;
    let kernel: Tensor<f32, Device> = Tensor::from_vec(vec![1., 1., 1.], vec![1, 1, 3], &device)?;

    let output = input.conv_transpose1d(&kernel, None, 1, 0, 0, 1)?;
    // out_length = (3 - 1) * 1 + 3 = 5
    assert_eq!(output.dims(), &[1, 1, 5]);
    // Each input contributes to 3 output positions
    // out[0] = 1*1 = 1
    // out[1] = 1*1 + 2*1 = 3
    // out[2] = 1*1 + 2*1 + 3*1 = 6
    // out[3] = 2*1 + 3*1 = 5
    // out[4] = 3*1 = 3
    assert_eq!(output.to_vec()?, vec![1., 3., 6., 5., 3.]);
    Ok(())
}

#[test]
fn test_conv_transpose1d_with_stride() -> Result<()> {
    let device = get_device();
    // Input: (batch=1, in_channels=1, length=2)
    // Kernel: (in_channels=1, out_channels=1, kernel_size=2)
    // Stride=2
    let input: Tensor<f32, Device> = Tensor::from_vec(vec![1., 2.], vec![1, 1, 2], &device)?;
    let kernel: Tensor<f32, Device> = Tensor::from_vec(vec![1., 1.], vec![1, 1, 2], &device)?;

    let output = input.conv_transpose1d(&kernel, None, 2, 0, 0, 1)?;
    // out_length = (2 - 1) * 2 + 2 = 4
    assert_eq!(output.dims(), &[1, 1, 4]);
    // out[0] = 1*1 = 1
    // out[1] = 1*1 = 1
    // out[2] = 2*1 = 2
    // out[3] = 2*1 = 2
    assert_eq!(output.to_vec()?, vec![1., 1., 2., 2.]);
    Ok(())
}

#[test]
fn test_conv_transpose1d_multi_channel() -> Result<()> {
    let device = get_device();
    // Input: (batch=1, in_channels=2, length=2)
    // Kernel: (in_channels=2, out_channels=1, kernel_size=2)
    let input: Tensor<f32, Device> =
        Tensor::from_vec(vec![1., 2., 3., 4.], vec![1, 2, 2], &device)?;
    let kernel: Tensor<f32, Device> =
        Tensor::from_vec(vec![1., 1., 1., 1.], vec![2, 1, 2], &device)?;

    let output = input.conv_transpose1d(&kernel, None, 1, 0, 0, 1)?;
    assert_eq!(output.dims(), &[1, 1, 3]);
    // Channel 0 contributes: [1, 1+2, 2] = [1, 3, 2]
    // Channel 1 contributes: [3, 3+4, 4] = [3, 7, 4]
    // Total: [4, 10, 6]
    assert_eq!(output.to_vec()?, vec![4., 10., 6.]);
    Ok(())
}

#[test]
fn test_conv_transpose1d_batch() -> Result<()> {
    let device = get_device();
    // Input: (batch=2, in_channels=1, length=2)
    let input: Tensor<f32, Device> =
        Tensor::from_vec(vec![1., 2., 3., 4.], vec![2, 1, 2], &device)?;
    let kernel: Tensor<f32, Device> = Tensor::from_vec(vec![1., 1.], vec![1, 1, 2], &device)?;

    let output = input.conv_transpose1d(&kernel, None, 1, 0, 0, 1)?;
    assert_eq!(output.dims(), &[2, 1, 3]);
    // Batch 0: [1, 1+2, 2] = [1, 3, 2]
    // Batch 1: [3, 3+4, 4] = [3, 7, 4]
    assert_eq!(output.to_vec()?, vec![1., 3., 2., 3., 7., 4.]);
    Ok(())
}

#[test]
fn test_conv_transpose1d_f16() -> Result<()> {
    let device = get_device();
    let input_data: Vec<half::f16> =
        vec![1., 2., 3.].into_iter().map(half::f16::from_f32).collect();
    let kernel_data: Vec<half::f16> =
        vec![1., 1., 1.].into_iter().map(half::f16::from_f32).collect();
    let input: Tensor<half::f16, Device> = Tensor::from_vec(input_data, vec![1, 1, 3], &device)?;
    let kernel: Tensor<half::f16, Device> = Tensor::from_vec(kernel_data, vec![1, 1, 3], &device)?;

    let output = input.conv_transpose1d(&kernel, None, 1, 0, 0, 1)?;
    assert_eq!(output.dims(), &[1, 1, 5]);
    let result: Vec<f32> = output.to_vec()?.iter().map(|x| x.to_f32()).collect();
    let expected = [1., 3., 6., 5., 3.];
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 0.1, "Expected {} but got {}", e, r);
    }
    Ok(())
}

// =============================================================================
// FP8 quantization
// =============================================================================

#[test]
fn test_quantize_fp8_bf16() -> Result<()> {
    use xn::cuda_backend::quantization::Fp8Tensor;

    let device = get_device();
    let data: Vec<half::bf16> = [
        1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,
        -0.5, 0.25, -0.125, 0.0, 1.0, -1.0, 2.0, -2.0, 100.0, -100.0, 50.0, -50.0, 25.0, -25.0,
        12.5, -12.5,
    ]
    .iter()
    .map(|&v| half::bf16::from_f32(v))
    .collect();

    let t: Tensor<half::bf16, Device> = Tensor::from_vec(data.clone(), vec![4, 8], &device)?;
    let fp8 = Fp8Tensor::quantize(&t)?;

    let scale: Vec<f32> = device.stream().clone_dtoh(&fp8.scales)?;
    assert_eq!(scale.len(), 1);
    let expected_scale = 100.0f32 / 448.0;
    assert!(
        (scale[0] - expected_scale).abs() < 1e-5,
        "scale mismatch: got {} expected {}",
        scale[0],
        expected_scale,
    );

    // Dequantize back to bf16 and check round-trip error.
    let out: Tensor<half::bf16, Device> = fp8.dequantize()?;
    assert_eq!(out.dims(), &[4, 8]);
    let result: Vec<half::bf16> = out.to_vec()?;

    for (i, (&orig, &deq)) in data.iter().zip(result.iter()).enumerate() {
        let o = orig.to_f32();
        let d = deq.to_f32();
        let tol = f32::max(o.abs() * 0.1, 0.5);
        assert!(
            (o - d).abs() <= tol,
            "round-trip mismatch at index {i}: original {o}, dequantized {d}, tol {tol}",
        );
    }

    Ok(())
}

#[test]
fn test_quantize_fp8_f32() -> Result<()> {
    use xn::cuda_backend::quantization::Fp8Tensor;

    let device = get_device();
    let data: Vec<f32> = vec![
        1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,
        -0.5, 0.25, -0.125, 0.0, 1.0, -1.0, 2.0, -2.0, 100.0, -100.0, 50.0, -50.0, 25.0, -25.0,
        12.5, -12.5,
    ];

    let t: Tensor<f32, Device> = Tensor::from_vec(data.clone(), vec![4, 8], &device)?;
    let fp8 = Fp8Tensor::quantize(&t)?;

    let scale: Vec<f32> = device.stream().clone_dtoh(&fp8.scales)?;
    assert_eq!(scale.len(), 1);
    let expected_scale = 100.0f32 / 448.0;
    assert!(
        (scale[0] - expected_scale).abs() < 1e-5,
        "scale mismatch: got {} expected {}",
        scale[0],
        expected_scale,
    );

    // Dequantize back to f32 and check round-trip error.
    let out: Tensor<f32, Device> = fp8.dequantize()?;
    assert_eq!(out.dims(), &[4, 8]);
    let result: Vec<f32> = out.to_vec()?;

    for (i, (&orig, &deq)) in data.iter().zip(result.iter()).enumerate() {
        let tol = f32::max(orig.abs() * 0.1, 0.5);
        assert!(
            (orig - deq).abs() <= tol,
            "round-trip mismatch at index {i}: original {orig}, dequantized {deq}, tol {tol}",
        );
    }

    Ok(())
}

#[test]
fn test_quantize_fp8_f16() -> Result<()> {
    use xn::cuda_backend::quantization::Fp8Tensor;

    let device = get_device();
    let data: Vec<half::f16> = [
        1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,
        -0.5, 0.25, -0.125, 0.0, 1.0, -1.0, 2.0, -2.0, 100.0, -100.0, 50.0, -50.0, 25.0, -25.0,
        12.5, -12.5,
    ]
    .iter()
    .map(|&v| half::f16::from_f32(v))
    .collect();

    let t: Tensor<half::f16, Device> = Tensor::from_vec(data.clone(), vec![4, 8], &device)?;
    let fp8 = Fp8Tensor::quantize(&t)?;

    let scale: Vec<f32> = device.stream().clone_dtoh(&fp8.scales)?;
    assert_eq!(scale.len(), 1);
    let expected_scale = 100.0f32 / 448.0;
    assert!(
        (scale[0] - expected_scale).abs() < 1e-5,
        "scale mismatch: got {} expected {}",
        scale[0],
        expected_scale,
    );

    let out: Tensor<half::f16, Device> = fp8.dequantize()?;
    assert_eq!(out.dims(), &[4, 8]);
    let result: Vec<half::f16> = out.to_vec()?;

    for (i, (&orig, &deq)) in data.iter().zip(result.iter()).enumerate() {
        let o = orig.to_f32();
        let d = deq.to_f32();
        let tol = f32::max(o.abs() * 0.1, 0.5);
        assert!(
            (o - d).abs() <= tol,
            "round-trip mismatch at index {i}: original {o}, dequantized {d}, tol {tol}",
        );
    }

    Ok(())
}

#[test]
fn test_fp8_matmul_t() -> Result<()> {
    use xn::cuda_backend::quantization::Fp8Tensor;

    let device = get_device();

    // cuBLASLt FP8 requires sufficiently large dimensions for tensor core usage.
    // A[M, K], W[N, K] → C[M, N] = A × W^T
    const M: usize = 32;
    const K: usize = 64;
    const N: usize = 32;

    // Build A: each row i has value (i+1) in every column.
    let a_data: Vec<half::bf16> =
        (0..M * K).map(|idx| half::bf16::from_f32((idx / K + 1) as f32)).collect();

    // Build W as a scaled identity-like matrix: W[j, j] = 1.0 for j < N, rest 0.
    // This means C = A × W^T picks the first N columns of A (all identical per row).
    // So C[i, j] = (i+1) for all j.
    let w_data: Vec<half::bf16> = (0..N * K)
        .map(|idx| {
            let row = idx / K;
            let col = idx % K;
            if row == col { half::bf16::from_f32(1.0) } else { half::bf16::from_f32(0.0) }
        })
        .collect();

    let a_t: Tensor<half::bf16, Device> = Tensor::from_vec(a_data, vec![M, K], &device)?;
    let w_t: Tensor<half::bf16, Device> = Tensor::from_vec(w_data, vec![N, K], &device)?;

    let a_fp8 = Fp8Tensor::quantize(&a_t)?;
    let w_fp8 = Fp8Tensor::quantize(&w_t)?;

    let c: Tensor<half::bf16, Device> = a_fp8.matmul_t(&w_fp8)?;
    assert_eq!(c.dims(), &[M, N]);

    let result: Vec<half::bf16> = c.to_vec()?;

    // Expected: C[i, j] ≈ (i+1) for all j.
    for i in 0..M {
        let expected = (i + 1) as f32;
        for j in 0..N {
            let got = result[i * N + j].to_f32();
            let tol = f32::max(expected.abs() * 0.15, 1.0);
            assert!(
                (got - expected).abs() <= tol,
                "matmul mismatch at [{i}, {j}]: got {got}, expected {expected}, tol {tol}",
            );
        }
    }

    Ok(())
}

#[test]
fn test_quantize_fp8_per_token_bf16() -> Result<()> {
    use xn::cuda_backend::quantization::{Fp8ScaleMode, Fp8Tensor};

    let device = get_device();
    let data: Vec<half::bf16> = [
        1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,
        -0.5, 0.25, -0.125, 0.0, 1.0, -1.0, 2.0, -2.0, 100.0, -100.0, 50.0, -50.0, 25.0, -25.0,
        12.5, -12.5,
    ]
    .iter()
    .map(|&v| half::bf16::from_f32(v))
    .collect();

    let t: Tensor<half::bf16, Device> = Tensor::from_vec(data.clone(), vec![4, 8], &device)?;
    let fp8 = Fp8Tensor::quantize_per_token(&t)?;

    assert_eq!(fp8.scale_mode, Fp8ScaleMode::PerToken);

    // Per-token: should have 4 scales (one per row).
    let scales: Vec<f32> = device.stream().clone_dtoh(&fp8.scales)?;
    assert_eq!(scales.len(), 4);

    // Row 0: max abs = 8, scale = 8/448
    // Row 1: max abs = 80, scale = 80/448
    // Row 2: max abs = 2, scale = 2/448
    // Row 3: max abs = 100, scale = 100/448
    let expected_scales = [8.0f32 / 448.0, 80.0 / 448.0, 2.0 / 448.0, 100.0 / 448.0];
    for (i, (&got, &expected)) in scales.iter().zip(expected_scales.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-5,
            "scale[{i}] mismatch: got {got} expected {expected}",
        );
    }

    // Dequantize back and check round-trip error.
    let out: Tensor<half::bf16, Device> = fp8.dequantize()?;
    assert_eq!(out.dims(), &[4, 8]);
    let result: Vec<half::bf16> = out.to_vec()?;

    for (i, (&orig, &deq)) in data.iter().zip(result.iter()).enumerate() {
        let o = orig.to_f32();
        let d = deq.to_f32();
        let tol = f32::max(o.abs() * 0.1, 0.5);
        assert!(
            (o - d).abs() <= tol,
            "per-token round-trip mismatch at index {i}: original {o}, dequantized {d}, tol {tol}",
        );
    }

    Ok(())
}

#[test]
fn test_fp8_matmul_t_per_token() -> Result<()> {
    use xn::cuda_backend::quantization::Fp8Tensor;

    let device = get_device();

    // FP8 per-token matmul requires Hopper (compute capability >= 9.0).
    let (cc_major, _) = device.compute_cap()?;
    if cc_major < 9 {
        eprintln!("skipping test_fp8_matmul_t_per_token: requires compute capability >= 9.0");
        return Ok(());
    }

    // A[M, K], W[N, K] → C[M, N] = A × W^T
    const M: usize = 32;
    const K: usize = 64;
    const N: usize = 32;

    // Build A: each row i has value (i+1) in every column.
    let a_data: Vec<half::bf16> =
        (0..M * K).map(|idx| half::bf16::from_f32((idx / K + 1) as f32)).collect();

    // Build W as a scaled identity-like matrix: W[j, j] = 1.0 for j < N, rest 0.
    // C = A × W^T picks the first N columns of A → C[i, j] = (i+1) for all j.
    let w_data: Vec<half::bf16> = (0..N * K)
        .map(|idx| {
            let row = idx / K;
            let col = idx % K;
            if row == col { half::bf16::from_f32(1.0) } else { half::bf16::from_f32(0.0) }
        })
        .collect();

    let a_t: Tensor<half::bf16, Device> = Tensor::from_vec(a_data, vec![M, K], &device)?;
    let w_t: Tensor<half::bf16, Device> = Tensor::from_vec(w_data, vec![N, K], &device)?;

    // Both per-token.
    let a_fp8 = Fp8Tensor::quantize_per_token(&a_t)?;
    let w_fp8 = Fp8Tensor::quantize_per_token(&w_t)?;

    let c: Tensor<half::bf16, Device> = a_fp8.matmul_t(&w_fp8)?;
    assert_eq!(c.dims(), &[M, N]);

    let result: Vec<half::bf16> = c.to_vec()?;

    // Expected: C[i, j] ≈ (i+1) for all j.
    for i in 0..M {
        let expected = (i + 1) as f32;
        for j in 0..N {
            let got = result[i * N + j].to_f32();
            let tol = f32::max(expected.abs() * 0.15, 1.0);
            assert!(
                (got - expected).abs() <= tol,
                "per-token matmul mismatch at [{i}, {j}]: got {got}, expected {expected}, tol {tol}",
            );
        }
    }

    // Mixed scale modes must be rejected.
    let a_fp8_pt = Fp8Tensor::quantize_per_token(&a_t)?;
    let w_fp8_scalar = Fp8Tensor::quantize(&w_t)?;
    assert!(a_fp8_pt.matmul_t(&w_fp8_scalar).is_err(), "mixed scale modes should be rejected",);

    Ok(())
}

// =============================================================================
// q8_0 weights on the GPU
// =============================================================================
//
// The oracle is the *dequantized* weight multiplied on the CPU backend, not
// the original f32 weight: quantizing to 8 bits is lossy, and comparing
// against what the kernel actually holds isolates the kernel from the
// quantizer.

fn q8_assert_close(want: &[f32], got: &[f32], tol: f32) {
    assert_eq!(want.len(), got.len(), "length mismatch");
    for (i, (x, y)) in want.iter().zip(got.iter()).enumerate() {
        let d = (x - y).abs();
        let scale = x.abs().max(y.abs());
        assert!(d <= tol + tol * scale, "mismatch at {i}: cpu={x} cuda={y} (|d|={d})");
    }
}

fn q8_roundtrip(w: &[f32]) -> Vec<f32> {
    use xn::quantized::GgmlType;
    use xn::quantized::k_quants::BlockQ8_0;
    let mut blocks = vec![BlockQ8_0::zeros(); w.len() / 32];
    BlockQ8_0::from_float(w, &mut blocks).unwrap();
    let mut out = vec![0f32; w.len()];
    BlockQ8_0::to_float(&blocks, &mut out).unwrap();
    out
}

fn cmp_q8_matmul(m: usize, k: usize, n: usize) -> Result<()> {
    use xn::cuda_backend::q8::Q8Tensor;
    use xn::{CPU, Shape};
    let d = get_device();
    let w: Vec<f32> = (0..n * k)
        .map(|i| ((i % 71) as f32 - 35.0) * 0.013 * (1.0 + (i / k) as f32 * 0.1))
        .collect();
    let x: Vec<f32> = (0..m * k).map(|i| ((i % 53) as f32 - 26.0) * 0.021).collect();

    let wq = Q8Tensor::from_f32(&d, &w, &Shape::from((n, k)))?;
    let xt: Tensor<f32, Device> = Tensor::from_vec(x.clone(), (m, k), &d)?;
    let got = wq.matmul_t(&xt)?.to_vec()?;

    let wd = q8_roundtrip(&w);
    let wt: Tensor<f32, xn::CpuDevice> = Tensor::from_vec(wd, (n, k), &CPU)?;
    let xc: Tensor<f32, xn::CpuDevice> = Tensor::from_vec(x, (m, k), &CPU)?;
    let want = xc.matmul_t(&wt)?.to_vec()?;
    q8_assert_close(&want, &got, 1e-4);
    Ok(())
}

#[test]
fn test_q8_matmul_decode() -> Result<()> {
    // m == 1 takes the single-row kernel.
    for (k, n) in [(32, 4), (64, 1), (256, 7), (1024, 256), (1536, 512)] {
        cmp_q8_matmul(1, k, n)?;
    }
    Ok(())
}

#[test]
fn test_q8_matmul_rows() -> Result<()> {
    // 2..=16 takes the row-block kernels, including every MR variant and
    // partial row blocks.
    for (m, k, n) in [(2, 64, 8), (3, 256, 5), (4, 512, 64), (6, 1024, 130), (16, 128, 33)] {
        cmp_q8_matmul(m, k, n)?;
    }
    Ok(())
}

#[test]
fn test_q8_matmul_large_m() -> Result<()> {
    // 17..=128 walks m in blocks of up to 32 (r16 / r32 with a 2D grid);
    // past 128 the weight is dequantized and goes through cuBLAS.
    for (m, k, n) in [(17, 64, 4), (30, 288, 96), (64, 512, 128), (125, 1024, 96), (200, 96, 65)] {
        cmp_q8_matmul(m, k, n)?;
    }
    Ok(())
}

#[test]
fn test_q8_matmul_batched_shape() -> Result<()> {
    use xn::cuda_backend::q8::Q8Tensor;
    use xn::{CPU, Shape};
    let (b, t, k, n) = (2usize, 3usize, 128usize, 16usize);
    let d = get_device();
    let w: Vec<f32> = (0..n * k).map(|i| ((i % 29) as f32 - 14.0) * 0.02).collect();
    let x: Vec<f32> = (0..b * t * k).map(|i| ((i % 37) as f32 - 18.0) * 0.011).collect();
    let wq = Q8Tensor::from_f32(&d, &w, &Shape::from((n, k)))?;
    let xt: Tensor<f32, Device> = Tensor::from_vec(x.clone(), (b, t, k), &d)?;
    let out = wq.matmul_t(&xt)?;
    assert_eq!(out.dims(), &[b, t, n]);

    let wd = q8_roundtrip(&w);
    let wt: Tensor<f32, xn::CpuDevice> = Tensor::from_vec(wd, (n, k), &CPU)?;
    let xc: Tensor<f32, xn::CpuDevice> = Tensor::from_vec(x, (b, t, k), &CPU)?;
    q8_assert_close(&xc.matmul_t(&wt)?.to_vec()?, &out.to_vec()?, 1e-4);
    Ok(())
}

#[test]
fn test_q8_gguf_blocks_match_in_process_quantization() -> Result<()> {
    use xn::Shape;
    use xn::cuda_backend::q8::Q8Tensor;
    use xn::quantized::{GgmlDType, QTensor};
    let (k, n, m) = (256usize, 24usize, 3usize);
    let d = get_device();
    let w: Vec<f32> = (0..n * k).map(|i| ((i % 67) as f32 - 33.0) * 0.019).collect();
    let x: Vec<f32> = (0..m * k).map(|i| ((i % 41) as f32 - 20.0) * 0.027).collect();
    let qt = QTensor::quantize_f32(&w, &Shape::from((n, k)), GgmlDType::Q8_0)?;
    let from_blocks = Q8Tensor::from_qtensor(&d, &qt)?;
    let from_f32 = Q8Tensor::from_f32(&d, &w, &Shape::from((n, k)))?;
    let xt: Tensor<f32, Device> = Tensor::from_vec(x, (m, k), &d)?;
    assert_eq!(from_blocks.matmul_t(&xt)?.to_vec()?, from_f32.matmul_t(&xt)?.to_vec()?);
    Ok(())
}

#[test]
fn test_q8_linear_matches_dequantized() -> Result<()> {
    use xn::cuda_backend::q8::Q8F32;
    use xn::nn::Linear;
    use xn::{BackendQ, CPU};
    let d = get_device();
    let (k, n, m) = (256usize, 64usize, 2usize);
    let w: Vec<f32> = (0..n * k).map(|i| ((i % 61) as f32 - 30.0) * 0.017).collect();
    let bias: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();

    let wt: Tensor<f32, Device> = Tensor::from_vec(w.clone(), (n, k), &d)?;
    let bt: Tensor<f32, Device> = Tensor::from_vec(bias.clone(), (n,), &d)?;
    let lin = Linear::new(wt).with_bias(bt);
    let q = Q8F32::from_linear(lin)?;

    let x: Vec<f32> = (0..m * k).map(|i| ((i % 43) as f32 - 21.0) * 0.03).collect();
    let xt: Tensor<f32, Device> = Tensor::from_vec(x.clone(), (m, k), &d)?;
    let got = q.forward(&xt)?.to_vec()?;

    let wd = q8_roundtrip(&w);
    let wc: Tensor<f32, xn::CpuDevice> = Tensor::from_vec(wd, (n, k), &CPU)?;
    let bc: Tensor<f32, xn::CpuDevice> = Tensor::from_vec(bias, (n,), &CPU)?;
    let xc: Tensor<f32, xn::CpuDevice> = Tensor::from_vec(x, (m, k), &CPU)?;
    let want = xc.matmul_t(&wc)?.broadcast_add(&bc)?.to_vec()?;
    q8_assert_close(&want, &got, 1e-4);
    Ok(())
}

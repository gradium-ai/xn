use xn::{Backend, Result, Tensor, TensorView};

/// Macro to generate tests for the CPU backend and any enabled GPU backends.
/// Each test function takes a device reference and runs the test logic.
macro_rules! test_all_backends {
    ($test_name:ident, $test_fn:ident) => {
        paste::paste! {
            #[test]
            fn [<$test_name _cpu>]() -> Result<()> {
                $test_fn(&xn::CPU)
            }

            #[cfg(feature = "cuda")]
            #[test]
            fn [<$test_name _cuda>]() -> Result<()> {
                let device = xn::cuda_backend::Device::new(0)?;
                $test_fn(&device)
            }

            #[cfg(feature = "metal")]
            #[test]
            fn [<$test_name _metal>]() -> Result<()> {
                let device = xn::metal_backend::Device::new(0)?;
                $test_fn(&device)
            }

            #[cfg(feature = "vulkan")]
            #[test]
            fn [<$test_name _vulkan>]() -> Result<()> {
                let device = xn::vulkan_backend::Device::new(0)?;
                $test_fn(&device)
            }

            #[cfg(feature = "webgpu")]
            #[test]
            fn [<$test_name _webgpu>]() -> Result<()> {
                $test_fn(&webgpu_device())
            }

            #[cfg(feature = "xla")]
            #[test]
            fn [<$test_name _xla>]() -> Result<()> {
                $test_fn(&xla_device())
            }
        }
    };
}

/// A single WebGPU device shared across the whole test binary. Unlike the
/// native backends, opening many independent `wgpu` devices (one per test
/// thread) and driving them concurrently is unreliable, so all tests share one
/// device. The backend serializes GPU submission internally and each tensor
/// owns a distinct buffer, so parallel tests interleave batches safely.
#[cfg(feature = "webgpu")]
fn webgpu_device() -> xn::webgpu_backend::Device {
    use std::sync::OnceLock;
    static DEVICE: OnceLock<xn::webgpu_backend::Device> = OnceLock::new();
    DEVICE.get_or_init(|| xn::webgpu_backend::Device::new(0).expect("init webgpu device")).clone()
}

/// A single XLA device (PJRT client) shared across the whole test binary:
/// creating a client is expensive and sharing it lets all tests reuse the
/// executable cache.
#[cfg(feature = "xla")]
fn xla_device() -> xn::xla_backend::Device {
    use std::sync::OnceLock;
    static DEVICE: OnceLock<xn::xla_backend::Device> = OnceLock::new();
    DEVICE.get_or_init(|| xn::xla_backend::Device::new(0).expect("init xla device")).clone()
}

// =============================================================================
// Cat tests
// =============================================================================

fn test_cat_dim0_impl<B: Backend>(dev: &B) -> Result<()> {
    // Two 2x3 tensors concatenated along dim 0 -> 4x3
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;
    let b: Tensor<f32, B> = Tensor::from_vec(vec![7., 8., 9., 10., 11., 12.], (2, 3), dev)?;

    let c = Tensor::cat(&[&a, &b], 0)?;
    assert_eq!(c.dims(), &[4, 3]);
    assert_eq!(c.to_vec()?, vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]);
    Ok(())
}
test_all_backends!(test_cat_dim0, test_cat_dim0_impl);

fn test_cat_dim1_impl<B: Backend>(dev: &B) -> Result<()> {
    // Two 2x3 tensors concatenated along dim 1 -> 2x6
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;
    let b: Tensor<f32, B> = Tensor::from_vec(vec![7., 8., 9., 10., 11., 12.], (2, 3), dev)?;

    let c = Tensor::cat(&[&a, &b], 1)?;
    assert_eq!(c.dims(), &[2, 6]);
    // Row 0: [1,2,3] ++ [7,8,9] = [1,2,3,7,8,9]
    // Row 1: [4,5,6] ++ [10,11,12] = [4,5,6,10,11,12]
    assert_eq!(c.to_vec()?, vec![1., 2., 3., 7., 8., 9., 4., 5., 6., 10., 11., 12.]);
    Ok(())
}
test_all_backends!(test_cat_dim1, test_cat_dim1_impl);

fn test_cat_3d_dim1_impl<B: Backend>(dev: &B) -> Result<()> {
    // Two 2x2x3 tensors concatenated along dim 1 -> 2x4x3
    let a: Tensor<f32, B> = Tensor::from_vec((1..=12).map(|x| x as f32).collect(), (2, 2, 3), dev)?;
    let b: Tensor<f32, B> =
        Tensor::from_vec((13..=24).map(|x| x as f32).collect(), (2, 2, 3), dev)?;

    let c = Tensor::cat(&[&a, &b], 1)?;
    assert_eq!(c.dims(), &[2, 4, 3]);
    // Batch 0: [[1,2,3],[4,5,6]] ++ [[13,14,15],[16,17,18]]
    // Batch 1: [[7,8,9],[10,11,12]] ++ [[19,20,21],[22,23,24]]
    assert_eq!(
        c.to_vec()?,
        vec![
            1., 2., 3., 4., 5., 6., 13., 14., 15., 16., 17., 18., // batch 0
            7., 8., 9., 10., 11., 12., 19., 20., 21., 22., 23., 24. // batch 1
        ]
    );
    Ok(())
}
test_all_backends!(test_cat_3d_dim1, test_cat_3d_dim1_impl);

// =============================================================================
// Reshape tests
// =============================================================================

fn test_reshape_impl<B: Backend>(dev: &B) -> Result<()> {
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;

    // Reshape to 3x2
    let b = a.reshape((3, 2))?;
    assert_eq!(b.dims(), &[3, 2]);
    assert_eq!(b.to_vec()?, vec![1., 2., 3., 4., 5., 6.]);

    // Reshape to 6
    let c = a.reshape((6,))?;
    assert_eq!(c.dims(), &[6]);

    // Reshape to 1x6
    let d = a.reshape((1, 6))?;
    assert_eq!(d.dims(), &[1, 6]);

    // Reshape to 1x2x3
    let e = a.reshape((1, 2, 3))?;
    assert_eq!(e.dims(), &[1, 2, 3]);
    Ok(())
}
test_all_backends!(test_reshape, test_reshape_impl);

fn test_reshape_with_hole_impl<B: Backend>(dev: &B) -> Result<()> {
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;

    // Reshape with inferred dimension
    let b = a.reshape((3, ()))?;
    assert_eq!(b.dims(), &[3, 2]);

    let c = a.reshape(((), 2))?;
    assert_eq!(c.dims(), &[3, 2]);

    let d = a.reshape((1, (), 3))?;
    assert_eq!(d.dims(), &[1, 2, 3]);
    Ok(())
}
test_all_backends!(test_reshape_with_hole, test_reshape_with_hole_impl);

// =============================================================================
// Index select tests
// =============================================================================

fn test_index_select_dim0_impl<B: Backend>(dev: &B) -> Result<()> {
    // Select rows from a 4x3 tensor
    let a: Tensor<f32, B> =
        Tensor::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.], (4, 3), dev)?;

    // Select rows 0, 2, 3
    let idx = Tensor::from_vec(vec![0i64, 2, 3], 3, dev)?;
    let b = a.index_select(&idx, 0)?;
    assert_eq!(b.dims(), &[3, 3]);
    assert_eq!(b.to_vec()?, vec![1., 2., 3., 7., 8., 9., 10., 11., 12.]);

    // Select with repetition
    let idx = Tensor::from_vec(vec![1i64, 1, 0], 3, dev)?;
    let c = a.index_select(&idx, 0)?;
    assert_eq!(c.dims(), &[3, 3]);
    assert_eq!(c.to_vec()?, vec![4., 5., 6., 4., 5., 6., 1., 2., 3.]);
    Ok(())
}
test_all_backends!(test_index_select_dim0, test_index_select_dim0_impl);

fn test_index_select_dim1_impl<B: Backend>(dev: &B) -> Result<()> {
    // Select columns from a 2x4 tensor
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8.], (2, 4), dev)?;

    // Select columns 0, 2
    let idx = Tensor::from_vec(vec![0i64, 2], 2, dev)?;
    let b = a.index_select(&idx, 1)?;
    assert_eq!(b.dims(), &[2, 2]);
    // Row 0: [1, 3], Row 1: [5, 7]
    assert_eq!(b.to_vec()?, vec![1., 3., 5., 7.]);
    Ok(())
}
test_all_backends!(test_index_select_dim1, test_index_select_dim1_impl);

fn test_index_select_3d_impl<B: Backend>(dev: &B) -> Result<()> {
    // 2x3x2 tensor
    let a: Tensor<f32, B> = Tensor::from_vec((1..=12).map(|x| x as f32).collect(), (2, 3, 2), dev)?;
    // Data layout:
    // Batch 0: [[1,2], [3,4], [5,6]]
    // Batch 1: [[7,8], [9,10], [11,12]]

    // Select along dim 1 (middle dimension)
    let idx = Tensor::from_vec(vec![0i64, 2], 2, dev)?;
    let b = a.index_select(&idx, 1)?;
    assert_eq!(b.dims(), &[2, 2, 2]);
    // Batch 0: [[1,2], [5,6]]
    // Batch 1: [[7,8], [11,12]]
    assert_eq!(b.to_vec()?, vec![1., 2., 5., 6., 7., 8., 11., 12.]);
    Ok(())
}
test_all_backends!(test_index_select_3d, test_index_select_3d_impl);

fn test_index_select_narrowed_indices_impl<B: Backend>(dev: &B) -> Result<()> {
    // Regression test: narrowing indices from index 0 produces a TensorView with
    // start_offset=0 and contiguous strides. contiguous() used to return a Tensor
    // sharing the original (larger) storage, causing index_select to process too
    // many indices.
    let a: Tensor<f32, B> = Tensor::from_vec(vec![10., 20., 30., 40., 50.], (5, 1), dev)?;

    // Create indices [0, 1, 2, 3, 4] then narrow to just [0, 1]
    let all_ids = Tensor::from_vec(vec![0i64, 1, 2, 3, 4], 5, dev)?;
    let narrowed_ids = all_ids.narrow(0, 0..2)?.contiguous()?;
    assert_eq!(narrowed_ids.dims(), &[2]);

    let b = a.index_select(&narrowed_ids, 0)?;
    assert_eq!(b.dims(), &[2, 1]);
    // Should select only rows 0 and 1, not all 5
    assert_eq!(b.to_vec()?, vec![10., 20.]);
    Ok(())
}
test_all_backends!(test_index_select_narrowed_indices, test_index_select_narrowed_indices_impl);

/// An index of -1 should select a row of zeros rather than reading from the source.
/// Generic over the float dtype so the same checks run for f32/f16/bf16 on every backend.
fn test_index_select_minus_one_dtype<T: xn::WithDTypeF, B: Backend>(dev: &B) -> Result<()> {
    let f = T::from_f32;
    // 4x3 tensor of rows 1..=12.
    let data: Vec<T> = (1..=12).map(|x| f(x as f32)).collect();
    let a: Tensor<T, B> = Tensor::from_vec(data, (4, 3), dev)?;

    // dim 0: select rows [2, -1, 0]; the -1 row must be all zeros.
    let idx = Tensor::from_vec(vec![2i64, -1, 0], 3, dev)?;
    let b = a.index_select(&idx, 0)?;
    assert_eq!(b.dims(), &[3, 3]);
    let got: Vec<f32> = b.to_vec()?.into_iter().map(<T as xn::WithDTypeF>::to_f32).collect();
    assert_eq!(got, vec![7., 8., 9., 0., 0., 0., 1., 2., 3.]);

    // dim 1: select columns [-1, 1, -1]; the -1 columns must be zeros.
    let idx = Tensor::from_vec(vec![-1i64, 1, -1], 3, dev)?;
    let c = a.index_select(&idx, 1)?;
    assert_eq!(c.dims(), &[4, 3]);
    let got: Vec<f32> = c.to_vec()?.into_iter().map(<T as xn::WithDTypeF>::to_f32).collect();
    assert_eq!(got, vec![0., 2., 0., 0., 5., 0., 0., 8., 0., 0., 11., 0.]);

    // All indices -1 -> all zeros.
    let idx = Tensor::from_vec(vec![-1i64, -1], 2, dev)?;
    let d = a.index_select(&idx, 0)?;
    assert_eq!(d.dims(), &[2, 3]);
    let got: Vec<f32> = d.to_vec()?.into_iter().map(<T as xn::WithDTypeF>::to_f32).collect();
    assert_eq!(got, vec![0.; 6]);
    Ok(())
}

fn test_index_select_minus_one_impl<B: Backend>(dev: &B) -> Result<()> {
    test_index_select_minus_one_dtype::<f32, B>(dev)?;
    test_index_select_minus_one_dtype::<half::f16, B>(dev)?;
    test_index_select_minus_one_dtype::<half::bf16, B>(dev)?;
    Ok(())
}
test_all_backends!(test_index_select_minus_one, test_index_select_minus_one_impl);

// =============================================================================
// Reduce tests
// =============================================================================

fn test_max_dim0_impl<B: Backend>(dev: &B) -> Result<()> {
    // 3x4 tensor, max along dim 0 -> 4
    let a: Tensor<f32, B> =
        Tensor::from_vec(vec![1., 5., 3., 4., 8., 2., 7., 6., 9., 0., 1., 2.], (3, 4), dev)?;
    // Column-wise max:
    // col 0: max(1, 8, 9) = 9
    // col 1: max(5, 2, 0) = 5
    // col 2: max(3, 7, 1) = 7
    // col 3: max(4, 6, 2) = 6
    let b = a.max(0)?;
    assert_eq!(b.dims(), &[4]);
    assert_eq!(b.to_vec()?, vec![9., 5., 7., 6.]);
    Ok(())
}
test_all_backends!(test_max_dim0, test_max_dim0_impl);

fn test_max_dim1_impl<B: Backend>(dev: &B) -> Result<()> {
    // 3x4 tensor, max along dim 1 -> 3
    let a: Tensor<f32, B> =
        Tensor::from_vec(vec![1., 5., 3., 4., 8., 2., 7., 6., 9., 0., 1., 2.], (3, 4), dev)?;
    // Row-wise max:
    // row 0: max(1, 5, 3, 4) = 5
    // row 1: max(8, 2, 7, 6) = 8
    // row 2: max(9, 0, 1, 2) = 9
    let b = a.max(1)?;
    assert_eq!(b.dims(), &[3]);
    assert_eq!(b.to_vec()?, vec![5., 8., 9.]);
    Ok(())
}
test_all_backends!(test_max_dim1, test_max_dim1_impl);

fn test_min_dim0_impl<B: Backend>(dev: &B) -> Result<()> {
    // 3x4 tensor, min along dim 0 -> 4
    let a: Tensor<f32, B> =
        Tensor::from_vec(vec![1., 5., 3., 4., 8., 2., 7., 6., 9., 0., 1., 2.], (3, 4), dev)?;
    // Column-wise min:
    // col 0: min(1, 8, 9) = 1
    // col 1: min(5, 2, 0) = 0
    // col 2: min(3, 7, 1) = 1
    // col 3: min(4, 6, 2) = 2
    let b = a.min(0)?;
    assert_eq!(b.dims(), &[4]);
    assert_eq!(b.to_vec()?, vec![1., 0., 1., 2.]);
    Ok(())
}
test_all_backends!(test_min_dim0, test_min_dim0_impl);

fn test_min_dim1_impl<B: Backend>(dev: &B) -> Result<()> {
    // 3x4 tensor, min along dim 1 -> 3
    let a: Tensor<f32, B> =
        Tensor::from_vec(vec![1., 5., 3., 4., 8., 2., 7., 6., 9., 0., 1., 2.], (3, 4), dev)?;
    // Row-wise min:
    // row 0: min(1, 5, 3, 4) = 1
    // row 1: min(8, 2, 7, 6) = 2
    // row 2: min(9, 0, 1, 2) = 0
    let b = a.min(1)?;
    assert_eq!(b.dims(), &[3]);
    assert_eq!(b.to_vec()?, vec![1., 2., 0.]);
    Ok(())
}
test_all_backends!(test_min_dim1, test_min_dim1_impl);

fn test_argmin_dim0_impl<B: Backend>(dev: &B) -> Result<()> {
    // 3x4 tensor, argmin along dim 0 -> 4
    let a: Tensor<f32, B> =
        Tensor::from_vec(vec![1., 5., 3., 4., 8., 2., 7., 6., 9., 0., 1., 2.], (3, 4), dev)?;
    // Column-wise argmin:
    // col 0: argmin(1, 8, 9) = 0
    // col 1: argmin(5, 2, 0) = 2
    // col 2: argmin(3, 7, 1) = 2
    // col 3: argmin(4, 6, 2) = 2
    let b = a.argmin(0)?;
    assert_eq!(b.dims(), &[4]);
    assert_eq!(b.to_vec()?, vec![0i64, 2, 2, 2]);
    Ok(())
}
test_all_backends!(test_argmin_dim0, test_argmin_dim0_impl);

fn test_argmin_dim1_impl<B: Backend>(dev: &B) -> Result<()> {
    // 3x4 tensor, argmin along dim 1 -> 3
    let a: Tensor<f32, B> =
        Tensor::from_vec(vec![1., 5., 3., 4., 8., 2., 7., 6., 9., 0., 1., 2.], (3, 4), dev)?;
    // Row-wise argmin:
    // row 0: argmin(1, 5, 3, 4) = 0
    // row 1: argmin(8, 2, 7, 6) = 1
    // row 2: argmin(9, 0, 1, 2) = 1
    let b = a.argmin(1)?;
    assert_eq!(b.dims(), &[3]);
    assert_eq!(b.to_vec()?, vec![0i64, 1, 1]);
    Ok(())
}
test_all_backends!(test_argmin_dim1, test_argmin_dim1_impl);

fn test_argmax_dim0_impl<B: Backend>(dev: &B) -> Result<()> {
    // 3x4 tensor, argmax along dim 0 -> 4
    let a: Tensor<f32, B> =
        Tensor::from_vec(vec![1., 5., 3., 4., 8., 2., 7., 6., 9., 0., 1., 2.], (3, 4), dev)?;
    // Column-wise argmax:
    // col 0: argmax(1, 8, 9) = 2
    // col 1: argmax(5, 2, 0) = 0
    // col 2: argmax(3, 7, 1) = 1
    // col 3: argmax(4, 6, 2) = 1
    let b = a.argmax(0)?;
    assert_eq!(b.dims(), &[4]);
    assert_eq!(b.to_vec()?, vec![2i64, 0, 1, 1]);
    Ok(())
}
test_all_backends!(test_argmax_dim0, test_argmax_dim0_impl);

fn test_argmax_dim1_impl<B: Backend>(dev: &B) -> Result<()> {
    // 3x4 tensor, argmax along dim 1 -> 3
    let a: Tensor<f32, B> =
        Tensor::from_vec(vec![1., 5., 3., 4., 8., 2., 7., 6., 9., 0., 1., 2.], (3, 4), dev)?;
    // Row-wise argmax:
    // row 0: argmax(1, 5, 3, 4) = 1
    // row 1: argmax(8, 2, 7, 6) = 0
    // row 2: argmax(9, 0, 1, 2) = 0
    let b = a.argmax(1)?;
    assert_eq!(b.dims(), &[3]);
    assert_eq!(b.to_vec()?, vec![1i64, 0, 0]);
    Ok(())
}
test_all_backends!(test_argmax_dim1, test_argmax_dim1_impl);

fn test_max_3d_impl<B: Backend>(dev: &B) -> Result<()> {
    // 2x3x2 tensor, max along dim 1 -> 2x2
    let a: Tensor<f32, B> = Tensor::from_vec((1..=12).map(|x| x as f32).collect(), (2, 3, 2), dev)?;
    // Batch 0: [[1,2], [3,4], [5,6]] -> max along rows: [5, 6]
    // Batch 1: [[7,8], [9,10], [11,12]] -> max along rows: [11, 12]
    let b = a.max(1)?;
    assert_eq!(b.dims(), &[2, 2]);
    assert_eq!(b.to_vec()?, vec![5., 6., 11., 12.]);
    Ok(())
}
test_all_backends!(test_max_3d, test_max_3d_impl);

// =============================================================================
// Broadcast tests
// =============================================================================

fn test_broadcast_add_same_shape_impl<B: Backend>(dev: &B) -> Result<()> {
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4.], (2, 2), dev)?;
    let b: Tensor<f32, B> = Tensor::from_vec(vec![10., 20., 30., 40.], (2, 2), dev)?;
    let c = a.broadcast_add(&b)?;
    assert_eq!(c.dims(), &[2, 2]);
    assert_eq!(c.to_vec()?, vec![11., 22., 33., 44.]);
    Ok(())
}
test_all_backends!(test_broadcast_add_same_shape, test_broadcast_add_same_shape_impl);

fn test_broadcast_add_1d_to_2d_impl<B: Backend>(dev: &B) -> Result<()> {
    // [2, 3] + [3] -> [2, 3]
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;
    let b: Tensor<f32, B> = Tensor::from_vec(vec![10., 20., 30.], (3,), dev)?;
    let c = a.broadcast_add(&b)?;
    assert_eq!(c.dims(), &[2, 3]);
    // Row 0: [1+10, 2+20, 3+30] = [11, 22, 33]
    // Row 1: [4+10, 5+20, 6+30] = [14, 25, 36]
    assert_eq!(c.to_vec()?, vec![11., 22., 33., 14., 25., 36.]);
    Ok(())
}
test_all_backends!(test_broadcast_add_1d_to_2d, test_broadcast_add_1d_to_2d_impl);

fn test_broadcast_mul_column_impl<B: Backend>(dev: &B) -> Result<()> {
    // [2, 3] * [2, 1] -> [2, 3]
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;
    let b: Tensor<f32, B> = Tensor::from_vec(vec![2., 3.], (2, 1), dev)?;
    let c = a.broadcast_mul(&b)?;
    assert_eq!(c.dims(), &[2, 3]);
    // Row 0: [1*2, 2*2, 3*2] = [2, 4, 6]
    // Row 1: [4*3, 5*3, 6*3] = [12, 15, 18]
    assert_eq!(c.to_vec()?, vec![2., 4., 6., 12., 15., 18.]);
    Ok(())
}
test_all_backends!(test_broadcast_mul_column, test_broadcast_mul_column_impl);

fn test_broadcast_sub_impl<B: Backend>(dev: &B) -> Result<()> {
    // [2, 3] - [3] -> [2, 3]
    let a: Tensor<f32, B> = Tensor::from_vec(vec![10., 20., 30., 40., 50., 60.], (2, 3), dev)?;
    let b: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3.], (3,), dev)?;
    let c = a.broadcast_sub(&b)?;
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(c.to_vec()?, vec![9., 18., 27., 39., 48., 57.]);
    Ok(())
}
test_all_backends!(test_broadcast_sub, test_broadcast_sub_impl);

fn test_broadcast_div_impl<B: Backend>(dev: &B) -> Result<()> {
    // [2, 3] / [2, 1] -> [2, 3]
    let a: Tensor<f32, B> = Tensor::from_vec(vec![2., 4., 6., 9., 12., 15.], (2, 3), dev)?;
    let b: Tensor<f32, B> = Tensor::from_vec(vec![2., 3.], (2, 1), dev)?;
    let c = a.broadcast_div(&b)?;
    assert_eq!(c.dims(), &[2, 3]);
    // Row 0: [2/2, 4/2, 6/2] = [1, 2, 3]
    // Row 1: [9/3, 12/3, 15/3] = [3, 4, 5]
    // Division is compared approximately: unlike add/sub/mul, GPU float
    // division is not required to be correctly rounded (Vulkan permits
    // 2.5 ULP; RADV lowers it to rcp + multiply).
    let expected = [1., 2., 3., 3., 4., 5.];
    for (i, (got, want)) in c.to_vec()?.iter().zip(expected.iter()).enumerate() {
        assert!((got - want).abs() <= want.abs() * 1e-6, "index {i}: got {got}, want {want}");
    }
    Ok(())
}
test_all_backends!(test_broadcast_div, test_broadcast_div_impl);

fn test_broadcast_3d_impl<B: Backend>(dev: &B) -> Result<()> {
    // [2, 3, 4] + [4] -> [2, 3, 4]
    let a: Tensor<f32, B> = Tensor::from_vec((1..=24).map(|x| x as f32).collect(), (2, 3, 4), dev)?;
    let b: Tensor<f32, B> = Tensor::from_vec(vec![100., 200., 300., 400.], (4,), dev)?;
    let c = a.broadcast_add(&b)?;
    assert_eq!(c.dims(), &[2, 3, 4]);
    let c_vec = c.to_vec()?;
    // First element: 1 + 100 = 101
    assert_eq!(c_vec[0], 101.);
    // Second element: 2 + 200 = 202
    assert_eq!(c_vec[1], 202.);
    // Fifth element: 5 + 100 = 105
    assert_eq!(c_vec[4], 105.);
    Ok(())
}
test_all_backends!(test_broadcast_3d, test_broadcast_3d_impl);

// =============================================================================
// Unsqueeze tests
// =============================================================================

fn test_unsqueeze_dim0_impl<B: Backend>(dev: &B) -> Result<()> {
    // [3, 4] -> unsqueeze(0) -> [1, 3, 4]
    let a: Tensor<f32, B> = Tensor::from_vec((1..=12).map(|x| x as f32).collect(), (3, 4), dev)?;
    let b = a.unsqueeze(0)?;
    assert_eq!(b.dims(), &[1, 3, 4]);
    assert_eq!(b.to_vec()?, a.to_vec()?);
    Ok(())
}
test_all_backends!(test_unsqueeze_dim0, test_unsqueeze_dim0_impl);

fn test_unsqueeze_dim1_impl<B: Backend>(dev: &B) -> Result<()> {
    // [3, 4] -> unsqueeze(1) -> [3, 1, 4]
    let a: Tensor<f32, B> = Tensor::from_vec((1..=12).map(|x| x as f32).collect(), (3, 4), dev)?;
    let b = a.unsqueeze(1)?;
    assert_eq!(b.dims(), &[3, 1, 4]);
    assert_eq!(b.to_vec()?, a.to_vec()?);
    Ok(())
}
test_all_backends!(test_unsqueeze_dim1, test_unsqueeze_dim1_impl);

fn test_unsqueeze_dim_last_impl<B: Backend>(dev: &B) -> Result<()> {
    // [3, 4] -> unsqueeze(2) -> [3, 4, 1]
    let a: Tensor<f32, B> = Tensor::from_vec((1..=12).map(|x| x as f32).collect(), (3, 4), dev)?;
    let b = a.unsqueeze(2)?;
    assert_eq!(b.dims(), &[3, 4, 1]);
    assert_eq!(b.to_vec()?, a.to_vec()?);
    Ok(())
}
test_all_backends!(test_unsqueeze_dim_last, test_unsqueeze_dim_last_impl);

fn test_unsqueeze_1d_impl<B: Backend>(dev: &B) -> Result<()> {
    // [4] -> unsqueeze(0) -> [1, 4]
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4.], (4,), dev)?;
    let b = a.unsqueeze(0)?;
    assert_eq!(b.dims(), &[1, 4]);

    // [4] -> unsqueeze(1) -> [4, 1]
    let c = a.unsqueeze(1)?;
    assert_eq!(c.dims(), &[4, 1]);
    Ok(())
}
test_all_backends!(test_unsqueeze_1d, test_unsqueeze_1d_impl);

// =============================================================================
// Pad with zeros tests
// =============================================================================

fn test_pad_with_zeros_1d_impl<B: Backend>(dev: &B) -> Result<()> {
    // [4] -> pad_with_zeros(0, 2, 3) -> [9]
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4.], (4,), dev)?;
    let b = a.pad_with_zeros(0, 2, 3)?;
    assert_eq!(b.dims(), &[9]);
    assert_eq!(b.to_vec()?, vec![0., 0., 1., 2., 3., 4., 0., 0., 0.]);
    Ok(())
}
test_all_backends!(test_pad_with_zeros_1d, test_pad_with_zeros_1d_impl);

fn test_pad_with_zeros_2d_dim0_impl<B: Backend>(dev: &B) -> Result<()> {
    // [2, 3] -> pad_with_zeros(0, 1, 1) -> [4, 3]
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;
    let b = a.pad_with_zeros(0, 1, 1)?;
    assert_eq!(b.dims(), &[4, 3]);
    // Row 0: zeros, Row 1: [1,2,3], Row 2: [4,5,6], Row 3: zeros
    assert_eq!(b.to_vec()?, vec![0., 0., 0., 1., 2., 3., 4., 5., 6., 0., 0., 0.]);
    Ok(())
}
test_all_backends!(test_pad_with_zeros_2d_dim0, test_pad_with_zeros_2d_dim0_impl);

fn test_pad_with_zeros_2d_dim1_impl<B: Backend>(dev: &B) -> Result<()> {
    // [2, 3] -> pad_with_zeros(1, 1, 2) -> [2, 6]
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;
    let b = a.pad_with_zeros(1, 1, 2)?;
    assert_eq!(b.dims(), &[2, 6]);
    // Row 0: [0, 1, 2, 3, 0, 0]
    // Row 1: [0, 4, 5, 6, 0, 0]
    assert_eq!(b.to_vec()?, vec![0., 1., 2., 3., 0., 0., 0., 4., 5., 6., 0., 0.]);
    Ok(())
}
test_all_backends!(test_pad_with_zeros_2d_dim1, test_pad_with_zeros_2d_dim1_impl);

fn test_pad_with_zeros_3d_impl<B: Backend>(dev: &B) -> Result<()> {
    // [2, 2, 3] -> pad_with_zeros(1, 1, 0) -> [2, 3, 3]
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let a: Tensor<f32, B> = Tensor::from_vec(data, (2, 2, 3), dev)?;
    let b = a.pad_with_zeros(1, 1, 0)?;
    assert_eq!(b.dims(), &[2, 3, 3]);
    // First batch: [[0,0,0], [1,2,3], [4,5,6]]
    // Second batch: [[0,0,0], [7,8,9], [10,11,12]]
    assert_eq!(
        b.to_vec()?,
        vec![0., 0., 0., 1., 2., 3., 4., 5., 6., 0., 0., 0., 7., 8., 9., 10., 11., 12.]
    );
    Ok(())
}
test_all_backends!(test_pad_with_zeros_3d, test_pad_with_zeros_3d_impl);

// =============================================================================
// Conv1d tests
// =============================================================================

fn test_conv1d_simple_impl<B: Backend>(dev: &B) -> Result<()> {
    // Input: (batch=1, in_channels=1, length=5)
    // Kernel: (out_channels=1, in_channels=1, kernel_size=3)
    // No padding, stride=1, groups=1
    let input: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5.], (1, 1, 5), dev)?;
    let kernel: Tensor<f32, B> = Tensor::from_vec(vec![1., 0., -1.], (1, 1, 3), dev)?;

    let output = input.conv1d(&kernel, None, 1, 0, 1, 1)?;
    assert_eq!(output.dims(), &[1, 1, 3]);
    // output[i] = input[i]*1 + input[i+1]*0 + input[i+2]*(-1)
    // output[0] = 1 - 3 = -2
    // output[1] = 2 - 4 = -2
    // output[2] = 3 - 5 = -2
    assert_eq!(output.to_vec()?, vec![-2., -2., -2.]);
    Ok(())
}
test_all_backends!(test_conv1d_simple, test_conv1d_simple_impl);

fn test_conv1d_with_padding_impl<B: Backend>(dev: &B) -> Result<()> {
    // Input: (batch=1, in_channels=1, length=4)
    // Kernel: (out_channels=1, in_channels=1, kernel_size=3)
    // Padding=1, stride=1
    let input: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4.], (1, 1, 4), dev)?;
    let kernel: Tensor<f32, B> = Tensor::from_vec(vec![1., 1., 1.], (1, 1, 3), dev)?;

    let output = input.conv1d(&kernel, None, 1, 1, 1, 1)?;
    assert_eq!(output.dims(), &[1, 1, 4]);
    // With padding=1, we have [0, 1, 2, 3, 4, 0] as effective input
    // output[0] = 0 + 1 + 2 = 3
    // output[1] = 1 + 2 + 3 = 6
    // output[2] = 2 + 3 + 4 = 9
    // output[3] = 3 + 4 + 0 = 7
    assert_eq!(output.to_vec()?, vec![3., 6., 9., 7.]);
    Ok(())
}
test_all_backends!(test_conv1d_with_padding, test_conv1d_with_padding_impl);

fn test_conv1d_with_stride_impl<B: Backend>(dev: &B) -> Result<()> {
    // Input: (batch=1, in_channels=1, length=6)
    // Kernel: (out_channels=1, in_channels=1, kernel_size=2)
    // Stride=2
    let input: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (1, 1, 6), dev)?;
    let kernel: Tensor<f32, B> = Tensor::from_vec(vec![1., 1.], (1, 1, 2), dev)?;

    let output = input.conv1d(&kernel, None, 2, 0, 1, 1)?;
    // out_length = (6 - 2) / 2 + 1 = 3
    assert_eq!(output.dims(), &[1, 1, 3]);
    // output[0] = 1 + 2 = 3
    // output[1] = 3 + 4 = 7
    // output[2] = 5 + 6 = 11
    assert_eq!(output.to_vec()?, vec![3., 7., 11.]);
    Ok(())
}
test_all_backends!(test_conv1d_with_stride, test_conv1d_with_stride_impl);

fn test_conv1d_with_bias_impl<B: Backend>(dev: &B) -> Result<()> {
    let input: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5.], (1, 1, 5), dev)?;
    let kernel: Tensor<f32, B> = Tensor::from_vec(vec![1., 1., 1.], (1, 1, 3), dev)?;
    let bias: Tensor<f32, B> = Tensor::from_vec(vec![10.], (1,), dev)?;

    let output = input.conv1d(&kernel, Some(&bias), 1, 0, 1, 1)?;
    assert_eq!(output.dims(), &[1, 1, 3]);
    // Without bias: [6, 9, 12], with bias: [16, 19, 22]
    assert_eq!(output.to_vec()?, vec![16., 19., 22.]);
    Ok(())
}
test_all_backends!(test_conv1d_with_bias, test_conv1d_with_bias_impl);

fn test_conv1d_multi_channel_impl<B: Backend>(dev: &B) -> Result<()> {
    // Input: (batch=1, in_channels=2, length=3)
    // Kernel: (out_channels=2, in_channels=2, kernel_size=2)
    let input: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (1, 2, 3), dev)?;
    // kernel[0] for out_channel 0: [[1,1], [0,0]] - only uses in_channel 0
    // kernel[1] for out_channel 1: [[0,0], [1,1]] - only uses in_channel 1
    let kernel: Tensor<f32, B> =
        Tensor::from_vec(vec![1., 1., 0., 0., 0., 0., 1., 1.], (2, 2, 2), dev)?;

    let output = input.conv1d(&kernel, None, 1, 0, 1, 1)?;
    assert_eq!(output.dims(), &[1, 2, 2]);
    // out[0,0] = 1+2 = 3, out[0,1] = 2+3 = 5
    // out[1,0] = 4+5 = 9, out[1,1] = 5+6 = 11
    assert_eq!(output.to_vec()?, vec![3., 5., 9., 11.]);
    Ok(())
}
test_all_backends!(test_conv1d_multi_channel, test_conv1d_multi_channel_impl);

fn test_conv1d_batch_simple_impl<B: Backend>(dev: &B) -> Result<()> {
    // batch=2, in_channels=1, length=4, kernel_size=2
    // Use distinct values per batch to catch cross-batch contamination.
    let input: Tensor<f32, B> =
        Tensor::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8.], (2, 1, 4), dev)?;
    let kernel: Tensor<f32, B> = Tensor::from_vec(vec![1., 1.], (1, 1, 2), dev)?;

    let output = input.conv1d(&kernel, None, 1, 0, 1, 1)?;
    assert_eq!(output.dims(), &[2, 1, 3]);
    // Batch 0: [1+2, 2+3, 3+4] = [3, 5, 7]
    // Batch 1: [5+6, 6+7, 7+8] = [11, 13, 15]
    assert_eq!(output.to_vec()?, vec![3., 5., 7., 11., 13., 15.]);
    Ok(())
}
test_all_backends!(test_conv1d_batch_simple, test_conv1d_batch_simple_impl);

fn test_conv1d_batch_multi_channel_impl<B: Backend>(dev: &B) -> Result<()> {
    // batch=2, in_channels=2, out_channels=2, length=3, kernel_size=2
    // Input shape: [2, 2, 3]
    #[rustfmt::skip]
    let input: Tensor<f32, B> = Tensor::from_vec(vec![
        // Batch 0, channel 0     Batch 0, channel 1
        1., 2., 3.,               4., 5., 6.,
        // Batch 1, channel 0     Batch 1, channel 1
        7., 8., 9.,               10., 11., 12.,
    ], (2, 2, 3), dev)?;
    // Kernel shape: [2, 2, 2] (out_channels=2, in_channels=2, kernel_size=2)
    // out_channel 0: uses in_ch0 with [1,0] and in_ch1 with [0,1]
    // out_channel 1: uses in_ch0 with [0,1] and in_ch1 with [1,0]
    #[rustfmt::skip]
    let kernel: Tensor<f32, B> = Tensor::from_vec(vec![
        1., 0.,  0., 1.,   // out_ch 0
        0., 1.,  1., 0.,   // out_ch 1
    ], (2, 2, 2), dev)?;

    let output = input.conv1d(&kernel, None, 1, 0, 1, 1)?;
    assert_eq!(output.dims(), &[2, 2, 2]);
    // Batch 0, out_ch 0: in_ch0*[1,0] + in_ch1*[0,1] at each position
    //   pos 0: 1*1+2*0 + 4*0+5*1 = 1+5 = 6
    //   pos 1: 2*1+3*0 + 5*0+6*1 = 2+6 = 8
    // Batch 0, out_ch 1: in_ch0*[0,1] + in_ch1*[1,0]
    //   pos 0: 1*0+2*1 + 4*1+5*0 = 2+4 = 6
    //   pos 1: 2*0+3*1 + 5*1+6*0 = 3+5 = 8
    // Batch 1, out_ch 0:
    //   pos 0: 7*1+8*0 + 10*0+11*1 = 7+11 = 18
    //   pos 1: 8*1+9*0 + 11*0+12*1 = 8+12 = 20
    // Batch 1, out_ch 1:
    //   pos 0: 7*0+8*1 + 10*1+11*0 = 8+10 = 18
    //   pos 1: 8*0+9*1 + 11*1+12*0 = 9+11 = 20
    assert_eq!(output.to_vec()?, vec![6., 8., 6., 8., 18., 20., 18., 20.]);
    Ok(())
}
test_all_backends!(test_conv1d_batch_multi_channel, test_conv1d_batch_multi_channel_impl);

fn test_conv1d_batch_with_padding_impl<B: Backend>(dev: &B) -> Result<()> {
    // batch=3, in_channels=1, length=3, kernel_size=3, padding=1
    #[rustfmt::skip]
    let input: Tensor<f32, B> = Tensor::from_vec(vec![
        1., 2., 3.,     // batch 0
        4., 5., 6.,     // batch 1
        7., 8., 9.,     // batch 2
    ], (3, 1, 3), dev)?;
    let kernel: Tensor<f32, B> = Tensor::from_vec(vec![1., 1., 1.], (1, 1, 3), dev)?;

    let output = input.conv1d(&kernel, None, 1, 1, 1, 1)?;
    assert_eq!(output.dims(), &[3, 1, 3]);
    // With padding=1, effective input has 0s on each side
    // Batch 0: [0,1,2,3,0] -> [0+1+2, 1+2+3, 2+3+0] = [3, 6, 5]
    // Batch 1: [0,4,5,6,0] -> [0+4+5, 4+5+6, 5+6+0] = [9, 15, 11]
    // Batch 2: [0,7,8,9,0] -> [0+7+8, 7+8+9, 8+9+0] = [15, 24, 17]
    assert_eq!(output.to_vec()?, vec![3., 6., 5., 9., 15., 11., 15., 24., 17.]);
    Ok(())
}
test_all_backends!(test_conv1d_batch_with_padding, test_conv1d_batch_with_padding_impl);

fn test_conv1d_batch_with_stride_impl<B: Backend>(dev: &B) -> Result<()> {
    // batch=2, in_channels=1, length=6, kernel_size=2, stride=2
    #[rustfmt::skip]
    let input: Tensor<f32, B> = Tensor::from_vec(vec![
        1., 2., 3., 4., 5., 6.,     // batch 0
        7., 8., 9., 10., 11., 12.,  // batch 1
    ], (2, 1, 6), dev)?;
    let kernel: Tensor<f32, B> = Tensor::from_vec(vec![1., -1.], (1, 1, 2), dev)?;

    let output = input.conv1d(&kernel, None, 2, 0, 1, 1)?;
    // out_length = (6 - 2) / 2 + 1 = 3
    assert_eq!(output.dims(), &[2, 1, 3]);
    // Batch 0: [1-2, 3-4, 5-6] = [-1, -1, -1]
    // Batch 1: [7-8, 9-10, 11-12] = [-1, -1, -1]
    assert_eq!(output.to_vec()?, vec![-1., -1., -1., -1., -1., -1.]);
    Ok(())
}
test_all_backends!(test_conv1d_batch_with_stride, test_conv1d_batch_with_stride_impl);

fn test_conv1d_batch_with_bias_impl<B: Backend>(dev: &B) -> Result<()> {
    // batch=2, in_channels=1, out_channels=2, length=3, kernel_size=2
    #[rustfmt::skip]
    let input: Tensor<f32, B> = Tensor::from_vec(vec![
        1., 2., 3.,   // batch 0
        4., 5., 6.,   // batch 1
    ], (2, 1, 3), dev)?;
    // Two output channels with different kernels
    let kernel: Tensor<f32, B> = Tensor::from_vec(vec![1., 1., 1., -1.], (2, 1, 2), dev)?;
    let bias: Tensor<f32, B> = Tensor::from_vec(vec![10., 100.], (2,), dev)?;

    let output = input.conv1d(&kernel, Some(&bias), 1, 0, 1, 1)?;
    assert_eq!(output.dims(), &[2, 2, 2]);
    // Batch 0, out_ch 0 (kernel [1,1], bias 10): [1+2+10, 2+3+10] = [13, 15]
    // Batch 0, out_ch 1 (kernel [1,-1], bias 100): [1-2+100, 2-3+100] = [99, 99]
    // Batch 1, out_ch 0: [4+5+10, 5+6+10] = [19, 21]
    // Batch 1, out_ch 1: [4-5+100, 5-6+100] = [99, 99]
    assert_eq!(output.to_vec()?, vec![13., 15., 99., 99., 19., 21., 99., 99.]);
    Ok(())
}
test_all_backends!(test_conv1d_batch_with_bias, test_conv1d_batch_with_bias_impl);

fn test_conv1d_batch_large_impl<B: Backend>(dev: &B) -> Result<()> {
    // batch=4, in_channels=3, out_channels=2, length=8, kernel_size=3
    // Use a simple identity-like pattern: each output channel sums one input channel.
    let batch = 4;
    let in_ch = 3;
    let out_ch = 2;
    let length = 8;
    let k_size = 3;

    // Input: sequential values so each batch element is distinguishable
    let input_data: Vec<f32> = (0..batch * in_ch * length).map(|i| i as f32).collect();
    let input: Tensor<f32, B> = Tensor::from_vec(input_data.clone(), (batch, in_ch, length), dev)?;

    // Kernel [out_ch=2, in_ch=3, k_size=3]:
    // out_ch 0: sum across all in_channels with kernel [1, 0, 0]
    // out_ch 1: sum across all in_channels with kernel [0, 0, 1]
    let mut kernel_data = vec![0.0f32; out_ch * in_ch * k_size];
    for c in 0..in_ch {
        kernel_data[c * k_size] = 1.0; // out_ch 0, [1,0,0]
        kernel_data[in_ch * k_size + c * k_size + 2] = 1.0; // out_ch 1, [0,0,1]
    }
    let kernel: Tensor<f32, B> = Tensor::from_vec(kernel_data, (out_ch, in_ch, k_size), dev)?;

    let output = input.conv1d(&kernel, None, 1, 0, 1, 1)?;
    let out_length = length - k_size + 1; // 6
    assert_eq!(output.dims(), &[batch, out_ch, out_length]);

    let result = output.to_vec()?;
    // Verify each batch element independently
    for b in 0..batch {
        for pos in 0..out_length {
            // out_ch 0 uses kernel [1,0,0] per in_channel -> picks input[ch][pos]
            let mut expected_ch0 = 0.0f32;
            let mut expected_ch1 = 0.0f32;
            for c in 0..in_ch {
                let base = b * in_ch * length + c * length;
                expected_ch0 += input_data[base + pos]; // kernel [1,0,0] picks first
                expected_ch1 += input_data[base + pos + 2]; // kernel [0,0,1] picks last
            }
            let idx_ch0 = b * out_ch * out_length + pos;
            let idx_ch1 = b * out_ch * out_length + out_length + pos;
            assert!(
                (result[idx_ch0] - expected_ch0).abs() < 1e-4,
                "batch {b} out_ch 0 pos {pos}: expected {expected_ch0}, got {}",
                result[idx_ch0]
            );
            assert!(
                (result[idx_ch1] - expected_ch1).abs() < 1e-4,
                "batch {b} out_ch 1 pos {pos}: expected {expected_ch1}, got {}",
                result[idx_ch1]
            );
        }
    }
    Ok(())
}
test_all_backends!(test_conv1d_batch_large, test_conv1d_batch_large_impl);

// =============================================================================
// Conv transpose 1d tests
// =============================================================================

fn test_conv_transpose1d_simple_impl<B: Backend>(dev: &B) -> Result<()> {
    // Input: (batch=1, in_channels=1, length=3)
    // Kernel: (in_channels=1, out_channels=1, kernel_size=3)
    // stride=1, no padding
    let input: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3.], (1, 1, 3), dev)?;
    let kernel: Tensor<f32, B> = Tensor::from_vec(vec![1., 1., 1.], (1, 1, 3), dev)?;

    let output = input.conv_transpose1d(&kernel, None, 1, 0, 0, 1)?;
    // out_length = (3-1)*1 + 3 + 0 - 0 = 5
    assert_eq!(output.dims(), &[1, 1, 5]);
    // Each input value contributes to 3 consecutive output positions
    // output[0] = 1*1 = 1
    // output[1] = 1*1 + 2*1 = 3
    // output[2] = 1*1 + 2*1 + 3*1 = 6
    // output[3] = 2*1 + 3*1 = 5
    // output[4] = 3*1 = 3
    assert_eq!(output.to_vec()?, vec![1., 3., 6., 5., 3.]);
    Ok(())
}
test_all_backends!(test_conv_transpose1d_simple, test_conv_transpose1d_simple_impl);

fn test_conv_transpose1d_with_stride_impl<B: Backend>(dev: &B) -> Result<()> {
    // Input: (batch=1, in_channels=1, length=3)
    // Kernel: (in_channels=1, out_channels=1, kernel_size=2)
    // stride=2
    let input: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3.], (1, 1, 3), dev)?;
    let kernel: Tensor<f32, B> = Tensor::from_vec(vec![1., 1.], (1, 1, 2), dev)?;

    let output = input.conv_transpose1d(&kernel, None, 2, 0, 0, 1)?;
    // out_length = (3-1)*2 + 2 + 0 - 0 = 6
    assert_eq!(output.dims(), &[1, 1, 6]);
    // Input at position i contributes to output positions i*stride + k
    // input[0]=1 -> output[0], output[1]
    // input[1]=2 -> output[2], output[3]
    // input[2]=3 -> output[4], output[5]
    assert_eq!(output.to_vec()?, vec![1., 1., 2., 2., 3., 3.]);
    Ok(())
}
test_all_backends!(test_conv_transpose1d_with_stride, test_conv_transpose1d_with_stride_impl);

fn test_conv_transpose1d_with_bias_impl<B: Backend>(dev: &B) -> Result<()> {
    let input: Tensor<f32, B> = Tensor::from_vec(vec![1., 2.], (1, 1, 2), dev)?;
    let kernel: Tensor<f32, B> = Tensor::from_vec(vec![1., 1.], (1, 1, 2), dev)?;
    let bias: Tensor<f32, B> = Tensor::from_vec(vec![5.], (1,), dev)?;

    let output = input.conv_transpose1d(&kernel, Some(&bias), 1, 0, 0, 1)?;
    // out_length = (2-1)*1 + 2 = 3
    assert_eq!(output.dims(), &[1, 1, 3]);
    // Without bias: [1, 3, 2], with bias: [6, 8, 7]
    assert_eq!(output.to_vec()?, vec![6., 8., 7.]);
    Ok(())
}
test_all_backends!(test_conv_transpose1d_with_bias, test_conv_transpose1d_with_bias_impl);

fn test_conv_transpose1d_batch_simple_impl<B: Backend>(dev: &B) -> Result<()> {
    // batch=2, in_channels=1, out_channels=1, length=3, kernel_size=3
    #[rustfmt::skip]
    let input: Tensor<f32, B> = Tensor::from_vec(vec![
        1., 2., 3.,   // batch 0
        4., 5., 6.,   // batch 1
    ], (2, 1, 3), dev)?;
    let kernel: Tensor<f32, B> = Tensor::from_vec(vec![1., 1., 1.], (1, 1, 3), dev)?;

    let output = input.conv_transpose1d(&kernel, None, 1, 0, 0, 1)?;
    // out_length = (3-1)*1 + 3 = 5
    assert_eq!(output.dims(), &[2, 1, 5]);
    // Batch 0: [1, 1+2, 1+2+3, 2+3, 3] = [1, 3, 6, 5, 3]
    // Batch 1: [4, 4+5, 4+5+6, 5+6, 6] = [4, 9, 15, 11, 6]
    assert_eq!(output.to_vec()?, vec![1., 3., 6., 5., 3., 4., 9., 15., 11., 6.]);
    Ok(())
}
test_all_backends!(test_conv_transpose1d_batch_simple, test_conv_transpose1d_batch_simple_impl);

fn test_conv_transpose1d_batch_multi_channel_impl<B: Backend>(dev: &B) -> Result<()> {
    // batch=2, in_channels=2, out_channels=1, length=2, kernel_size=2
    #[rustfmt::skip]
    let input: Tensor<f32, B> = Tensor::from_vec(vec![
        // Batch 0: in_ch0, in_ch1
        1., 2.,   3., 4.,
        // Batch 1: in_ch0, in_ch1
        5., 6.,   7., 8.,
    ], (2, 2, 2), dev)?;
    // Kernel: [in_ch=2, out_ch=1, k=2]
    let kernel: Tensor<f32, B> = Tensor::from_vec(vec![1., 1., 1., 1.], (2, 1, 2), dev)?;

    let output = input.conv_transpose1d(&kernel, None, 1, 0, 0, 1)?;
    // out_length = (2-1)*1 + 2 = 3
    assert_eq!(output.dims(), &[2, 1, 3]);
    // Batch 0: ch0 contrib [1, 1+2, 2]=[1,3,2], ch1 contrib [3, 3+4, 4]=[3,7,4]
    //   total: [4, 10, 6]
    // Batch 1: ch0 contrib [5, 5+6, 6]=[5,11,6], ch1 contrib [7, 7+8, 8]=[7,15,8]
    //   total: [12, 26, 14]
    assert_eq!(output.to_vec()?, vec![4., 10., 6., 12., 26., 14.]);
    Ok(())
}
test_all_backends!(
    test_conv_transpose1d_batch_multi_channel,
    test_conv_transpose1d_batch_multi_channel_impl
);

fn test_conv_transpose1d_batch_with_stride_impl<B: Backend>(dev: &B) -> Result<()> {
    // batch=3, in_channels=1, out_channels=1, length=2, kernel_size=2, stride=2
    #[rustfmt::skip]
    let input: Tensor<f32, B> = Tensor::from_vec(vec![
        1., 2.,   // batch 0
        3., 4.,   // batch 1
        5., 6.,   // batch 2
    ], (3, 1, 2), dev)?;
    let kernel: Tensor<f32, B> = Tensor::from_vec(vec![1., 1.], (1, 1, 2), dev)?;

    let output = input.conv_transpose1d(&kernel, None, 2, 0, 0, 1)?;
    // out_length = (2-1)*2 + 2 = 4
    assert_eq!(output.dims(), &[3, 1, 4]);
    // Each input[i] contributes to output[i*stride], output[i*stride+1]
    // Batch 0: [1, 1, 2, 2]
    // Batch 1: [3, 3, 4, 4]
    // Batch 2: [5, 5, 6, 6]
    assert_eq!(output.to_vec()?, vec![1., 1., 2., 2., 3., 3., 4., 4., 5., 5., 6., 6.]);
    Ok(())
}
test_all_backends!(
    test_conv_transpose1d_batch_with_stride,
    test_conv_transpose1d_batch_with_stride_impl
);

fn test_conv_transpose1d_batch_with_bias_impl<B: Backend>(dev: &B) -> Result<()> {
    // batch=2, in_channels=1, out_channels=2, length=2, kernel_size=2
    #[rustfmt::skip]
    let input: Tensor<f32, B> = Tensor::from_vec(vec![
        1., 2.,   // batch 0
        3., 4.,   // batch 1
    ], (2, 1, 2), dev)?;
    // Kernel: [in_ch=1, out_ch=2, k=2]
    let kernel: Tensor<f32, B> = Tensor::from_vec(vec![1., 1., 2., 2.], (1, 2, 2), dev)?;
    let bias: Tensor<f32, B> = Tensor::from_vec(vec![10., 100.], (2,), dev)?;

    let output = input.conv_transpose1d(&kernel, Some(&bias), 1, 0, 0, 1)?;
    // out_length = (2-1)*1 + 2 = 3
    assert_eq!(output.dims(), &[2, 2, 3]);
    // Batch 0, out_ch 0 (kernel [1,1], bias 10): [1, 1+2, 2] + 10 = [11, 13, 12]
    // Batch 0, out_ch 1 (kernel [2,2], bias 100): [2, 2+4, 4] + 100 = [102, 106, 104]
    // Batch 1, out_ch 0: [3, 3+4, 4] + 10 = [13, 17, 14]
    // Batch 1, out_ch 1: [6, 6+8, 8] + 100 = [106, 114, 108]
    assert_eq!(
        output.to_vec()?,
        vec![11., 13., 12., 102., 106., 104., 13., 17., 14., 106., 114., 108.]
    );
    Ok(())
}
test_all_backends!(
    test_conv_transpose1d_batch_with_bias,
    test_conv_transpose1d_batch_with_bias_impl
);

// -----------------------------------------------------------------------------
// conv_transpose1d fallback path (cpu)
//
// The col2im fast path only applies when groups == 1 && padding == 0 &&
// output_padding == 0; anything else goes through conv_transpose1d_direct. The tests above
// all satisfy the fast-path condition, so these cover the fallback by equivalence against
// the fast path.
// -----------------------------------------------------------------------------

/// Grouped conv_transpose1d equals per-group conv_transpose1d (groups=1) concatenated along
/// the channel axis. The reference side takes the well-covered col2im path.
fn check_grouped_matches_per_group(
    batch: usize,
    in_channels: usize,
    out_c_per_group: usize,
    length: usize,
    kernel_size: usize,
    stride: usize,
    groups: usize,
) -> Result<()> {
    let dev = &xn::CPU;
    let in_c_per_group = in_channels / groups;
    let n_in = batch * in_channels * length;
    let n_k = in_channels * out_c_per_group * kernel_size;
    // Deterministic, non-symmetric values so channel/offset mix-ups actually show up.
    let input: Tensor<f32, _> = Tensor::from_vec(
        (0..n_in).map(|i| (i % 7) as f32 - 3.).collect(),
        (batch, in_channels, length),
        dev,
    )?;
    let kernel: Tensor<f32, _> = Tensor::from_vec(
        (0..n_k).map(|i| ((i % 5) as f32 - 2.) * 0.5).collect(),
        (in_channels, out_c_per_group, kernel_size),
        dev,
    )?;

    let got = input.conv_transpose1d(&kernel, None, stride, 0, 0, groups)?;

    let mut parts = Vec::new();
    for g in 0..groups {
        let ic0 = g * in_c_per_group;
        let sub_in = input.narrow(1, ic0..ic0 + in_c_per_group)?.contiguous()?;
        let sub_k = kernel.narrow(0, ic0..ic0 + in_c_per_group)?.contiguous()?;
        parts.push(sub_in.conv_transpose1d(&sub_k, None, stride, 0, 0, 1)?);
    }
    let refs: Vec<&Tensor<f32, _>> = parts.iter().collect();
    let want = Tensor::cat(&refs, 1)?;

    assert_eq!(got.dims(), want.dims(), "shape mismatch");
    let (g, w) = (got.to_vec()?, want.to_vec()?);
    for (i, (a, b)) in g.iter().zip(w.iter()).enumerate() {
        assert!((a - b).abs() < 1e-4, "index {i}: got {a}, want {b}");
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Fused single-query attention (sdpa_decode)
// -----------------------------------------------------------------------------

/// Reference: the transpose/matmul/softmax sequence a caller would otherwise write.
fn sdpa_reference(
    q: &Tensor<f32, xn::CpuDevice>,
    k: &xn::TensorView<f32, xn::CpuDevice>,
    v: &xn::TensorView<f32, xn::CpuDevice>,
    mask: Option<&Tensor<f32, xn::CpuDevice>>,
    scale: f32,
) -> Result<Vec<f32>> {
    let d = q.dims()[3];
    let (b, h) = (q.dims()[0], q.dims()[2]);
    let qt = xn::TensorView::from(q).transpose(1, 2)?;
    let kt = k.transpose(1, 2)?;
    let vt = v.transpose(1, 2)?;
    let attn = qt.matmul_t(&kt)?.scale(scale)?;
    let attn = match mask {
        Some(m) => attn.broadcast_add(m)?,
        None => attn,
    };
    let attn = attn.softmax()?;
    attn.matmul(&vt)?.transpose(1, 2)?.reshape((b, 1, h * d))?.contiguous()?.to_vec()
}

fn check_sdpa(b: usize, h: usize, d: usize, kv: usize, cache_kv: usize) -> Result<()> {
    check_sdpa_masked(b, h, d, kv, cache_kv, false)?;
    check_sdpa_masked(b, h, d, kv, cache_kv, true)
}

fn check_sdpa_masked(
    b: usize,
    h: usize,
    d: usize,
    kv: usize,
    cache_kv: usize,
    masked: bool,
) -> Result<()> {
    let dev = &xn::CPU;
    let mut seed = 0x2545_f491u32;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) - 0.5
    };
    let q: Tensor<f32, _> =
        Tensor::from_vec((0..b * h * d).map(|_| rnd()).collect(), (b, 1, h, d), dev)?;
    // Allocate the cache at `cache_kv` and use only its first `kv` positions, so the operands
    // are a contiguous prefix of a longer storage -- the real decode situation.
    let kc: Tensor<f32, _> = Tensor::from_vec(
        (0..b * cache_kv * h * d).map(|_| rnd()).collect(),
        (b, cache_kv, h, d),
        dev,
    )?;
    let vc: Tensor<f32, _> = Tensor::from_vec(
        (0..b * cache_kv * h * d).map(|_| rnd()).collect(),
        (b, cache_kv, h, d),
        dev,
    )?;
    let k = kc.narrow(1, 0..kv)?;
    let v = vc.narrow(1, 0..kv)?;
    let scale = 1.0 / (d as f32).sqrt();
    // Sliding-window style: drop the oldest half of the context.
    let mask: Option<Tensor<f32, _>> = if masked {
        let cut = kv / 2;
        Some(Tensor::from_vec(
            (0..kv).map(|j| if j < cut { f32::NEG_INFINITY } else { 0.0 }).collect(),
            (1, 1, 1, kv),
            dev,
        )?)
    } else {
        None
    };

    let got = q.sdpa_decode(&k, &v, mask.as_ref(), scale)?;
    assert_eq!(got.dims(), &[b, 1, h * d]);
    let got = got.to_vec()?;
    let want = sdpa_reference(&q, &k, &v, mask.as_ref(), scale)?;
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= 1e-5 * w.abs().max(1.0),
            "b={b} h={h} d={d} kv={kv}: index {i}: got {g}, want {w}"
        );
    }
    Ok(())
}

#[test]
fn test_conv_transpose1d_depthwise_cpu() -> Result<()> {
    // groups == in_channels == out_channels: the depthwise branch.
    check_grouped_matches_per_group(1, 8, 1, 5, 4, 2, 8)?;
    check_grouped_matches_per_group(2, 6, 1, 4, 6, 3, 6)?;
    // Shaped like mimi's frame-rate adapter: kernel_size == 2 * stride, depthwise.
    check_grouped_matches_per_group(1, 16, 1, 3, 8, 4, 16)
}

#[test]
fn test_conv_transpose1d_grouped_cpu() -> Result<()> {
    // 1 < groups < in_channels, so in_c_per_group > 1 (the general fallback branch).
    check_grouped_matches_per_group(1, 8, 2, 5, 3, 1, 2)?;
    check_grouped_matches_per_group(2, 12, 3, 4, 4, 2, 4)?;
    // out_c_per_group > 1 together with in_c_per_group == 1 exercises the depthwise branch's
    // kernel indexing without groups == out_channels.
    check_grouped_matches_per_group(1, 4, 3, 4, 3, 2, 4)
}

#[test]
fn test_conv_transpose1d_padding_crops_cpu() -> Result<()> {
    // padding p on a transposed conv drops p elements from each end of the output.
    let dev = &xn::CPU;
    let input: Tensor<f32, _> =
        Tensor::from_vec((0..12).map(|i| (i % 5) as f32 - 2.).collect(), (1, 3, 4), dev)?;
    let kernel: Tensor<f32, _> =
        Tensor::from_vec((0..18).map(|i| (i % 4) as f32 - 1.).collect(), (3, 2, 3), dev)?;

    let unpadded = input.conv_transpose1d(&kernel, None, 2, 0, 0, 1)?;
    let full_len = unpadded.dims()[2];
    for p in 1..=2 {
        let padded = input.conv_transpose1d(&kernel, None, 2, p, 0, 1)?;
        assert_eq!(padded.dims(), &[1, 2, full_len - 2 * p], "padding {p} shape");
        let want = unpadded.narrow(2, p..full_len - p)?.contiguous()?.to_vec()?;
        let got = padded.to_vec()?;
        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "padding {p} index {i}: got {a}, want {b}");
        }
    }
    Ok(())
}

#[test]
fn test_conv_transpose1d_output_padding_appends_zeros_cpu() -> Result<()> {
    // output_padding only extends the output; the extra tail positions receive no
    // contributions and stay zero.
    let dev = &xn::CPU;
    let input: Tensor<f32, _> =
        Tensor::from_vec((0..8).map(|i| (i % 3) as f32 - 1.).collect(), (1, 2, 4), dev)?;
    let kernel: Tensor<f32, _> =
        Tensor::from_vec((0..8).map(|i| (i % 3) as f32).collect(), (2, 2, 2), dev)?;

    let base = input.conv_transpose1d(&kernel, None, 2, 0, 0, 1)?;
    let base_len = base.dims()[2];
    let base_v = base.to_vec()?;
    let q = 3;
    let extended = input.conv_transpose1d(&kernel, None, 2, 0, q, 1)?;
    assert_eq!(extended.dims(), &[1, 2, base_len + q]);
    let ext_v = extended.to_vec()?;
    for c in 0..2 {
        for i in 0..base_len {
            let (a, b) = (ext_v[c * (base_len + q) + i], base_v[c * base_len + i]);
            assert!((a - b).abs() < 1e-4, "ch {c} index {i}: got {a}, want {b}");
        }
        for i in base_len..base_len + q {
            assert_eq!(ext_v[c * (base_len + q) + i], 0., "ch {c} tail index {i} not zero");
        }
    }
    Ok(())
}

#[test]
fn test_sdpa_decode_matches_composed_cpu() -> Result<()> {
    // exact-size cache (no prefix), then genuine prefixes of a longer cache
    check_sdpa(1, 12, 64, 7, 7)?;
    check_sdpa(1, 12, 64, 200, 4096)?;
    check_sdpa(1, 1, 8, 1, 16)?;
    check_sdpa(2, 4, 16, 33, 64)?;
    check_sdpa(1, 8, 64, 1, 4096)
}

#[test]
fn test_sdpa_decode_rejects_bad_shapes_cpu() -> Result<()> {
    let dev = &xn::CPU;
    let q: Tensor<f32, _> = Tensor::full(0.5, (1, 1, 4, 8), dev)?;
    let kc: Tensor<f32, _> = Tensor::full(0.25, (1, 5, 4, 8), dev)?;
    let bad: Tensor<f32, _> = Tensor::full(0.25, (1, 5, 4, 16), dev)?;
    let k = kc.narrow(1, 0..5)?;
    let badv = bad.narrow(1, 0..5)?;
    // mismatched head dim between q and v
    assert!(q.sdpa_decode(&k, &badv, None, 1.0).is_err());
    // a query with more than one position is not a decode step
    let q2: Tensor<f32, _> = Tensor::full(0.5, (1, 2, 4, 8), dev)?;
    assert!(q2.sdpa_decode(&k, &k, None, 1.0).is_err());
    Ok(())
}

// =============================================================================
// Pad with same tests
// =============================================================================

fn test_pad_with_same_1d_impl<B: Backend>(dev: &B) -> Result<()> {
    // [1, 2, 3, 4] -> pad_with_same(0, 2, 3) -> [1, 1, 1, 2, 3, 4, 4, 4, 4]
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4.], (4,), dev)?;
    let b = a.pad_with_same(0, 2, 3)?;
    assert_eq!(b.dims(), &[9]);
    assert_eq!(b.to_vec()?, vec![1., 1., 1., 2., 3., 4., 4., 4., 4.]);
    Ok(())
}
test_all_backends!(test_pad_with_same_1d, test_pad_with_same_1d_impl);

fn test_pad_with_same_2d_dim0_impl<B: Backend>(dev: &B) -> Result<()> {
    // [2, 3] -> pad_with_same(0, 1, 1) -> [4, 3]
    // Replicates first and last rows
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;
    let b = a.pad_with_same(0, 1, 1)?;
    assert_eq!(b.dims(), &[4, 3]);
    // Row 0: copy of row 0 = [1,2,3]
    // Row 1: original row 0 = [1,2,3]
    // Row 2: original row 1 = [4,5,6]
    // Row 3: copy of row 1 = [4,5,6]
    assert_eq!(b.to_vec()?, vec![1., 2., 3., 1., 2., 3., 4., 5., 6., 4., 5., 6.]);
    Ok(())
}
test_all_backends!(test_pad_with_same_2d_dim0, test_pad_with_same_2d_dim0_impl);

fn test_pad_with_same_2d_dim1_impl<B: Backend>(dev: &B) -> Result<()> {
    // [2, 3] -> pad_with_same(1, 1, 2) -> [2, 6]
    // Replicates first and last columns
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;
    let b = a.pad_with_same(1, 1, 2)?;
    assert_eq!(b.dims(), &[2, 6]);
    // Row 0: [1, 1, 2, 3, 3, 3]
    // Row 1: [4, 4, 5, 6, 6, 6]
    assert_eq!(b.to_vec()?, vec![1., 1., 2., 3., 3., 3., 4., 4., 5., 6., 6., 6.]);
    Ok(())
}
test_all_backends!(test_pad_with_same_2d_dim1, test_pad_with_same_2d_dim1_impl);

fn test_pad_with_same_3d_impl<B: Backend>(dev: &B) -> Result<()> {
    // [2, 2, 2] -> pad_with_same(1, 1, 1) -> [2, 4, 2]
    let data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let a: Tensor<f32, B> = Tensor::from_vec(data, (2, 2, 2), dev)?;
    let b = a.pad_with_same(1, 1, 1)?;
    assert_eq!(b.dims(), &[2, 4, 2]);
    // First batch [2,2]: [[1,2], [3,4]] -> [[1,2], [1,2], [3,4], [3,4]]
    // Second batch [2,2]: [[5,6], [7,8]] -> [[5,6], [5,6], [7,8], [7,8]]
    assert_eq!(b.to_vec()?, vec![1., 2., 1., 2., 3., 4., 3., 4., 5., 6., 5., 6., 7., 8., 7., 8.]);
    Ok(())
}
test_all_backends!(test_pad_with_same_3d, test_pad_with_same_3d_impl);

// =============================================================================
// Sum keepdim tests
// =============================================================================

fn test_sum_keepdim_1d_impl<B: Backend>(dev: &B) -> Result<()> {
    // [5] -> sum_keepdim(0) -> [1]
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5.], (5,), dev)?;
    let b = a.sum_keepdim(vec![0])?;
    assert_eq!(b.dims(), &[1]);
    assert_eq!(b.to_vec()?, vec![15.]);
    Ok(())
}
test_all_backends!(test_sum_keepdim_1d, test_sum_keepdim_1d_impl);

fn test_sum_keepdim_2d_dim0_impl<B: Backend>(dev: &B) -> Result<()> {
    // [3, 4] -> sum_keepdim(0) -> [1, 4]
    let a: Tensor<f32, B> =
        Tensor::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.], (3, 4), dev)?;
    let b = a.sum_keepdim(vec![0])?;
    assert_eq!(b.dims(), &[1, 4]);
    // Column sums: 1+5+9=15, 2+6+10=18, 3+7+11=21, 4+8+12=24
    assert_eq!(b.to_vec()?, vec![15., 18., 21., 24.]);
    Ok(())
}
test_all_backends!(test_sum_keepdim_2d_dim0, test_sum_keepdim_2d_dim0_impl);

fn test_sum_keepdim_2d_dim1_impl<B: Backend>(dev: &B) -> Result<()> {
    // [3, 4] -> sum_keepdim(1) -> [3, 1]
    let a: Tensor<f32, B> =
        Tensor::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.], (3, 4), dev)?;
    let b = a.sum_keepdim(vec![1])?;
    assert_eq!(b.dims(), &[3, 1]);
    // Row sums: 1+2+3+4=10, 5+6+7+8=26, 9+10+11+12=42
    assert_eq!(b.to_vec()?, vec![10., 26., 42.]);
    Ok(())
}
test_all_backends!(test_sum_keepdim_2d_dim1, test_sum_keepdim_2d_dim1_impl);

fn test_sum_keepdim_3d_impl<B: Backend>(dev: &B) -> Result<()> {
    // [2, 3, 2] -> sum_keepdim(1) -> [2, 1, 2]
    let a: Tensor<f32, B> = Tensor::from_vec((1..=12).map(|x| x as f32).collect(), (2, 3, 2), dev)?;
    let b = a.sum_keepdim(vec![1])?;
    assert_eq!(b.dims(), &[2, 1, 2]);
    // Batch 0: [[1,2], [3,4], [5,6]] -> sum along dim 1 -> [9, 12]
    // Batch 1: [[7,8], [9,10], [11,12]] -> sum along dim 1 -> [27, 30]
    assert_eq!(b.to_vec()?, vec![9., 12., 27., 30.]);
    Ok(())
}
test_all_backends!(test_sum_keepdim_3d, test_sum_keepdim_3d_impl);

fn test_sum_keepdim_multiple_dims_impl<B: Backend>(dev: &B) -> Result<()> {
    // [2, 3, 4] -> sum_keepdim([1, 2]) -> [2, 1, 1]
    let a: Tensor<f32, B> = Tensor::from_vec((1..=24).map(|x| x as f32).collect(), (2, 3, 4), dev)?;
    let b = a.sum_keepdim(vec![1, 2])?;
    assert_eq!(b.dims(), &[2, 1, 1]);
    // Batch 0: sum of 1..12 = 78
    // Batch 1: sum of 13..24 = 222
    assert_eq!(b.to_vec()?, vec![78., 222.]);
    Ok(())
}
test_all_backends!(test_sum_keepdim_multiple_dims, test_sum_keepdim_multiple_dims_impl);

// =============================================================================
// Slice set tests
// =============================================================================

fn test_slice_set_dim0_impl<B: Backend>(dev: &B) -> Result<()> {
    // dst: [4, 3], src: [2, 3], set at offset 1 along dim 0
    let dst: Tensor<f32, B> = Tensor::zeros((4, 3), dev)?;
    let src: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;

    dst.slice_set(&src, 0, 1)?;
    // Row 0: [0, 0, 0]
    // Row 1: [1, 2, 3]
    // Row 2: [4, 5, 6]
    // Row 3: [0, 0, 0]
    assert_eq!(dst.to_vec()?, vec![0., 0., 0., 1., 2., 3., 4., 5., 6., 0., 0., 0.]);
    Ok(())
}
test_all_backends!(test_slice_set_dim0, test_slice_set_dim0_impl);

fn test_slice_set_dim1_impl<B: Backend>(dev: &B) -> Result<()> {
    // dst: [2, 6], src: [2, 3], set at offset 2 along dim 1
    let dst: Tensor<f32, B> = Tensor::zeros((2, 6), dev)?;
    let src: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;

    dst.slice_set(&src, 1, 2)?;
    // Row 0: [0, 0, 1, 2, 3, 0]
    // Row 1: [0, 0, 4, 5, 6, 0]
    assert_eq!(dst.to_vec()?, vec![0., 0., 1., 2., 3., 0., 0., 0., 4., 5., 6., 0.]);
    Ok(())
}
test_all_backends!(test_slice_set_dim1, test_slice_set_dim1_impl);

fn test_slice_set_3d_impl<B: Backend>(dev: &B) -> Result<()> {
    // dst: [2, 4, 3], src: [2, 2, 3], set at offset 1 along dim 1
    let dst: Tensor<f32, B> = Tensor::zeros((2, 4, 3), dev)?;
    let src: Tensor<f32, B> =
        Tensor::from_vec((1..=12).map(|x| x as f32).collect(), (2, 2, 3), dev)?;

    dst.slice_set(&src, 1, 1)?;
    // Batch 0: [[0,0,0], [1,2,3], [4,5,6], [0,0,0]]
    // Batch 1: [[0,0,0], [7,8,9], [10,11,12], [0,0,0]]
    assert_eq!(
        dst.to_vec()?,
        vec![
            0., 0., 0., 1., 2., 3., 4., 5., 6., 0., 0., 0., // batch 0
            0., 0., 0., 7., 8., 9., 10., 11., 12., 0., 0., 0. // batch 1
        ]
    );
    Ok(())
}
test_all_backends!(test_slice_set_3d, test_slice_set_3d_impl);

fn test_slice_set_at_start_impl<B: Backend>(dev: &B) -> Result<()> {
    // dst: [4, 2], src: [2, 2], set at offset 0
    let dst: Tensor<f32, B> = Tensor::full(9., (4, 2), dev)?;
    let src: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4.], (2, 2), dev)?;

    dst.slice_set(&src, 0, 0)?;
    assert_eq!(dst.to_vec()?, vec![1., 2., 3., 4., 9., 9., 9., 9.]);
    Ok(())
}
test_all_backends!(test_slice_set_at_start, test_slice_set_at_start_impl);

fn test_slice_set_at_end_impl<B: Backend>(dev: &B) -> Result<()> {
    // dst: [4, 2], src: [2, 2], set at offset 2 (at the end)
    let dst: Tensor<f32, B> = Tensor::full(9., (4, 2), dev)?;
    let src: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4.], (2, 2), dev)?;

    dst.slice_set(&src, 0, 2)?;
    assert_eq!(dst.to_vec()?, vec![9., 9., 9., 9., 1., 2., 3., 4.]);
    Ok(())
}
test_all_backends!(test_slice_set_at_end, test_slice_set_at_end_impl);

fn test_slice_set_1d_impl<B: Backend>(dev: &B) -> Result<()> {
    // dst: [8], src: [3], set at offset 2
    let dst: Tensor<f32, B> = Tensor::zeros((8,), dev)?;
    let src: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3.], (3,), dev)?;

    dst.slice_set(&src, 0, 2)?;
    assert_eq!(dst.to_vec()?, vec![0., 0., 1., 2., 3., 0., 0., 0.]);
    Ok(())
}
test_all_backends!(test_slice_set_1d, test_slice_set_1d_impl);

// =============================================================================
// Scatter tests
// =============================================================================

fn test_scatter_dim0_impl<B: Backend>(dev: &B) -> Result<()> {
    // dst: 3x3 zeros, scatter src values into dst along dim 0
    // Each (column, target row) pair is unique to avoid non-determinism with parallel writes.
    let dst: Tensor<f32, B> = Tensor::zeros((3, 3), dev)?;
    let src: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;
    let ids: Tensor<i64, B> = Tensor::from_vec(vec![0i64, 2, 1, 2, 0, 0], (2, 3), dev)?;

    let result = dst.scatter(&ids, &src, 0)?;
    assert_eq!(result.dims(), &[3, 3]);
    // Col 0: src[0][0]=1 -> row 0, src[1][0]=4 -> row 2
    // Col 1: src[0][1]=2 -> row 2, src[1][1]=5 -> row 0
    // Col 2: src[0][2]=3 -> row 1, src[1][2]=6 -> row 0
    assert_eq!(result.to_vec()?, vec![1., 5., 6., 0., 0., 3., 4., 2., 0.]);
    Ok(())
}
test_all_backends!(test_scatter_dim0, test_scatter_dim0_impl);

fn test_scatter_dim1_impl<B: Backend>(dev: &B) -> Result<()> {
    // dst: 2x4 zeros, scatter src into dst along dim 1
    let dst: Tensor<f32, B> = Tensor::zeros((2, 4), dev)?;
    let src: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;
    let ids: Tensor<i64, B> = Tensor::from_vec(vec![3i64, 0, 1, 2, 3, 0], (2, 3), dev)?;

    let result = dst.scatter(&ids, &src, 1)?;
    assert_eq!(result.dims(), &[2, 4]);
    // Row 0: col3=1, col0=2, col1=3 -> [2, 3, 0, 1]
    // Row 1: col2=4, col3=5, col0=6 -> [6, 0, 4, 5]
    assert_eq!(result.to_vec()?, vec![2., 3., 0., 1., 6., 0., 4., 5.]);
    Ok(())
}
test_all_backends!(test_scatter_dim1, test_scatter_dim1_impl);

fn test_scatter_set_dim0_impl<B: Backend>(dev: &B) -> Result<()> {
    // In-place scatter_set
    let dst: Tensor<f32, B> =
        Tensor::from_vec(vec![10., 20., 30., 40., 50., 60., 70., 80., 90.], (3, 3), dev)?;
    let src: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3.], (1, 3), dev)?;
    let ids: Tensor<i64, B> = Tensor::from_vec(vec![2i64, 0, 1], (1, 3), dev)?;

    dst.scatter_set(&ids, &src, 0)?;
    // Row 2 col 0 = 1, Row 0 col 1 = 2, Row 1 col 2 = 3
    assert_eq!(dst.to_vec()?, vec![10., 2., 30., 40., 50., 3., 1., 80., 90.]);
    Ok(())
}
test_all_backends!(test_scatter_set_dim0, test_scatter_set_dim0_impl);

fn test_scatter_3d_impl<B: Backend>(dev: &B) -> Result<()> {
    // 3D scatter along dim 1
    // dst: (2, 3, 2) zeros
    let dst: Tensor<f32, B> = Tensor::zeros((2, 3, 2), dev)?;
    // src: (2, 1, 2)
    let src: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4.], (2, 1, 2), dev)?;
    // ids: (2, 1, 2) - scatter along dim 1
    let ids: Tensor<i64, B> = Tensor::from_vec(vec![2i64, 0, 1, 2], (2, 1, 2), dev)?;

    let result = dst.scatter(&ids, &src, 1)?;
    assert_eq!(result.dims(), &[2, 3, 2]);
    // Batch 0: dst[0][2][0]=1, dst[0][0][1]=2
    // Batch 1: dst[1][1][0]=3, dst[1][2][1]=4
    assert_eq!(result.to_vec()?, vec![0., 2., 0., 0., 1., 0., 0., 0., 3., 0., 0., 4.]);
    Ok(())
}
test_all_backends!(test_scatter_3d, test_scatter_3d_impl);

// =============================================================================
// Broadcast tests
// =============================================================================

fn test_broadcast_as_add_dim_impl<B: Backend>(dev: &B) -> Result<()> {
    // (3,) -> (2, 3): broadcast by prepending a dimension
    let t: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3.], (3,), dev)?;
    let view = t.broadcast_as((2, 3))?;
    assert_eq!(view.dims(), &[2, 3]);
    let result = view.contiguous()?;
    assert_eq!(result.to_vec()?, vec![1., 2., 3., 1., 2., 3.]);
    Ok(())
}
test_all_backends!(test_broadcast_as_add_dim, test_broadcast_as_add_dim_impl);

fn test_broadcast_as_expand_dim_impl<B: Backend>(dev: &B) -> Result<()> {
    // (2, 1) -> (2, 3): expand dim of size 1
    let t: Tensor<f32, B> = Tensor::from_vec(vec![10., 20.], (2, 1), dev)?;
    let view = t.broadcast_as((2, 3))?;
    assert_eq!(view.dims(), &[2, 3]);
    let result = view.contiguous()?;
    assert_eq!(result.to_vec()?, vec![10., 10., 10., 20., 20., 20.]);
    Ok(())
}
test_all_backends!(test_broadcast_as_expand_dim, test_broadcast_as_expand_dim_impl);

fn test_broadcast_as_3d_impl<B: Backend>(dev: &B) -> Result<()> {
    // (1, 3) -> (2, 4, 3): prepend dim and expand dim 0
    let t: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3.], (1, 3), dev)?;
    let view = t.broadcast_as((2, 4, 3))?;
    assert_eq!(view.dims(), &[2, 4, 3]);
    let result = view.contiguous()?;
    let expected: Vec<f32> = [1., 2., 3.].repeat(8);
    assert_eq!(result.to_vec()?, expected);
    Ok(())
}
test_all_backends!(test_broadcast_as_3d, test_broadcast_as_3d_impl);

fn test_broadcast_as_noop_impl<B: Backend>(dev: &B) -> Result<()> {
    // (2, 3) -> (2, 3): no-op broadcast
    let t: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;
    let view = t.broadcast_as((2, 3))?;
    assert_eq!(view.dims(), &[2, 3]);
    let result = view.contiguous()?;
    assert_eq!(result.to_vec()?, vec![1., 2., 3., 4., 5., 6.]);
    Ok(())
}
test_all_backends!(test_broadcast_as_noop, test_broadcast_as_noop_impl);

fn test_broadcast_as_from_view_impl<B: Backend>(dev: &B) -> Result<()> {
    // TensorView::broadcast_as: narrow then broadcast
    let t: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3), dev)?;
    let view: TensorView<f32, B> = TensorView::from(&t);
    let narrowed = view.narrow(0, ..1)?; // (1, 3)
    let broadcast = narrowed.broadcast_as((3, 3))?;
    let result = broadcast.contiguous()?;
    assert_eq!(result.to_vec()?, vec![1., 2., 3., 1., 2., 3., 1., 2., 3.]);
    Ok(())
}
test_all_backends!(test_broadcast_as_from_view, test_broadcast_as_from_view_impl);

fn test_broadcast_as_error_impl<B: Backend>(dev: &B) -> Result<()> {
    // Incompatible shapes should error
    let t: Tensor<f32, B> = Tensor::from_vec(vec![1., 2., 3.], (3,), dev)?;
    assert!(t.broadcast_as((2, 4)).is_err());
    Ok(())
}
test_all_backends!(test_broadcast_as_error, test_broadcast_as_error_impl);

// =============================================================================
// Matmul with transposed view tests
// =============================================================================

fn test_matmul_transposed_view_impl<B: Backend>(dev: &B) -> Result<()> {
    // Simulate the attention pattern: Q @ K^T where K^T is a transposed TensorView.
    // Shape: Q is (1, 2, 3, 4), K is (1, 2, 3, 4), K^T via transpose(2,3) is (1, 2, 4, 3).
    // Result should be (1, 2, 3, 3).
    let q_data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let k_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1).collect();
    let q: Tensor<f32, B> = Tensor::from_vec(q_data, (1, 2, 3, 4), dev)?;
    let k: Tensor<f32, B> = Tensor::from_vec(k_data.clone(), (1, 2, 3, 4), dev)?;

    // Method 1: matmul with a transposed view (zero-copy transpose)
    let k_t_view = k.transpose(2, 3)?;
    let result_view = q.matmul(&k_t_view)?;

    // Method 2: matmul_t with the original K (the old reliable way)
    let k2: Tensor<f32, B> = Tensor::from_vec(k_data, (1, 2, 3, 4), dev)?;
    let result_matmul_t = q.matmul_t(&k2)?;

    // Both should produce the same result.
    let v1 = result_view.to_vec()?;
    let v2 = result_matmul_t.to_vec()?;
    assert_eq!(v1.len(), v2.len());
    for (a, b) in v1.iter().zip(v2.iter()) {
        assert!((a - b).abs() < 1e-4, "mismatch: {a} vs {b}");
    }
    Ok(())
}
test_all_backends!(test_matmul_transposed_view, test_matmul_transposed_view_impl);

// =============================================================================
// to (dtype cast) tests
// =============================================================================

fn test_to_f32_to_f16_impl<B: Backend>(dev: &B) -> Result<()> {
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1.0, 2.5, -3.0, 0.0], (2, 2), dev)?;
    let b: Tensor<half::f16, B> = a.to()?;
    assert_eq!(b.dims(), &[2, 2]);
    let result: Vec<f32> = b.to_vec()?.iter().map(|v| v.to_f32()).collect();
    assert_eq!(result, vec![1.0, 2.5, -3.0, 0.0]);
    Ok(())
}
test_all_backends!(test_to_f32_to_f16, test_to_f32_to_f16_impl);

fn test_to_f16_to_f32_impl<B: Backend>(dev: &B) -> Result<()> {
    let data: Vec<half::f16> =
        vec![1.0, 2.5, -3.0, 0.0].into_iter().map(half::f16::from_f32).collect();
    let a: Tensor<half::f16, B> = Tensor::from_vec(data, (4,), dev)?;
    let b: Tensor<f32, B> = a.to()?;
    assert_eq!(b.dims(), &[4]);
    assert_eq!(b.to_vec()?, vec![1.0, 2.5, -3.0, 0.0]);
    Ok(())
}
test_all_backends!(test_to_f16_to_f32, test_to_f16_to_f32_impl);

fn test_to_f32_to_bf16_impl<B: Backend>(dev: &B) -> Result<()> {
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1.0, -0.5, 100.0], (3,), dev)?;
    let b: Tensor<half::bf16, B> = a.to()?;
    let result: Vec<f32> = b.to_vec()?.iter().map(|v| v.to_f32()).collect();
    assert_eq!(result, vec![1.0, -0.5, 100.0]);
    Ok(())
}
test_all_backends!(test_to_f32_to_bf16, test_to_f32_to_bf16_impl);

fn test_to_bf16_to_f32_impl<B: Backend>(dev: &B) -> Result<()> {
    let data: Vec<half::bf16> =
        vec![1.0, -0.5, 100.0].into_iter().map(half::bf16::from_f32).collect();
    let a: Tensor<half::bf16, B> = Tensor::from_vec(data, (3,), dev)?;
    let b: Tensor<f32, B> = a.to()?;
    assert_eq!(b.to_vec()?, vec![1.0, -0.5, 100.0]);
    Ok(())
}
test_all_backends!(test_to_bf16_to_f32, test_to_bf16_to_f32_impl);

fn test_to_f16_to_bf16_impl<B: Backend>(dev: &B) -> Result<()> {
    let data: Vec<half::f16> = vec![1.0, 2.0, -3.0].into_iter().map(half::f16::from_f32).collect();
    let a: Tensor<half::f16, B> = Tensor::from_vec(data, (3,), dev)?;
    let b: Tensor<half::bf16, B> = a.to()?;
    let result: Vec<f32> = b.to_vec()?.iter().map(|v| v.to_f32()).collect();
    assert_eq!(result, vec![1.0, 2.0, -3.0]);
    Ok(())
}
test_all_backends!(test_to_f16_to_bf16, test_to_f16_to_bf16_impl);

fn test_to_f32_to_i64_impl<B: Backend>(dev: &B) -> Result<()> {
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1.9, -2.1, 0.0, 42.0], (4,), dev)?;
    let b: Tensor<i64, B> = a.to()?;
    assert_eq!(b.to_vec()?, vec![1i64, -2, 0, 42]);
    Ok(())
}
test_all_backends!(test_to_f32_to_i64, test_to_f32_to_i64_impl);

fn test_to_i64_to_f32_impl<B: Backend>(dev: &B) -> Result<()> {
    let a: Tensor<i64, B> = Tensor::from_vec(vec![1, -2, 0, 42], (4,), dev)?;
    let b: Tensor<f32, B> = a.to()?;
    assert_eq!(b.to_vec()?, vec![1.0, -2.0, 0.0, 42.0]);
    Ok(())
}
test_all_backends!(test_to_i64_to_f32, test_to_i64_to_f32_impl);

fn test_to_f32_to_u8_impl<B: Backend>(dev: &B) -> Result<()> {
    let a: Tensor<f32, B> = Tensor::from_vec(vec![0.0, 1.0, 127.0, 255.0], (4,), dev)?;
    let b: Tensor<u8, B> = a.to()?;
    assert_eq!(b.to_vec()?, vec![0u8, 1, 127, 255]);
    Ok(())
}
test_all_backends!(test_to_f32_to_u8, test_to_f32_to_u8_impl);

fn test_to_u8_to_f32_impl<B: Backend>(dev: &B) -> Result<()> {
    let a: Tensor<u8, B> = Tensor::from_vec(vec![0, 1, 127, 255], (4,), dev)?;
    let b: Tensor<f32, B> = a.to()?;
    assert_eq!(b.to_vec()?, vec![0.0, 1.0, 127.0, 255.0]);
    Ok(())
}
test_all_backends!(test_to_u8_to_f32, test_to_u8_to_f32_impl);

fn test_to_i64_to_u8_impl<B: Backend>(dev: &B) -> Result<()> {
    let a: Tensor<i64, B> = Tensor::from_vec(vec![0, 1, 127, 255], (4,), dev)?;
    let b: Tensor<u8, B> = a.to()?;
    assert_eq!(b.to_vec()?, vec![0u8, 1, 127, 255]);
    Ok(())
}
test_all_backends!(test_to_i64_to_u8, test_to_i64_to_u8_impl);

fn test_to_same_dtype_impl<B: Backend>(dev: &B) -> Result<()> {
    let a: Tensor<f32, B> = Tensor::from_vec(vec![1.0, 2.0, 3.0], (3,), dev)?;
    let b: Tensor<f32, B> = a.to()?;
    assert_eq!(b.to_vec()?, vec![1.0, 2.0, 3.0]);
    Ok(())
}
test_all_backends!(test_to_same_dtype, test_to_same_dtype_impl);

fn test_to_preserves_shape_impl<B: Backend>(dev: &B) -> Result<()> {
    let a: Tensor<f32, B> = Tensor::from_vec((1..=24).map(|v| v as f32).collect(), (2, 3, 4), dev)?;
    let b: Tensor<half::f16, B> = a.to()?;
    assert_eq!(b.dims(), &[2, 3, 4]);
    assert_eq!(b.elem_count(), 24);
    // Verify first and last values roundtrip.
    let result: Vec<f32> = b.to_vec()?.iter().map(|v| v.to_f32()).collect();
    assert_eq!(result[0], 1.0);
    assert_eq!(result[23], 24.0);
    Ok(())
}
test_all_backends!(test_to_preserves_shape, test_to_preserves_shape_impl);

// =============================================================================
// Random generation tests
// =============================================================================

fn test_rand_uniform_impl<B: Backend>(dev: &B) -> Result<()> {
    let t: Tensor<f32, B> = Tensor::from_vec(vec![0.0; 1000], 1000, dev)?;
    let r = t.rand_uniform_like(0.0, 1.0)?;
    assert_eq!(r.dims(), &[1000]);
    let vals = r.to_vec()?;
    for &v in &vals {
        assert!((0.0..=1.0).contains(&v), "rand_uniform value {v} out of [0, 1]");
    }
    // Check it's not all the same value (extremely unlikely for 1000 elements).
    assert!(vals.windows(2).any(|w| w[0] != w[1]));
    Ok(())
}
test_all_backends!(test_rand_uniform, test_rand_uniform_impl);

fn test_rand_uniform_shape_impl<B: Backend>(dev: &B) -> Result<()> {
    let t: Tensor<f32, B> = Tensor::from_vec(vec![0.0], 1, dev)?;
    let r = t.rand_uniform((4, 8), 0.0, 1.0)?;
    assert_eq!(r.dims(), &[4, 8]);
    let vals = r.to_vec()?;
    assert_eq!(vals.len(), 32);
    for &v in &vals {
        assert!((0.0..=1.0).contains(&v), "rand_uniform value {v} out of [0, 1]");
    }
    Ok(())
}
test_all_backends!(test_rand_uniform_shape, test_rand_uniform_shape_impl);

fn test_rand_uniform_bounds_impl<B: Backend>(dev: &B) -> Result<()> {
    let t: Tensor<f32, B> = Tensor::from_vec(vec![0.0; 1000], 1000, dev)?;
    let r = t.rand_uniform_like(-5.0, 3.0)?;
    let vals = r.to_vec()?;
    for &v in &vals {
        assert!((-5.0..=3.0).contains(&v), "rand_uniform value {v} out of [-5, 3]");
    }
    assert!(vals.windows(2).any(|w| w[0] != w[1]));
    Ok(())
}
test_all_backends!(test_rand_uniform_bounds, test_rand_uniform_bounds_impl);

fn test_rand_uniform_invalid_bounds_impl<B: Backend>(dev: &B) -> Result<()> {
    let t: Tensor<f32, B> = Tensor::from_vec(vec![0.0; 10], 10, dev)?;
    let r = t.rand_uniform_like(5.0, 2.0);
    assert!(r.is_err());
    let err_msg = r.unwrap_err().to_string();
    assert!(err_msg.contains("upper bound"), "error should mention upper bound: {err_msg}");
    Ok(())
}
test_all_backends!(test_rand_uniform_invalid_bounds, test_rand_uniform_invalid_bounds_impl);

fn test_randn_impl<B: Backend>(dev: &B) -> Result<()> {
    let t: Tensor<f32, B> = Tensor::from_vec(vec![0.0; 1000], 1000, dev)?;
    let r = t.randn_like(0.0, 1.0)?;
    assert_eq!(r.dims(), &[1000]);
    let vals = r.to_vec()?;
    for &v in &vals {
        assert!(v.is_finite(), "randn value is not finite: {v}");
    }
    // Check it's not all the same value.
    assert!(vals.windows(2).any(|w| w[0] != w[1]));
    Ok(())
}
test_all_backends!(test_randn, test_randn_impl);

fn test_randn_shape_impl<B: Backend>(dev: &B) -> Result<()> {
    let t: Tensor<f32, B> = Tensor::from_vec(vec![0.0], 1, dev)?;
    let r = t.randn((4, 8), 5.0, 0.1)?;
    assert_eq!(r.dims(), &[4, 8]);
    let vals = r.to_vec()?;
    assert_eq!(vals.len(), 32);
    for &v in &vals {
        assert!(v.is_finite(), "randn value is not finite: {v}");
    }
    Ok(())
}
test_all_backends!(test_randn_shape, test_randn_shape_impl);

// =============================================================================
// Unary op tests
// =============================================================================

// =============================================================================
// Rope tests
// =============================================================================

/// Test that rope_i with batched (3D) cos/sin correctly applies per-batch rotary embeddings.
///
/// When cos/sin are 3D (batch, t, half_dim), each batch element should use its own
/// cos/sin values. This test creates two batch elements with different cos/sin:
///   batch 0: identity rotation (cos=1, sin=0)
///   batch 1: 90-degree rotation (cos=0, sin=1)
///
/// The interleaved rope formula is:
///   dst[2i]   = src[2i] * cos[i] - src[2i+1] * sin[i]
///   dst[2i+1] = src[2i] * sin[i] + src[2i+1] * cos[i]
fn test_rope_i_batched_cos_sin_impl<B: Backend>(dev: &B) -> Result<()> {
    // Input: (batch=2, heads=1, t=1, d=4)
    // Both batch elements have the same input values.
    let src: Tensor<f32, B> = Tensor::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, // batch 0
            1.0, 2.0, 3.0, 4.0, // batch 1
        ],
        (2, 1, 1, 4),
        dev,
    )?;

    // Batched cos/sin: (batch=2, t=1, half_dim=2)
    // batch 0: cos=[1,1] sin=[0,0] → identity
    // batch 1: cos=[0,0] sin=[1,1] → 90-degree rotation
    let cos: Tensor<f32, B> = Tensor::from_vec(vec![1.0, 1.0, 0.0, 0.0], (2, 1, 2), dev)?;
    let sin: Tensor<f32, B> = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], (2, 1, 2), dev)?;

    let out = src.rope_i(&cos, &sin, 0)?;
    let out_data = out.to_vec()?;

    // batch 0 (identity): [1*1 - 2*0, 1*0 + 2*1, 3*1 - 4*0, 3*0 + 4*1] = [1, 2, 3, 4]
    assert_approx_eq(&out_data[0..4], &[1.0, 2.0, 3.0, 4.0], 1e-5);

    // batch 1 (90-degree): [1*0 - 2*1, 1*1 + 2*0, 3*0 - 4*1, 3*1 + 4*0] = [-2, 1, -4, 3]
    assert_approx_eq(&out_data[4..8], &[-2.0, 1.0, -4.0, 3.0], 1e-5);

    Ok(())
}
test_all_backends!(test_rope_i_batched_cos_sin, test_rope_i_batched_cos_sin_impl);

/// Same test but for the non-interleaved rope variant.
///
/// The non-interleaved rope formula (for head_dim=d) splits into first and second halves:
///   dst[i]       = src[i]     * cos[i] - src[i+d/2] * sin[i]    for i in 0..d/2
///   dst[i+d/2]   = src[i]     * sin[i] + src[i+d/2] * cos[i]    for i in 0..d/2
fn test_rope_batched_cos_sin_impl<B: Backend>(dev: &B) -> Result<()> {
    // Input: (batch=2, heads=1, t=1, d=4)
    let src: Tensor<f32, B> = Tensor::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, // batch 0
            1.0, 2.0, 3.0, 4.0, // batch 1
        ],
        (2, 1, 1, 4),
        dev,
    )?;

    // Batched cos/sin: (batch=2, t=1, half_dim=2)
    let cos: Tensor<f32, B> = Tensor::from_vec(vec![1.0, 1.0, 0.0, 0.0], (2, 1, 2), dev)?;
    let sin: Tensor<f32, B> = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], (2, 1, 2), dev)?;

    let out = src.rope(&cos, &sin, 0)?;
    let out_data = out.to_vec()?;

    // batch 0 (identity): cos=[1,1], sin=[0,0]
    //   dst[0] = 1*1 - 3*0 = 1, dst[1] = 2*1 - 4*0 = 2
    //   dst[2] = 1*0 + 3*1 = 3, dst[3] = 2*0 + 4*1 = 4
    assert_approx_eq(&out_data[0..4], &[1.0, 2.0, 3.0, 4.0], 1e-5);

    // batch 1 (90-degree): cos=[0,0], sin=[1,1]
    //   dst[0] = 1*0 - 3*1 = -3, dst[1] = 2*0 - 4*1 = -4
    //   dst[2] = 1*1 + 3*0 = 1,  dst[3] = 2*1 + 4*0 = 2
    assert_approx_eq(&out_data[4..8], &[-3.0, -4.0, 1.0, 2.0], 1e-5);

    Ok(())
}
test_all_backends!(test_rope_batched_cos_sin, test_rope_batched_cos_sin_impl);

fn assert_approx_eq(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() < tol, "mismatch at index {i}: {x} vs {y}");
    }
}

fn test_exp_impl<B: Backend>(dev: &B) -> Result<()> {
    let data = vec![0.0f32, 1.0, -1.0, 2.0];
    let expected: Vec<f32> = data.iter().map(|x| x.exp()).collect();
    let src: Tensor<f32, B> = Tensor::from_vec(data, 4, dev)?;
    let dst: Tensor<f32, B> = Tensor::zeros(4, dev)?;
    dst.exp_(&src)?;
    assert_approx_eq(&dst.to_vec()?, &expected, 1e-6);
    Ok(())
}
test_all_backends!(test_exp, test_exp_impl);

fn test_log_impl<B: Backend>(dev: &B) -> Result<()> {
    let data = vec![1.0f32, 2.0, 0.5, 10.0];
    let expected: Vec<f32> = data.iter().map(|x| x.ln()).collect();
    let src: Tensor<f32, B> = Tensor::from_vec(data, 4, dev)?;
    let dst: Tensor<f32, B> = Tensor::zeros(4, dev)?;
    dst.log_(&src)?;
    assert_approx_eq(&dst.to_vec()?, &expected, 1e-6);
    Ok(())
}
test_all_backends!(test_log, test_log_impl);

fn test_neg_impl<B: Backend>(dev: &B) -> Result<()> {
    let data = vec![1.0f32, -2.0, 0.0, 3.5];
    let expected: Vec<f32> = data.iter().map(|x| -x).collect();
    let src: Tensor<f32, B> = Tensor::from_vec(data, 4, dev)?;
    let dst: Tensor<f32, B> = Tensor::zeros(4, dev)?;
    dst.neg_(&src)?;
    assert_approx_eq(&dst.to_vec()?, &expected, 1e-6);
    Ok(())
}
test_all_backends!(test_neg, test_neg_impl);

fn test_exp_log_compose_impl<B: Backend>(dev: &B) -> Result<()> {
    let data = vec![0.5f32, 1.0, 2.0, 3.0];
    let expected: Vec<f32> = data.iter().map(|x| x.exp().ln()).collect();
    let src: Tensor<f32, B> = Tensor::from_vec(data, 4, dev)?;
    let tmp: Tensor<f32, B> = Tensor::zeros(4, dev)?;
    let dst: Tensor<f32, B> = Tensor::zeros(4, dev)?;
    tmp.exp_(&src)?;
    dst.log_(&tmp)?;
    assert_approx_eq(&dst.to_vec()?, &expected, 1e-5);
    Ok(())
}
test_all_backends!(test_exp_log_compose, test_exp_log_compose_impl);

fn test_log_neg_roundtrip_impl<B: Backend>(dev: &B) -> Result<()> {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let src: Tensor<f32, B> = Tensor::from_vec(data.clone(), 4, dev)?;
    let tmp: Tensor<f32, B> = Tensor::zeros(4, dev)?;
    let dst: Tensor<f32, B> = Tensor::zeros(4, dev)?;
    // exp(log(x)) should give back x
    tmp.log_(&src)?;
    dst.exp_(&tmp)?;
    assert_approx_eq(&dst.to_vec()?, &data, 1e-5);
    // neg(neg(x)) should give back x
    tmp.neg_(&src)?;
    dst.neg_(&tmp)?;
    assert_approx_eq(&dst.to_vec()?, &data, 1e-6);
    Ok(())
}
test_all_backends!(test_log_neg_roundtrip, test_log_neg_roundtrip_impl);

fn test_narrow_prefix_contiguous_impl<B: Backend>(dev: &B) -> Result<()> {
    // A zero-offset narrow with leading size-1 dims is a contiguous *prefix* of its
    // storage; contiguous() must not return a tensor aliasing the longer storage.
    // Regression test: q/k/v projections narrow a (1, 1, 3*d) qkv tensor and the q slice
    // starts at offset 0, which used to break ops validating storage length (e.g. rope_i).
    let (h, d) = (2, 4);
    let dm = h * d;
    let qkv_data: Vec<f32> = (0..3 * dm).map(|v| v as f32).collect();
    let qkv: Tensor<f32, B> = Tensor::from_vec(qkv_data, (1, 1, 3 * dm), dev)?;
    let q = qkv.narrow(2, ..dm)?.reshape((1, 1, h, d))?.transpose(1, 2)?.contiguous()?;
    assert_eq!(q.dims(), [1, h, 1, d]);
    let expected: Vec<f32> = (0..dm).map(|v| v as f32).collect();
    assert_eq!(q.to_vec()?, expected);
    // rope_i validates that the storage length matches the shape.
    let cos: Tensor<f32, B> = Tensor::full(1.0, (8, d / 2), dev)?;
    let sin: Tensor<f32, B> = Tensor::full(0.0, (8, d / 2), dev)?;
    let roped = q.rope_i(&cos, &sin, 0)?;
    assert_eq!(roped.to_vec()?, expected);
    Ok(())
}
test_all_backends!(test_narrow_prefix_contiguous, test_narrow_prefix_contiguous_impl);

fn test_matmul_batched_strided_impl<B: Backend>(dev: &B) -> Result<()> {
    // Batched matmuls where an operand is a transpose(1, 2) view of a (b, t, h, d)
    // tensor: the batch dims (b, h) of the view have strides (t*h*d, d), which do not
    // collapse to a single uniform batch stride when b > 1. Regression test: batch
    // entries past the first used to read from wrong offsets (the attention pattern of
    // a batched transformer, where only batch row 0 came out right).
    let (b, t, h, d) = (3, 5, 2, 4);
    let q_data: Vec<f32> = (0..b * t * h * d).map(|i| ((i * 7 % 23) as f32) - 11.0).collect();
    let k_data: Vec<f32> = (0..b * t * h * d).map(|i| ((i * 5 % 19) as f32) - 9.0).collect();
    let q: Tensor<f32, B> = Tensor::from_vec(q_data, (b, t, h, d), dev)?;
    let k: Tensor<f32, B> = Tensor::from_vec(k_data, (b, t, h, d), dev)?;

    // (b, t, h, d) -> (b, h, t, d) views with non-collapsible batch strides.
    let q_view = q.transpose(1, 2)?;
    let k_view = k.transpose(1, 2)?;
    let q_c = q.transpose(1, 2)?.contiguous()?;
    let k_c = k.transpose(1, 2)?.contiguous()?;

    // matmul_t: (b, h, t, d) x (b, h, t, d)^T -> (b, h, t, t)
    let expected = q_c.matmul_t(&k_c)?.to_vec()?;
    assert_approx_eq(&q_view.matmul_t(&k_c)?.to_vec()?, &expected, 1e-5);
    assert_approx_eq(&q_c.matmul_t(&k_view)?.to_vec()?, &expected, 1e-5);
    assert_approx_eq(&q_view.matmul_t(&k_view)?.to_vec()?, &expected, 1e-5);

    // matmul: (b, h, t, d) x (b, h, d, t) -> (b, h, t, t)
    let k_tt_view = k.transpose(1, 2)?.transpose(2, 3)?;
    let k_tt_c = k.transpose(1, 2)?.transpose(2, 3)?.contiguous()?;
    let expected = q_c.matmul(&k_tt_c)?.to_vec()?;
    assert_approx_eq(&q_view.matmul(&k_tt_c)?.to_vec()?, &expected, 1e-5);
    assert_approx_eq(&q_c.matmul(&k_tt_view)?.to_vec()?, &expected, 1e-5);
    assert_approx_eq(&q_view.matmul(&k_tt_view)?.to_vec()?, &expected, 1e-5);
    Ok(())
}
test_all_backends!(test_matmul_batched_strided, test_matmul_batched_strided_impl);

//! An XLA backend built on top of PJRT, with lazy graph capture.
//!
//! Storage lives on the PJRT device as rank-1 buffers, but operations do not
//! execute immediately: they record nodes of a computation graph (see the
//! [`lazy`] module) which is flushed when data is read back on the host or
//! `synchronize` is called. Recurring graph structures are compiled into a
//! single fused XLA executable and replayed, so steady-state inference loops
//! pay one host round-trip per step; the first occurrence of a structure runs
//! node by node with per-op executables, which keeps shape-changing workloads
//! from recompiling whole graphs. Values that change per step without
//! affecting shapes (offsets, positions) are runtime inputs rather than graph
//! constants, keeping the caches hot in decoding loops.
mod lazy;

use crate::{BinaryOp, DType, Result, UnaryOp, WithDType, WithDTypeF};
use lazy::{FusePolicy, Input, LazyNode, NodeState, Value};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, Weak};
use xla::{ElementType, XlaBuilder, XlaOp};

fn xerr(err: xla::Error) -> crate::Error {
    crate::Error::wrap(err)
}

fn ety(dtype: DType) -> ElementType {
    match dtype {
        DType::F16 => ElementType::F16,
        DType::BF16 => ElementType::Bf16,
        DType::F32 => ElementType::F32,
        DType::I64 => ElementType::S64,
        DType::U8 => ElementType::U8,
    }
}

fn dtype_key(dtype: DType) -> i64 {
    match dtype {
        DType::F16 => 0,
        DType::BF16 => 1,
        DType::F32 => 2,
        DType::I64 => 3,
        DType::U8 => 4,
    }
}

fn unary_key(op: UnaryOp) -> [i64; 2] {
    match op {
        UnaryOp::Cos => [0, 0],
        UnaryOp::Sin => [1, 0],
        UnaryOp::Exp => [2, 0],
        UnaryOp::Log => [3, 0],
        UnaryOp::Neg => [4, 0],
        UnaryOp::Sqr => [5, 0],
        UnaryOp::Sqrt => [6, 0],
        UnaryOp::Rsqrt => [7, 0],
        UnaryOp::Abs => [8, 0],
        UnaryOp::GeluErf => [9, 0],
        UnaryOp::Elu { alpha } => [10, alpha.to_bits() as i64],
        UnaryOp::Relu => [11, 0],
        UnaryOp::Silu => [12, 0],
        UnaryOp::Tanh => [13, 0],
        UnaryOp::Sigmoid => [14, 0],
    }
}

fn binary_key(op: BinaryOp) -> i64 {
    match op {
        BinaryOp::Add => 0,
        BinaryOp::Sub => 1,
        BinaryOp::Mul => 2,
        BinaryOp::Div => 3,
        BinaryOp::Maximum => 4,
        BinaryOp::Minimum => 5,
    }
}

type ExeCache<K> = Mutex<HashMap<K, Arc<xla::PjRtLoadedExecutable>>>;

pub(crate) struct Inner {
    client: xla::PjRtClient,
    /// Per-op executables, used the first time a graph structure is seen.
    node_cache: ExeCache<(&'static str, Vec<i64>)>,
    /// Whole-graph fused executables keyed by graph structure.
    graph_cache: ExeCache<Vec<u64>>,
    /// Graph structures seen once (candidates for fusion on repeat).
    seen_graphs: Mutex<HashSet<Vec<u64>>>,
    /// Pending nodes, flushed by `synchronize` or when the graph gets large.
    pending: Mutex<Vec<Weak<LazyNode>>>,
    /// Serializes flushes.
    flush_lock: Mutex<()>,
    fuse_policy: FusePolicy,
}

/// Flush automatically once this many nodes are pending.
const MAX_PENDING: usize = 4000;

#[derive(Clone)]
pub struct Device {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "XlaDevice({})", self.inner.client.platform_name())
    }
}

impl Device {
    fn from_client(client: xla::PjRtClient) -> Self {
        let inner = Inner {
            client,
            node_cache: Mutex::new(HashMap::new()),
            graph_cache: Mutex::new(HashMap::new()),
            seen_graphs: Mutex::new(HashSet::new()),
            pending: Mutex::new(Vec::new()),
            flush_lock: Mutex::new(()),
            fuse_policy: FusePolicy::from_env(),
        };
        Self { inner: Arc::new(inner) }
    }

    /// Create a device for the best available PJRT platform (TPU, then GPU,
    /// then CPU). The `_device_id` argument is currently ignored, PJRT picks
    /// its default device.
    pub fn new(_device_id: usize) -> Result<Self> {
        let client = xla::PjRtClient::auto(/* force_cpu= */ false).map_err(xerr)?;
        Ok(Self::from_client(client))
    }

    /// Create a device backed by the PJRT CPU client.
    pub fn cpu() -> Result<Self> {
        let client = xla::PjRtClient::cpu().map_err(xerr)?;
        Ok(Self::from_client(client))
    }

    pub fn platform_name(&self) -> String {
        self.inner.client.platform_name()
    }

    fn upload(&self, ty: ElementType, bytes: &[u8], len: usize) -> Result<xla::PjRtBuffer> {
        self.inner.client.buffer_from_host_raw_bytes(ty, bytes, &[len], None).map_err(xerr)
    }

    /// A one-element buffer holding a runtime scalar of the storage dtype.
    fn scalar_t<T: WithDType>(&self, v: T) -> Result<Input> {
        let bytes = unsafe {
            std::slice::from_raw_parts(&v as *const T as *const u8, std::mem::size_of::<T>())
        };
        let buf = self.upload(ety(T::DTYPE), bytes, 1)?;
        Ok(Input::Value(Value::Buffer { buf: Arc::new(buf), len: 1, ty: ety(T::DTYPE) }))
    }
}

pub struct Storage<T: WithDType> {
    value: Value,
    len: usize,
    device: Device,
    _phantom: std::marker::PhantomData<T>,
}

fn arg<T: WithDType>(st: &Storage<T>) -> Input {
    Input::Value(st.value.clone())
}

/// Record a lazy node: `build` produces the rank-1 result of length
/// `out_len` from the rank-1 inputs, and runs when the graph is flushed.
fn lazy_op(
    dev: &Device,
    op: &'static str,
    key_extra: &[i64],
    inputs: Vec<Input>,
    out_len: usize,
    out_dtype: DType,
    build: impl Fn(&XlaBuilder, &[XlaOp]) -> xla::Result<XlaOp> + Send + Sync + 'static,
) -> Result<Value> {
    let mut key = Vec::with_capacity(key_extra.len() + inputs.len() * 2 + 2);
    key.extend_from_slice(key_extra);
    key.push(out_len as i64);
    key.push(dtype_key(out_dtype));
    for i in inputs.iter() {
        match i {
            Input::Scalar(_) => key.extend([-1, -1]),
            Input::Value(v) => key.extend([v.len() as i64, v.ty().primitive_type() as i64]),
        }
    }
    let node = Arc::new(LazyNode {
        op,
        key,
        out_len,
        out_ty: ety(out_dtype),
        state: Mutex::new(NodeState::Pending { inputs, build: Box::new(build) }),
    });
    let value = Value::Node(node.clone());
    let flush_now = {
        let mut pending = dev.inner.pending.lock().unwrap();
        pending.push(Arc::downgrade(&node));
        pending.len() >= MAX_PENDING
    };
    if flush_now {
        lazy::flush_all(dev)?;
    }
    Ok(value)
}

/// When an operation only writes the first `len` elements of a destination of
/// `dst_len` elements, keep the destination tail unchanged by concatenating it
/// back after the freshly computed prefix.
fn splice(dst_p: &XlaOp, dst_len: usize, len: usize, prefix: XlaOp) -> xla::Result<XlaOp> {
    if len == dst_len {
        Ok(prefix)
    } else {
        let tail = dst_p.slice_in_dim(len as i64, dst_len as i64, 1, 0)?;
        prefix.concat_in_dim(&[tail], 0)
    }
}

/// Flat indices into a rank-1 array for an affine layout: for each position
/// `(i_0, .., i_r)` of `dims`, the index is `base + sum_d i_d * strides[d]`,
/// where `base` is a runtime scalar op (S64).
fn affine_index(
    b: &XlaBuilder,
    base: &XlaOp,
    dims: &[i64],
    strides: &[usize],
) -> xla::Result<XlaOp> {
    let mut acc = base.broadcast(dims)?;
    for (d, &stride) in strides.iter().enumerate() {
        if stride == 0 || dims[d] == 1 {
            continue;
        }
        let iota = b.iota(ElementType::S64, dims, d as i64)?;
        let term = iota.mul_(&b.c0(stride as i64)?)?;
        acc = acc.add_(&term)?;
    }
    Ok(acc)
}

/// Gather elements of a rank-1 array at the given (multi-dimensional) indices,
/// the result has the shape of `indices`.
fn gather_1d(src: &XlaOp, indices: &XlaOp) -> xla::Result<XlaOp> {
    src.take(indices, 0)
}

/// A computation combining an old and an update scalar by keeping the update,
/// used by scatter to overwrite the target elements.
fn replace_computation(ty: ElementType) -> xla::Result<xla::XlaComputation> {
    let b = XlaBuilder::new("replace");
    let _old = b.parameter(0, ty, &[], "old")?;
    let new = b.parameter(1, ty, &[], "new")?;
    b.build(&new)
}

/// Scatter `updates` (rank-1, n elements) into `dst` (rank-1) at the flat
/// positions given by `indices` (rank-1, n elements, S64).
fn scatter_1d(
    dst: &XlaOp,
    indices: &XlaOp,
    updates: &XlaOp,
    ty: ElementType,
) -> xla::Result<XlaOp> {
    let n = match indices.array_shape()?.dims() {
        [n] => *n,
        _ => {
            return Err(xla::Error::XlaError {
                msg: "expected a rank-1 index array".to_string(),
                backtrace: String::new(),
            });
        }
    };
    let indices = indices.reshape(&[n, 1])?;
    let comp = replace_computation(ty)?;
    dst.scatter(&indices, updates, &comp, &[], &[0], &[0], 1)
}

/// Convert to f32 for float computations that the cpu backend also performs
/// in f32 (the `half` crate computes through f32 as well).
fn to_f32(x: &XlaOp) -> xla::Result<XlaOp> {
    x.convert(ElementType::F32.primitive_type())
}

fn from_f32(x: &XlaOp, dtype: DType) -> xla::Result<XlaOp> {
    x.convert(ety(dtype).primitive_type())
}

fn build_unary(x: &XlaOp, op: UnaryOp, dtype: DType) -> xla::Result<XlaOp> {
    let b = x.builder().clone();
    let x = to_f32(x)?;
    let res = match op {
        UnaryOp::Cos => x.cos()?,
        UnaryOp::Sin => x.sin()?,
        UnaryOp::Exp => x.exp()?,
        UnaryOp::Log => x.log()?,
        UnaryOp::Neg => x.neg()?,
        UnaryOp::Sqr => x.mul_(&x)?,
        UnaryOp::Sqrt => x.sqrt()?,
        UnaryOp::Rsqrt => x.rsqrt()?,
        UnaryOp::Abs => x.abs()?,
        UnaryOp::GeluErf => x.gelu_erf()?,
        UnaryOp::Elu { alpha } => {
            let zero = b.c0(0f32)?;
            let alpha = b.c0(alpha)?;
            let neg = x.exp()?.sub_(&b.c0(1f32)?)?.mul_(&alpha)?;
            x.gt(&zero)?.select(&x, &neg)?
        }
        UnaryOp::Relu => x.max(&b.c0(0f32)?)?,
        UnaryOp::Silu => x.silu()?,
        UnaryOp::Tanh => x.tanh()?,
        UnaryOp::Sigmoid => x.logistic()?,
    };
    from_f32(&res, dtype)
}

fn build_binary(lhs: &XlaOp, rhs: &XlaOp, op: BinaryOp) -> xla::Result<XlaOp> {
    match op {
        BinaryOp::Add => lhs.add_(rhs),
        BinaryOp::Sub => lhs.sub_(rhs),
        BinaryOp::Mul => lhs.mul_(rhs),
        BinaryOp::Div => lhs.div_(rhs),
        BinaryOp::Maximum => lhs.max(rhs),
        BinaryOp::Minimum => lhs.min(rhs),
    }
}

fn product(dims: &[usize]) -> usize {
    dims.iter().product()
}

impl crate::Backend for Device {
    type Storage<T: WithDType> = Storage<T>;

    fn name(&self) -> String {
        format!("xla-{}", self.platform_name())
    }

    fn synchronize(&self) -> Result<()> {
        lazy::flush_all(self)
    }

    fn storage_len<T: WithDType>(storage: &Self::Storage<T>) -> usize {
        storage.len
    }

    unsafe fn alloc_uninit<T: WithDType>(len: usize, dev: &Self) -> Result<Self::Storage<T>> {
        // Allocation is a lazy zero-fill: it costs nothing until (unless)
        // some operation actually reads the initial contents, and XLA removes
        // it from fused graphs when every element gets overwritten.
        let ty = ety(T::DTYPE);
        let value = lazy_op(dev, "zeros", &[], vec![], len, T::DTYPE, move |b, _p| {
            b.zero(ty)?.broadcast(&[len as i64])
        })?;
        Ok(Storage { value, len, device: dev.clone(), _phantom: Default::default() })
    }

    fn from_vec<T: WithDType>(v: Vec<T>, dev: &Self) -> Result<Self::Storage<T>> {
        let len = v.len();
        let bytes = unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const u8, len * std::mem::size_of::<T>())
        };
        let buf = dev.upload(ety(T::DTYPE), bytes, len)?;
        let value = Value::Buffer { buf: Arc::new(buf), len, ty: ety(T::DTYPE) };
        Ok(Storage { value, len, device: dev.clone(), _phantom: Default::default() })
    }

    fn data<T: WithDType>(src: &Self::Storage<T>, len: usize) -> Result<std::borrow::Cow<'_, [T]>> {
        let buffer = lazy::force(&src.device, &src.value)?;
        let literal = buffer.to_literal_sync().map_err(xerr)?;
        let mut bytes = vec![0u8; literal.size_bytes()];
        literal.copy_untyped_to(&mut bytes).map_err(xerr)?;
        let v = T::vec_from_le_bytes(&bytes[..len * std::mem::size_of::<T>()]);
        Ok(std::borrow::Cow::Owned(v))
    }

    fn fill<T: WithDType>(dst: &mut Self::Storage<T>, elem: T, len: usize) -> Result<()> {
        let dev = dst.device.clone();
        let dst_len = dst.len;
        let inputs = vec![arg(dst), dev.scalar_t(elem)?];
        dst.value =
            lazy_op(&dev, "fill", &[len as i64], inputs, dst_len, T::DTYPE, move |_b, p| {
                let v = p[1].reshape(&[])?.broadcast(&[len as i64])?;
                splice(&p[0], dst_len, len, v)
            })?;
        Ok(())
    }

    fn rand_uniform(dst: &mut Self::Storage<f32>, len: usize, lo: f32, up: f32) -> Result<()> {
        // Generate on the host: XLA's rng op inside a cached executable would
        // replay the exact same values on every execution.
        use rand::Rng;
        let mut rng = rand::rng();
        let v: Vec<f32> = (0..len).map(|_| rng.random::<f32>() * (up - lo) + lo).collect();
        let src = Self::from_vec(v, &dst.device.clone())?;
        Self::copy(dst, &src, len)
    }

    fn randn(dst: &mut Self::Storage<f32>, len: usize, mean: f32, std: f32) -> Result<()> {
        use rand_distr::Distribution;
        let distr = match rand_distr::Normal::<f32>::new(mean, std) {
            Ok(d) => d,
            Err(e) => crate::bail!("failed to create normal distribution for randn: {e}"),
        };
        let mut rng = rand::rng();
        let v: Vec<f32> = (0..len).map(|_| distr.sample(&mut rng)).collect();
        let src = Self::from_vec(v, &dst.device.clone())?;
        Self::copy(dst, &src, len)
    }

    fn copy<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
    ) -> Result<()> {
        let dev = dst.device.clone();
        let dst_len = dst.len;
        let inputs = vec![arg(dst), arg(src)];
        dst.value =
            lazy_op(&dev, "copy", &[len as i64], inputs, dst_len, T::DTYPE, move |_b, p| {
                let v = p[1].slice_in_dim(0, len as i64, 1, 0)?;
                splice(&p[0], dst_len, len, v)
            })?;
        Ok(())
    }

    fn to_dtype<T: WithDType, U: WithDType>(
        dst: &mut Self::Storage<U>,
        src: &Self::Storage<T>,
        len: usize,
    ) -> Result<()> {
        let dev = dst.device.clone();
        let dst_len = dst.len;
        let inputs = vec![arg(dst), arg(src)];
        dst.value =
            lazy_op(&dev, "to_dtype", &[len as i64], inputs, dst_len, U::DTYPE, move |_b, p| {
                let v = p[1].slice_in_dim(0, len as i64, 1, 0)?;
                let v = v.convert(ety(U::DTYPE).primitive_type())?;
                splice(&p[0], dst_len, len, v)
            })?;
        Ok(())
    }

    fn inplace_unary<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        len: usize,
        op: UnaryOp,
    ) -> Result<()> {
        let dev = dst.device.clone();
        let dst_len = dst.len;
        let [k0, k1] = unary_key(op);
        let inputs = vec![arg(dst)];
        dst.value = lazy_op(
            &dev,
            "inplace_unary",
            &[len as i64, k0, k1],
            inputs,
            dst_len,
            T::DTYPE,
            move |_b, p| {
                let v = p[0].slice_in_dim(0, len as i64, 1, 0)?;
                let v = build_unary(&v, op, T::DTYPE)?;
                splice(&p[0], dst_len, len, v)
            },
        )?;
        Ok(())
    }

    fn unary<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
        op: UnaryOp,
    ) -> Result<()> {
        let dev = dst.device.clone();
        let dst_len = dst.len;
        let [k0, k1] = unary_key(op);
        let inputs = vec![arg(dst), arg(src)];
        dst.value = lazy_op(
            &dev,
            "unary",
            &[len as i64, k0, k1],
            inputs,
            dst_len,
            T::DTYPE,
            move |_b, p| {
                let v = p[1].slice_in_dim(0, len as i64, 1, 0)?;
                let v = build_unary(&v, op, T::DTYPE)?;
                splice(&p[0], dst_len, len, v)
            },
        )?;
        Ok(())
    }

    fn bin_assign<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        len: usize,
        op: BinaryOp,
    ) -> Result<()> {
        let dev = dst.device.clone();
        let dst_len = dst.len;
        let inputs = vec![arg(dst), arg(src)];
        dst.value = lazy_op(
            &dev,
            "bin_assign",
            &[len as i64, binary_key(op)],
            inputs,
            dst_len,
            T::DTYPE,
            move |_b, p| {
                let lhs = p[0].slice_in_dim(0, len as i64, 1, 0)?;
                let rhs = p[1].slice_in_dim(0, len as i64, 1, 0)?;
                let v = build_binary(&lhs, &rhs, op)?;
                splice(&p[0], dst_len, len, v)
            },
        )?;
        Ok(())
    }

    fn binary<T: WithDType>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        len: usize,
        op: BinaryOp,
    ) -> Result<()> {
        let dev = dst.device.clone();
        let dst_len = dst.len;
        let inputs = vec![arg(dst), arg(lhs), arg(rhs)];
        dst.value = lazy_op(
            &dev,
            "binary",
            &[len as i64, binary_key(op)],
            inputs,
            dst_len,
            T::DTYPE,
            move |_b, p| {
                let lhs = p[1].slice_in_dim(0, len as i64, 1, 0)?;
                let rhs = p[2].slice_in_dim(0, len as i64, 1, 0)?;
                let v = build_binary(&lhs, &rhs, op)?;
                splice(&p[0], dst_len, len, v)
            },
        )?;
        Ok(())
    }

    fn scale_add<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        scale: T,
        add: T,
        len: usize,
    ) -> Result<()> {
        let dev = dst.device.clone();
        let dst_len = dst.len;
        let inputs = vec![arg(dst), arg(src), dev.scalar_t(scale)?, dev.scalar_t(add)?];
        dst.value =
            lazy_op(&dev, "scale_add", &[len as i64], inputs, dst_len, T::DTYPE, move |_b, p| {
                let src = p[1].slice_in_dim(0, len as i64, 1, 0)?;
                let scale = p[2].reshape(&[])?;
                let add = p[3].reshape(&[])?;
                let v = src.mul_(&scale)?.add_(&add)?;
                splice(&p[0], dst_len, len, v)
            })?;
        Ok(())
    }

    fn transpose<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim1: usize,
        dim2: usize,
        dims: &[usize],
    ) -> Result<()> {
        let dev = dst.device.clone();
        let dst_len = dst.len;
        let n = product(dims);
        let dims_i64: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        let mut key: Vec<i64> = vec![dim1 as i64, dim2 as i64];
        key.extend(dims_i64.iter());
        let inputs = vec![arg(dst), arg(src)];
        dst.value = lazy_op(&dev, "transpose", &key, inputs, dst_len, T::DTYPE, move |_b, p| {
            let v = p[1].slice_in_dim(0, n as i64, 1, 0)?.reshape(&dims_i64)?;
            let v = v.swap_dims(dim1 as i64, dim2 as i64)?;
            let v = v.reshape(&[n as i64])?;
            splice(&p[0], dst_len, n, v)
        })?;
        Ok(())
    }

    fn copy2d<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        d1: usize,
        d2: usize,
        dst_s: usize,
        src_s: usize,
        dst_o: usize,
        src_o: usize,
    ) -> Result<()> {
        let dev = dst.device.clone();
        let dst_len = dst.len;
        let ty = ety(T::DTYPE);
        let key = [d1 as i64, d2 as i64, dst_s as i64, src_s as i64];
        let inputs =
            vec![arg(dst), arg(src), Input::Scalar(dst_o as i64), Input::Scalar(src_o as i64)];
        dst.value = lazy_op(&dev, "copy2d", &key, inputs, dst_len, T::DTYPE, move |b, p| {
            let dims = [d1 as i64, d2 as i64];
            let n = (d1 * d2) as i64;
            let dst_o = p[2].reshape(&[])?;
            let src_o = p[3].reshape(&[])?;
            let src_idx = affine_index(b, &src_o, &dims, &[src_s, 1])?;
            let values = gather_1d(&p[1], &src_idx)?.reshape(&[n])?;
            let dst_idx = affine_index(b, &dst_o, &dims, &[dst_s, 1])?.reshape(&[n])?;
            scatter_1d(&p[0], &dst_idx, &values, ty)
        })?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn rope<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        cos: &Self::Storage<T>,
        sin: &Self::Storage<T>,
        b: usize,
        h: usize,
        t: usize,
        d: usize,
        pos: usize,
        unbatched_rope: bool,
    ) -> Result<()> {
        if dst.len != b * h * t * d || src.len != b * h * t * d {
            crate::bail!("rope unexpected size for src/dst {} {b} {h} {t} {d}", dst.len)
        }
        let dev = dst.device.clone();
        let cs_len = if unbatched_rope { b * t * d / 2 } else { t * d / 2 };
        let key = [b as i64, h as i64, t as i64, d as i64, unbatched_rope as i64];
        let inputs =
            vec![arg(dst), arg(src), arg(cos), arg(sin), Input::Scalar((pos * d / 2) as i64)];
        dst.value = lazy_op(&dev, "rope", &key, inputs, dst.len, T::DTYPE, move |_b, p| {
            let (b, h, t, d) = (b as i64, h as i64, t as i64, d as i64);
            let pos_o = p[4].reshape(&[])?;
            let x = p[1].reshape(&[b, h, t, d])?;
            let x1 = x.slice_in_dim(0, d / 2, 1, 3)?;
            let x2 = x.slice_in_dim(d / 2, d, 1, 3)?;
            let out_dims = [b, h, t, d / 2];
            let cs = |cs_p: &XlaOp| -> xla::Result<XlaOp> {
                let sl = cs_p.dynamic_slice(&[&pos_o], &[cs_len as i64])?;
                if unbatched_rope {
                    sl.reshape(&[b, t, d / 2])?.broadcast_in_dim(&out_dims, &[0, 2, 3])
                } else {
                    sl.reshape(&[t, d / 2])?.broadcast_in_dim(&out_dims, &[2, 3])
                }
            };
            let cos = cs(&p[2])?;
            let sin = cs(&p[3])?;
            let o1 = x1.mul_(&cos)?.sub_(&x2.mul_(&sin)?)?;
            let o2 = x1.mul_(&sin)?.add_(&x2.mul_(&cos)?)?;
            let out = o1.concat_in_dim(&[o2], 3)?;
            out.reshape(&[b * h * t * d])
        })?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn rope_i<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        cos: &Self::Storage<T>,
        sin: &Self::Storage<T>,
        b: usize,
        h: usize,
        t: usize,
        d: usize,
        pos: usize,
        unbatched_rope: bool,
    ) -> Result<()> {
        if dst.len != b * h * t * d || src.len != b * h * t * d {
            crate::bail!("rope-i unexpected size for src/dst {} {b} {h} {t} {d}", dst.len)
        }
        let dev = dst.device.clone();
        let pairs = t * d / 2;
        let cs_len = if unbatched_rope { b * pairs } else { pairs };
        let key = [b as i64, h as i64, t as i64, d as i64, unbatched_rope as i64];
        let inputs =
            vec![arg(dst), arg(src), arg(cos), arg(sin), Input::Scalar((pos * d / 2) as i64)];
        dst.value = lazy_op(&dev, "rope_i", &key, inputs, dst.len, T::DTYPE, move |_b, p| {
            let (b, h) = (b as i64, h as i64);
            let pairs = pairs as i64;
            let pos_o = p[4].reshape(&[])?;
            let x = p[1].reshape(&[b, h, pairs, 2])?;
            let x1 = x.slice_in_dim(0, 1, 1, 3)?.reshape(&[b, h, pairs])?;
            let x2 = x.slice_in_dim(1, 2, 1, 3)?.reshape(&[b, h, pairs])?;
            let out_dims = [b, h, pairs];
            let cs = |cs_p: &XlaOp| -> xla::Result<XlaOp> {
                let sl = cs_p.dynamic_slice(&[&pos_o], &[cs_len as i64])?;
                if unbatched_rope {
                    sl.reshape(&[b, pairs])?.broadcast_in_dim(&out_dims, &[0, 2])
                } else {
                    sl.broadcast_in_dim(&out_dims, &[2])
                }
            };
            let cos = cs(&p[2])?;
            let sin = cs(&p[3])?;
            let o1 = x1.mul_(&cos)?.sub_(&x2.mul_(&sin)?)?.reshape(&[b, h, pairs, 1])?;
            let o2 = x1.mul_(&sin)?.add_(&x2.mul_(&cos)?)?.reshape(&[b, h, pairs, 1])?;
            let out = o1.concat_in_dim(&[o2], 3)?;
            out.reshape(&[b * h * pairs * 2])
        })?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm<T: WithDType>(
        dst: &mut Self::Storage<T>,
        (lhs, lhs_o): (&Self::Storage<T>, usize),
        (rhs, rhs_o): (&Self::Storage<T>, usize),
        m: usize,
        n: usize,
        k: usize,
        lhs_b: usize,
        lhs_b_stride: usize,
        rhs_b_stride: usize,
        (dst_cs, dst_rs): (usize, usize),
        (lhs_cs, lhs_rs): (usize, usize),
        (rhs_cs, rhs_rs): (usize, usize),
    ) -> Result<()> {
        // The destination is written contiguously per batch with element
        // (i, j) at `i * dst_rs + j * dst_cs`, only the row-major and
        // column-major layouts are supported here.
        let dst_row_major = (dst_cs == 1 || n == 1) && (dst_rs == n || m == 1);
        let dst_col_major = dst_cs == m && dst_rs == 1;
        if !dst_row_major && !dst_col_major {
            crate::bail!("xla gemm: unsupported dst strides ({dst_cs}, {dst_rs}) for m={m} n={n}")
        }
        let dev = dst.device.clone();
        let dst_len = dst.len;
        let out_len = lhs_b * m * n;
        let key = [
            m as i64,
            n as i64,
            k as i64,
            lhs_b as i64,
            lhs_b_stride as i64,
            rhs_b_stride as i64,
            lhs_cs as i64,
            lhs_rs as i64,
            rhs_cs as i64,
            rhs_rs as i64,
            dst_row_major as i64,
        ];
        let inputs = vec![
            arg(dst),
            arg(lhs),
            arg(rhs),
            Input::Scalar(lhs_o as i64),
            Input::Scalar(rhs_o as i64),
        ];
        dst.value = lazy_op(&dev, "gemm", &key, inputs, dst_len, T::DTYPE, move |b, p| {
            let (bs, m, n, k) = (lhs_b as i64, m as i64, n as i64, k as i64);
            let lhs_o = p[3].reshape(&[])?;
            let rhs_o = p[4].reshape(&[])?;
            // Contiguous row-major and transposed operands avoid the index
            // gather: XLA then lowers the dot to its native gemm kernels.
            // `d0`/`d1` are the dimensions of the stored (contiguous) matrix.
            let slice_operand =
                |p: &XlaOp, off: &XlaOp, d0: i64, d1: i64, b_stride: usize| -> xla::Result<XlaOp> {
                    let sz = d0 * d1;
                    if bs == 1 || b_stride as i64 == sz {
                        p.dynamic_slice(&[off], &[bs * sz])?.reshape(&[bs, d0, d1])
                    } else {
                        // b_stride == 0: the same matrix is used for every batch.
                        let x = p.dynamic_slice(&[off], &[sz])?.reshape(&[d0, d1])?;
                        x.broadcast_in_dim(&[bs, d0, d1], &[1, 2])
                    }
                };
            // Element (i, j) of the lhs (m x k) lives at `i * lhs_rs + j *
            // lhs_cs`: (cs, rs) == (1, k) is a stored [m, k] matrix while
            // (m, 1) is a stored [k, m] matrix; same reasoning for the rhs.
            let lhs_batch_ok = bs == 1 || lhs_b_stride as i64 == m * k || lhs_b_stride == 0;
            let rhs_batch_ok = bs == 1 || rhs_b_stride as i64 == k * n || rhs_b_stride == 0;
            let lhs = if lhs_batch_ok && (lhs_cs == 1 && lhs_rs as i64 == k) {
                Some((slice_operand(&p[1], &lhs_o, m, k, lhs_b_stride)?, 2i64))
            } else if lhs_batch_ok && (lhs_cs as i64 == m && lhs_rs == 1) {
                Some((slice_operand(&p[1], &lhs_o, k, m, lhs_b_stride)?, 1i64))
            } else {
                None
            };
            let rhs = if rhs_batch_ok && (rhs_cs == 1 && rhs_rs as i64 == n) {
                Some((slice_operand(&p[2], &rhs_o, k, n, rhs_b_stride)?, 1i64))
            } else if rhs_batch_ok && (rhs_cs as i64 == k && rhs_rs == 1) {
                Some((slice_operand(&p[2], &rhs_o, n, k, rhs_b_stride)?, 2i64))
            } else {
                None
            };
            let res = match (lhs, rhs) {
                (Some((lhs, lhs_c)), Some((rhs, rhs_c))) => {
                    lhs.dot_general(&rhs, &[lhs_c], &[rhs_c], &[0], &[0])?
                }
                _ => {
                    // General strided fallback through an index gather.
                    let lhs_idx =
                        affine_index(b, &lhs_o, &[bs, m, k], &[lhs_b_stride, lhs_rs, lhs_cs])?;
                    let rhs_idx =
                        affine_index(b, &rhs_o, &[bs, k, n], &[rhs_b_stride, rhs_rs, rhs_cs])?;
                    let lhs = gather_1d(&p[1], &lhs_idx)?;
                    let rhs = gather_1d(&p[2], &rhs_idx)?;
                    lhs.dot_general(&rhs, &[2], &[1], &[0], &[0])?
                }
            };
            let res = if dst_row_major { res } else { res.transpose(&[0, 2, 1])? };
            let res = res.reshape(&[bs * m * n])?;
            splice(&p[0], dst_len, out_len, res)
        })?;
        Ok(())
    }

    fn index_select<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        ids: &Self::Storage<i64>,
        num_ids: usize,
        dim: usize,
        dims: &[usize],
    ) -> Result<()> {
        let dev = dst.device.clone();
        let dst_len = dst.len;
        let dims_i64: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        let mut out_dims = dims_i64.clone();
        out_dims[dim] = num_ids as i64;
        let out_len: i64 = out_dims.iter().product();
        let mut key: Vec<i64> = vec![num_ids as i64, dim as i64];
        key.extend(dims_i64.iter());
        let inputs = vec![arg(dst), arg(src), arg(ids)];
        dst.value = lazy_op(&dev, "index_select", &key, inputs, dst_len, T::DTYPE, move |b, p| {
            let src_n: i64 = dims_i64.iter().product();
            let src = p[1].slice_in_dim(0, src_n, 1, 0)?.reshape(&dims_i64)?;
            let ids = p[2].slice_in_dim(0, num_ids as i64, 1, 0)?;
            // An index of -1 selects zeros rather than a row of the source.
            let zero = b.c0(0i64)?;
            let clamped = ids.max(&zero)?;
            let sel = src.take(&clamped, dim as i64)?;
            let mask = ids.ge(&zero)?.broadcast_in_dim(&out_dims, &[dim as i64])?;
            let zeros = sel.zeros_like()?;
            let v = mask.select(&sel, &zeros)?.reshape(&[out_len])?;
            splice(&p[0], dst_len, out_len as usize, v)
        })?;
        Ok(())
    }

    fn apply_causality_mask<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        bh: usize,
        t1: usize,
        t2: usize,
        offset: usize,
    ) -> Result<()> {
        let dev = dst.device.clone();
        let dst_len = dst.len;
        let n = bh * t1 * t2;
        let key = [bh as i64, t1 as i64, t2 as i64];
        let inputs = vec![arg(dst), Input::Scalar(offset as i64)];
        dst.value =
            lazy_op(&dev, "causality_mask", &key, inputs, dst_len, T::DTYPE, move |b, p| {
                let (bh, t1, t2) = (bh as i64, t1 as i64, t2 as i64);
                let x = p[0].slice_in_dim(0, n as i64, 1, 0)?.reshape(&[bh, t1, t2])?;
                let offset = p[1].reshape(&[])?;
                let i1 = b.iota(ElementType::S64, &[t1, t2], 0)?;
                let i2 = b.iota(ElementType::S64, &[t1, t2], 1)?;
                let mask = i2.gt(&i1.add_(&offset)?)?.broadcast_in_dim(&[bh, t1, t2], &[1, 2])?;
                let neg_inf =
                    from_f32(&b.c0(f32::NEG_INFINITY)?, T::DTYPE)?.broadcast(&[bh, t1, t2])?;
                let v = mask.select(&neg_inf, &x)?.reshape(&[bh * t1 * t2])?;
                splice(&p[0], dst_len, n, v)
            })?;
        Ok(())
    }

    fn softmax<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_m1: usize,
        d: usize,
    ) -> Result<()> {
        let dev = dst.device.clone();
        let dst_len = dst.len;
        let n = d * dim_m1;
        let key = [dim_m1 as i64, d as i64];
        let inputs = vec![arg(dst), arg(src)];
        dst.value = lazy_op(&dev, "softmax", &key, inputs, dst_len, T::DTYPE, move |_b, p| {
            let (d, dim_m1) = (d as i64, dim_m1 as i64);
            let x = p[1].slice_in_dim(0, n as i64, 1, 0)?.reshape(&[d, dim_m1])?;
            let x = to_f32(&x)?;
            let max = x.reduce_max(&[1], true)?;
            let e = x.sub_(&max)?.exp()?;
            let s = e.reduce_sum(&[1], true)?;
            let v = from_f32(&e.div_(&s)?, T::DTYPE)?.reshape(&[n as i64])?;
            splice(&p[0], dst_len, n, v)
        })?;
        Ok(())
    }

    fn rms_norm<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        alpha: &Self::Storage<T>,
        dim_m1: usize,
        d: usize,
        eps: f32,
    ) -> Result<()> {
        let dev = dst.device.clone();
        let dst_len = dst.len;
        let n = d * dim_m1;
        let key = [dim_m1 as i64, d as i64, eps.to_bits() as i64];
        let inputs = vec![arg(dst), arg(src), arg(alpha)];
        dst.value = lazy_op(&dev, "rms_norm", &key, inputs, dst_len, T::DTYPE, move |b, p| {
            let (d, dim_m1) = (d as i64, dim_m1 as i64);
            let x = p[1].slice_in_dim(0, n as i64, 1, 0)?.reshape(&[d, dim_m1])?;
            let x = to_f32(&x)?;
            let alpha = to_f32(&p[2].slice_in_dim(0, dim_m1, 1, 0)?)?;
            let sum2 = x.mul_(&x)?.reduce_sum(&[1], true)?;
            let m = sum2.div_(&b.c0(dim_m1 as f32)?)?.add_(&b.c0(eps)?)?.rsqrt()?;
            let alpha = alpha.broadcast_in_dim(&[d, dim_m1], &[1])?;
            let v = x.mul_(&m)?.mul_(&alpha)?;
            let v = from_f32(&v, T::DTYPE)?.reshape(&[n as i64])?;
            splice(&p[0], dst_len, n, v)
        })?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn layer_norm<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        weight: &Self::Storage<T>,
        bias: &Self::Storage<T>,
        dim_m1: usize,
        d: usize,
        eps: f32,
        remove_mean: bool,
    ) -> Result<()> {
        let dev = dst.device.clone();
        let dst_len = dst.len;
        let n = d * dim_m1;
        let key = [dim_m1 as i64, d as i64, eps.to_bits() as i64, remove_mean as i64];
        let inputs = vec![arg(dst), arg(src), arg(weight), arg(bias)];
        dst.value = lazy_op(&dev, "layer_norm", &key, inputs, dst_len, T::DTYPE, move |b, p| {
            let (d, dim_m1) = (d as i64, dim_m1 as i64);
            let x = p[1].slice_in_dim(0, n as i64, 1, 0)?.reshape(&[d, dim_m1])?;
            let x = to_f32(&x)?;
            let weight = to_f32(&p[2].slice_in_dim(0, dim_m1, 1, 0)?)?
                .broadcast_in_dim(&[d, dim_m1], &[1])?;
            let bias = to_f32(&p[3].slice_in_dim(0, dim_m1, 1, 0)?)?
                .broadcast_in_dim(&[d, dim_m1], &[1])?;
            let dim_f = b.c0(dim_m1 as f32)?;
            let mean = x.reduce_sum(&[1], true)?.div_(&dim_f)?;
            let centered = x.sub_(&mean)?;
            let var = centered.mul_(&centered)?.reduce_sum(&[1], true)?.div_(&dim_f)?;
            let inv_std = var.add_(&b.c0(eps)?)?.rsqrt()?;
            let normalized = if remove_mean { centered.clone() } else { x.clone() };
            let v = normalized.mul_(&inv_std)?.mul_(&weight)?.add_(&bias)?;
            let v = from_f32(&v, T::DTYPE)?.reshape(&[n as i64])?;
            splice(&p[0], dst_len, n, v)
        })?;
        Ok(())
    }

    fn reduce_max<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        reduce_impl::<T>(dst, src, dim_size, outer_size, inner_size, Reduce::Max)
    }

    fn reduce_min<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        reduce_impl::<T>(dst, src, dim_size, outer_size, inner_size, Reduce::Min)
    }

    fn reduce_sum<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        reduce_impl::<T>(dst, src, dim_size, outer_size, inner_size, Reduce::Sum)
    }

    fn reduce_argmax<T: WithDTypeF>(
        dst: &mut Self::Storage<i64>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        reduce_arg_impl::<T>(dst, src, dim_size, outer_size, inner_size, false)
    }

    fn reduce_argmin<T: WithDTypeF>(
        dst: &mut Self::Storage<i64>,
        src: &Self::Storage<T>,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<()> {
        reduce_arg_impl::<T>(dst, src, dim_size, outer_size, inner_size, true)
    }

    fn copy_strided<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        src_offset: usize,
        dims: &[usize],
        src_strides: &[usize],
    ) -> Result<()> {
        let dev = dst.device.clone();
        let dst_len = dst.len;
        let n = product(dims);
        let dims_i64: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        let strides = src_strides.to_vec();
        let mut key: Vec<i64> = dims_i64.clone();
        key.extend(strides.iter().map(|&s| s as i64));
        let inputs = vec![arg(dst), arg(src), Input::Scalar(src_offset as i64)];
        dst.value = lazy_op(&dev, "copy_strided", &key, inputs, dst_len, T::DTYPE, move |b, p| {
            let src_o = p[2].reshape(&[])?;
            let idx = affine_index(b, &src_o, &dims_i64, &strides)?;
            let v = gather_1d(&p[1], &idx)?.reshape(&[n as i64])?;
            splice(&p[0], dst_len, n, v)
        })?;
        Ok(())
    }

    fn scatter_set<T: WithDType>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        ids: &Self::Storage<i64>,
        dim: usize,
        dst_dims: &[usize],
        src_dims: &[usize],
    ) -> Result<()> {
        let left: usize = src_dims[..dim].iter().product();
        let right: usize = src_dims[dim + 1..].iter().product();
        let src_dim = src_dims[dim];
        let dst_dim = dst_dims[dim];
        let n = left * src_dim * right;
        let dev = dst.device.clone();
        let ty = ety(T::DTYPE);
        let key = [left as i64, src_dim as i64, right as i64, dst_dim as i64];
        let inputs = vec![arg(dst), arg(src), arg(ids)];
        dst.value = lazy_op(&dev, "scatter_set", &key, inputs, dst.len, T::DTYPE, move |b, p| {
            let dims = [left as i64, src_dim as i64, right as i64];
            // Flat destination index for source element (l, i, r):
            // l * dst_dim * right + ids[l, i, r] * right + r.
            let zero = b.c0(0i64)?;
            let base = affine_index(b, &zero, &dims, &[dst_dim * right, 0, 1])?;
            let ids = p[2].slice_in_dim(0, n as i64, 1, 0)?.reshape(&dims)?;
            let idx = base.add_(&ids.mul_(&b.c0(right as i64)?)?)?.reshape(&[n as i64])?;
            let updates = p[1].slice_in_dim(0, n as i64, 1, 0)?;
            scatter_1d(&p[0], &idx, &updates, ty)
        })?;
        Ok(())
    }

    fn broadcast_binary<T: WithDType>(
        dst: &mut Self::Storage<T>,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        dst_shape: &[usize],
        lhs_strides: &[usize],
        rhs_strides: &[usize],
        op: BinaryOp,
    ) -> Result<()> {
        let dev = dst.device.clone();
        let dst_len = dst.len;
        let n = product(dst_shape);
        let dims_i64: Vec<i64> = dst_shape.iter().map(|&d| d as i64).collect();
        let lhs_strides = lhs_strides.to_vec();
        let rhs_strides = rhs_strides.to_vec();
        let mut key: Vec<i64> = vec![binary_key(op)];
        key.extend(dims_i64.iter());
        key.extend(lhs_strides.iter().map(|&s| s as i64));
        key.extend(rhs_strides.iter().map(|&s| s as i64));
        let inputs = vec![arg(dst), arg(lhs), arg(rhs)];
        dst.value =
            lazy_op(&dev, "broadcast_binary", &key, inputs, dst_len, T::DTYPE, move |b, p| {
                let zero = b.c0(0i64)?;
                let lhs_idx = affine_index(b, &zero, &dims_i64, &lhs_strides)?;
                let rhs_idx = affine_index(b, &zero, &dims_i64, &rhs_strides)?;
                let lhs = gather_1d(&p[1], &lhs_idx)?;
                let rhs = gather_1d(&p[2], &rhs_idx)?;
                let v = build_binary(&lhs, &rhs, op)?.reshape(&[n as i64])?;
                splice(&p[0], dst_len, n, v)
            })?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn conv1d<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        kernel: &Self::Storage<T>,
        batch: usize,
        in_channels: usize,
        out_channels: usize,
        length: usize,
        out_length: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<()> {
        let dev = dst.device.clone();
        let dst_len = dst.len;
        let n = batch * out_channels * out_length;
        let key = [
            batch as i64,
            in_channels as i64,
            out_channels as i64,
            length as i64,
            kernel_size as i64,
            stride as i64,
            padding as i64,
            dilation as i64,
            groups as i64,
        ];
        let inputs = vec![arg(dst), arg(src), arg(kernel)];
        dst.value = lazy_op(&dev, "conv1d", &key, inputs, dst_len, T::DTYPE, move |_b, p| {
            let src_n = (batch * in_channels * length) as i64;
            let src = p[1].slice_in_dim(0, src_n, 1, 0)?.reshape(&[
                batch as i64,
                in_channels as i64,
                length as i64,
            ])?;
            let kernel_n = (out_channels * (in_channels / groups) * kernel_size) as i64;
            let kernel = p[2].slice_in_dim(0, kernel_n, 1, 0)?.reshape(&[
                out_channels as i64,
                (in_channels / groups) as i64,
                kernel_size as i64,
            ])?;
            let v =
                src.conv1d(&kernel, stride as i64, padding as i64, dilation as i64, groups as i64)?;
            let v = v.reshape(&[n as i64])?;
            splice(&p[0], dst_len, n, v)
        })?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn conv_transpose1d<T: WithDTypeF>(
        dst: &mut Self::Storage<T>,
        src: &Self::Storage<T>,
        kernel: &Self::Storage<T>,
        batch: usize,
        in_channels: usize,
        out_channels: usize,
        length: usize,
        out_length: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        groups: usize,
    ) -> Result<()> {
        let dev = dst.device.clone();
        let dst_len = dst.len;
        let n = batch * out_channels * out_length;
        let key = [
            batch as i64,
            in_channels as i64,
            out_channels as i64,
            length as i64,
            kernel_size as i64,
            stride as i64,
            padding as i64,
            output_padding as i64,
            groups as i64,
        ];
        let inputs = vec![arg(dst), arg(src), arg(kernel)];
        dst.value =
            lazy_op(&dev, "conv_transpose1d", &key, inputs, dst_len, T::DTYPE, move |_b, p| {
                let src_n = (batch * in_channels * length) as i64;
                let src = p[1].slice_in_dim(0, src_n, 1, 0)?.reshape(&[
                    batch as i64,
                    in_channels as i64,
                    length as i64,
                ])?;
                let kernel_n = (in_channels * (out_channels / groups) * kernel_size) as i64;
                let kernel = p[2].slice_in_dim(0, kernel_n, 1, 0)?.reshape(&[
                    in_channels as i64,
                    (out_channels / groups) as i64,
                    kernel_size as i64,
                ])?;
                let v = src.conv_transpose1d(
                    &kernel,
                    stride as i64,
                    padding as i64,
                    output_padding as i64,
                    /* dilation= */ 1,
                    groups as i64,
                )?;
                let v = v.reshape(&[n as i64])?;
                splice(&p[0], dst_len, n, v)
            })?;
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum Reduce {
    Max,
    Min,
    Sum,
}

fn reduce_impl<T: WithDTypeF>(
    dst: &mut Storage<T>,
    src: &Storage<T>,
    dim_size: usize,
    outer_size: usize,
    inner_size: usize,
    red: Reduce,
) -> Result<()> {
    let dev = dst.device.clone();
    let dst_len = dst.len;
    let n_in = outer_size * dim_size * inner_size;
    let n_out = outer_size * inner_size;
    let (name, red_key) = match red {
        Reduce::Max => ("reduce_max", 0i64),
        Reduce::Min => ("reduce_min", 1),
        Reduce::Sum => ("reduce_sum", 2),
    };
    let key = [dim_size as i64, outer_size as i64, inner_size as i64, red_key];
    let inputs = vec![arg(dst), arg(src)];
    dst.value = lazy_op(&dev, name, &key, inputs, dst_len, T::DTYPE, move |_b, p| {
        let dims = [outer_size as i64, dim_size as i64, inner_size as i64];
        let x = p[1].slice_in_dim(0, n_in as i64, 1, 0)?.reshape(&dims)?;
        let v = match red {
            Reduce::Max => x.reduce_max(&[1], false)?,
            Reduce::Min => x.reduce_min(&[1], false)?,
            Reduce::Sum => x.reduce_sum(&[1], false)?,
        };
        let v = v.reshape(&[n_out as i64])?;
        splice(&p[0], dst_len, n_out, v)
    })?;
    Ok(())
}

fn reduce_arg_impl<T: WithDTypeF>(
    dst: &mut Storage<i64>,
    src: &Storage<T>,
    dim_size: usize,
    outer_size: usize,
    inner_size: usize,
    is_min: bool,
) -> Result<()> {
    let dev = dst.device.clone();
    let dst_len = dst.len;
    let n_in = outer_size * dim_size * inner_size;
    let n_out = outer_size * inner_size;
    let name = if is_min { "reduce_argmin" } else { "reduce_argmax" };
    let key = [dim_size as i64, outer_size as i64, inner_size as i64, dtype_key(T::DTYPE)];
    let inputs = vec![arg(dst), arg(src)];
    dst.value = lazy_op(&dev, name, &key, inputs, dst_len, DType::I64, move |_b, p| {
        let dims = [outer_size as i64, dim_size as i64, inner_size as i64];
        let x = p[1].slice_in_dim(0, n_in as i64, 1, 0)?.reshape(&dims)?;
        let v =
            if is_min { x.argmin(ElementType::S64, 1)? } else { x.argmax(ElementType::S64, 1)? };
        let v = v.reshape(&[n_out as i64])?;
        splice(&p[0], dst_len, n_out, v)
    })?;
    Ok(())
}

//! Lazy graph capture for the XLA backend.
//!
//! Backend operations do not execute immediately: each one appends a
//! [`LazyNode`] holding the op descriptor and a build closure producing its
//! XLA subgraph. Nodes form a DAG through their inputs (`Value::Node` edges),
//! with materialized buffers as leaves. When a value is read on the host (or
//! `synchronize` is called), the pending subgraph is flushed.
//!
//! Flushing hashes the graph structure into a key and then follows a hybrid
//! policy: the first time a given structure is seen it is executed node by
//! node (reusing per-op executables, so shape-churning workloads never pay
//! for whole-graph compilation); when the same structure comes back, the
//! whole graph is compiled into a single fused executable and cached, so
//! steady-state loops replay one XLA program per step. Values that change
//! per step without affecting shapes (offsets, positions) are passed as
//! `Input::Scalar` and packed into a single side buffer at execution time,
//! keeping them out of the structural key.
use super::{Device, xerr};
use crate::Result;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, Weak};
use xla::{ElementType, XlaBuilder, XlaOp};

pub(super) type BuildFn = Box<dyn Fn(&XlaBuilder, &[XlaOp]) -> xla::Result<XlaOp> + Send + Sync>;

/// The contents of a storage: either an actual device buffer or a node of
/// the pending graph.
#[derive(Clone)]
pub(super) enum Value {
    Buffer { buf: Arc<xla::PjRtBuffer>, len: usize, ty: ElementType },
    Node(Arc<LazyNode>),
}

impl Value {
    pub fn len(&self) -> usize {
        match self {
            Value::Buffer { len, .. } => *len,
            Value::Node(n) => n.out_len,
        }
    }

    pub fn ty(&self) -> ElementType {
        match self {
            Value::Buffer { ty, .. } => *ty,
            Value::Node(n) => n.out_ty,
        }
    }
}

/// An input of a lazy node.
pub(super) enum Input {
    Value(Value),
    /// A runtime scalar (offset, position). Packed into a single S64 side
    /// buffer at execution time so that it does not affect the graph key.
    Scalar(i64),
}

pub(super) struct LazyNode {
    pub op: &'static str,
    /// Structural key: shape parameters, dtypes and input lens/types.
    pub key: Vec<i64>,
    pub out_len: usize,
    pub out_ty: ElementType,
    pub state: Mutex<NodeState>,
}

pub(super) enum NodeState {
    Pending {
        inputs: Vec<Input>,
        build: BuildFn,
    },
    Done(Arc<xla::PjRtBuffer>),
    /// Transient state while a flush is executing this node.
    Taken,
}

impl LazyNode {
    pub fn done_buffer(&self) -> Option<Arc<xla::PjRtBuffer>> {
        match &*self.state.lock().unwrap() {
            NodeState::Done(b) => Some(b.clone()),
            _ => None,
        }
    }

    fn is_pending(&self) -> bool {
        matches!(&*self.state.lock().unwrap(), NodeState::Pending { .. })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum FusePolicy {
    /// Compile a fused executable the first time a graph structure is seen.
    Always,
    /// Never fuse, always execute node by node.
    Never,
    /// Execute node by node on first sight, fuse from the second occurrence.
    OnRepeat,
}

impl FusePolicy {
    pub fn from_env() -> Self {
        match std::env::var("XN_XLA_FUSE").as_deref() {
            Ok("always") => Self::Always,
            Ok("never") => Self::Never,
            _ => Self::OnRepeat,
        }
    }
}

/// Force a value to a materialized buffer, flushing the pending graph if
/// needed.
pub(super) fn force(dev: &Device, value: &Value) -> Result<Arc<xla::PjRtBuffer>> {
    match value {
        Value::Buffer { buf, .. } => Ok(buf.clone()),
        Value::Node(node) => {
            if let Some(b) = node.done_buffer() {
                return Ok(b);
            }
            flush(dev, std::slice::from_ref(node))?;
            match node.done_buffer() {
                Some(b) => Ok(b),
                None => crate::bail!("xla: node {} not materialized after flush", node.op),
            }
        }
    }
}

/// Flush every pending node currently registered on the device.
pub(super) fn flush_all(dev: &Device) -> Result<()> {
    let pending: Vec<Weak<LazyNode>> = std::mem::take(&mut *dev.inner.pending.lock().unwrap());
    let roots: Vec<Arc<LazyNode>> =
        pending.iter().filter_map(Weak::upgrade).filter(|n| n.is_pending()).collect();
    if roots.is_empty() { Ok(()) } else { flush(dev, &roots) }
}

fn hash_str(s: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

/// How a node input maps onto the fused computation.
enum Slot {
    /// Leaf buffer, parameter index.
    Param(usize),
    /// Output of another pending node, topological index.
    Node(usize),
    /// Runtime scalar, index in the packed scalar buffer.
    Scalar(usize),
}

pub(super) fn flush(dev: &Device, roots: &[Arc<LazyNode>]) -> Result<()> {
    let inner = &dev.inner;
    let _flush_guard = inner.flush_lock.lock().unwrap();

    // Topological order (inputs before consumers) of the reachable pending
    // nodes; iterative so deep chains cannot overflow the stack.
    let mut topo: Vec<Arc<LazyNode>> = Vec::new();
    let mut index: HashMap<*const LazyNode, usize> = HashMap::new();
    {
        enum Frame {
            Enter(Arc<LazyNode>),
            Exit(Arc<LazyNode>),
        }
        let mut stack: Vec<Frame> = roots.iter().rev().map(|n| Frame::Enter(n.clone())).collect();
        let mut entered: HashSet<*const LazyNode> = HashSet::new();
        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Enter(n) => {
                    let ptr = Arc::as_ptr(&n);
                    if entered.contains(&ptr) {
                        continue;
                    }
                    entered.insert(ptr);
                    let children: Vec<Arc<LazyNode>> = match &*n.state.lock().unwrap() {
                        NodeState::Pending { inputs, .. } => inputs
                            .iter()
                            .filter_map(|i| match i {
                                Input::Value(Value::Node(c)) if c.is_pending() => Some(c.clone()),
                                _ => None,
                            })
                            .collect(),
                        _ => continue,
                    };
                    stack.push(Frame::Exit(n));
                    for c in children {
                        if !entered.contains(&Arc::as_ptr(&c)) {
                            stack.push(Frame::Enter(c));
                        }
                    }
                }
                Frame::Exit(n) => {
                    index.insert(Arc::as_ptr(&n), topo.len());
                    topo.push(n);
                }
            }
        }
    }
    if topo.is_empty() {
        return Ok(());
    }

    // Count graph-internal references to decide which nodes must become
    // outputs: a node referenced from outside the flushed graph (a live
    // storage, or a pending node not part of this flush) has to be
    // materialized; nodes only feeding other nodes of this graph are fused
    // away.
    let mut internal: HashMap<*const LazyNode, usize> = HashMap::new();
    for n in topo.iter() {
        if let NodeState::Pending { inputs, .. } = &*n.state.lock().unwrap() {
            for i in inputs.iter() {
                if let Input::Value(Value::Node(c)) = i {
                    let ptr = Arc::as_ptr(c);
                    if index.contains_key(&ptr) {
                        *internal.entry(ptr).or_insert(0) += 1;
                    }
                }
            }
        }
    }
    let mut outputs: Vec<usize> = Vec::new();
    for (i, n) in topo.iter().enumerate() {
        let ptr = Arc::as_ptr(n);
        // +1 accounts for the handle held by `topo` itself. This is
        // conservative under concurrent clones, which can only add outputs.
        let external = Arc::strong_count(n) > internal.get(&ptr).copied().unwrap_or(0) + 1;
        if external {
            outputs.push(i);
        }
    }

    // Structural key and input slot assignment.
    let mut key: Vec<u64> = Vec::new();
    let mut params: Vec<(Arc<xla::PjRtBuffer>, usize, ElementType)> = Vec::new();
    let mut param_slots: HashMap<*const xla::PjRtBuffer, usize> = HashMap::new();
    let mut scalars: Vec<i64> = Vec::new();
    let mut slots: Vec<Vec<Slot>> = Vec::with_capacity(topo.len());
    let param_slot = |params: &mut Vec<(Arc<xla::PjRtBuffer>, usize, ElementType)>,
                      param_slots: &mut HashMap<*const xla::PjRtBuffer, usize>,
                      buf: &Arc<xla::PjRtBuffer>,
                      len: usize,
                      ty: ElementType|
     -> usize {
        let ptr = Arc::as_ptr(buf);
        match param_slots.get(&ptr) {
            Some(&slot) => slot,
            None => {
                let slot = params.len();
                params.push((buf.clone(), len, ty));
                param_slots.insert(ptr, slot);
                slot
            }
        }
    };
    for n in topo.iter() {
        key.push(hash_str(n.op));
        key.extend(n.key.iter().map(|&v| v as u64));
        let mut node_slots = Vec::new();
        match &*n.state.lock().unwrap() {
            NodeState::Pending { inputs, .. } => {
                for i in inputs.iter() {
                    let slot = match i {
                        Input::Scalar(v) => {
                            scalars.push(*v);
                            Slot::Scalar(scalars.len() - 1)
                        }
                        Input::Value(Value::Buffer { buf, len, ty }) => {
                            Slot::Param(param_slot(&mut params, &mut param_slots, buf, *len, *ty))
                        }
                        Input::Value(Value::Node(c)) => match index.get(&Arc::as_ptr(c)) {
                            Some(&ci) => Slot::Node(ci),
                            None => {
                                // A node materialized by an earlier flush:
                                // its buffer is a leaf.
                                let buf = match c.done_buffer() {
                                    Some(b) => b,
                                    None => crate::bail!("xla: unmaterialized input {}", c.op),
                                };
                                Slot::Param(param_slot(
                                    &mut params,
                                    &mut param_slots,
                                    &buf,
                                    c.out_len,
                                    c.out_ty,
                                ))
                            }
                        },
                    };
                    match &slot {
                        Slot::Param(i) => key.extend([1, *i as u64]),
                        Slot::Node(i) => key.extend([2, *i as u64]),
                        Slot::Scalar(i) => key.extend([3, *i as u64]),
                    }
                    node_slots.push(slot);
                }
            }
            _ => crate::bail!("xla: node {} changed state during flush", n.op),
        }
        key.push(u64::MAX); // node separator
        slots.push(node_slots);
    }
    for &o in outputs.iter() {
        key.extend([4, o as u64]);
    }

    // Hybrid policy: run node by node the first time a structure shows up,
    // compile the fused graph when it repeats.
    let cached = inner.graph_cache.lock().unwrap().get(&key).cloned();
    let exe = match cached {
        Some(exe) => Some(exe),
        None => {
            let fuse = match inner.fuse_policy {
                FusePolicy::Always => true,
                FusePolicy::Never => false,
                FusePolicy::OnRepeat => !inner.seen_graphs.lock().unwrap().insert(key.clone()),
            };
            if fuse {
                let exe = compile_fused(dev, &topo, &slots, &outputs, &params, scalars.len())?;
                inner.graph_cache.lock().unwrap().insert(key, exe.clone());
                Some(exe)
            } else {
                None
            }
        }
    };

    match exe {
        Some(exe) => execute_fused(dev, &exe, &topo, &outputs, &params, &scalars),
        None => execute_eager(dev, &topo),
    }
}

/// Build and compile the whole pending graph as a single computation whose
/// root is the tuple of output nodes.
fn compile_fused(
    dev: &Device,
    topo: &[Arc<LazyNode>],
    slots: &[Vec<Slot>],
    outputs: &[usize],
    params: &[(Arc<xla::PjRtBuffer>, usize, ElementType)],
    n_scalars: usize,
) -> Result<Arc<xla::PjRtLoadedExecutable>> {
    let b = XlaBuilder::new("fused");
    let mut xparams = Vec::with_capacity(params.len());
    for (i, (_, len, ty)) in params.iter().enumerate() {
        let p = b.parameter(i as i64, *ty, &[*len as i64], &format!("p{i}")).map_err(xerr)?;
        xparams.push(p);
    }
    let scalar_param = if n_scalars > 0 {
        let p = b
            .parameter(params.len() as i64, ElementType::S64, &[n_scalars as i64], "scalars")
            .map_err(xerr)?;
        Some(p)
    } else {
        None
    };
    let mut built: Vec<XlaOp> = Vec::with_capacity(topo.len());
    for (n, node_slots) in topo.iter().zip(slots.iter()) {
        let mut inputs = Vec::with_capacity(node_slots.len());
        for slot in node_slots.iter() {
            let op = match slot {
                Slot::Param(i) => xparams[*i].clone(),
                Slot::Node(i) => built[*i].clone(),
                Slot::Scalar(i) => {
                    let p = scalar_param.as_ref().expect("scalar param");
                    // Shape [1], matching what the build closures expect.
                    p.slice_in_dim(*i as i64, *i as i64 + 1, 1, 0).map_err(xerr)?
                }
            };
            inputs.push(op);
        }
        let state = n.state.lock().unwrap();
        let NodeState::Pending { build, .. } = &*state else {
            crate::bail!("xla: node {} changed state during fusion", n.op)
        };
        let op = build(&b, &inputs).map_err(xerr)?;
        drop(state);
        built.push(op);
    }
    let outs: Vec<&XlaOp> = outputs.iter().map(|&o| &built[o]).collect();
    let root = b.tuple(&outs).map_err(xerr)?;
    let comp = b.build(&root).map_err(xerr)?;
    let exe = dev.inner.client.compile(&comp).map_err(xerr)?;
    Ok(Arc::new(exe))
}

fn execute_fused(
    dev: &Device,
    exe: &xla::PjRtLoadedExecutable,
    topo: &[Arc<LazyNode>],
    outputs: &[usize],
    params: &[(Arc<xla::PjRtBuffer>, usize, ElementType)],
    scalars: &[i64],
) -> Result<()> {
    let scalar_buf = if scalars.is_empty() {
        None
    } else {
        let bytes: Vec<u8> = scalars.iter().flat_map(|v| v.to_le_bytes()).collect();
        Some(dev.upload(ElementType::S64, &bytes, scalars.len())?)
    };
    let mut args: Vec<&xla::PjRtBuffer> = params.iter().map(|(b, _, _)| b.as_ref()).collect();
    if let Some(sb) = scalar_buf.as_ref() {
        args.push(sb);
    }
    let outs = exe.execute_b(&args).map_err(xerr)?;
    let bufs: Vec<xla::PjRtBuffer> = outs.into_iter().flatten().collect();
    if bufs.len() != outputs.len() {
        crate::bail!(
            "xla: fused execution returned {} buffers, expected {}",
            bufs.len(),
            outputs.len()
        )
    }
    for (&o, buf) in outputs.iter().zip(bufs) {
        *topo[o].state.lock().unwrap() = NodeState::Done(Arc::new(buf));
    }
    Ok(())
}

/// Execute the pending nodes one by one, compiling (or reusing) a small
/// executable per node. This is the first-sight path of the hybrid policy:
/// per-node executables are shared across steps even when the overall graph
/// shape keeps changing.
fn execute_eager(dev: &Device, topo: &[Arc<LazyNode>]) -> Result<()> {
    for n in topo.iter() {
        let state = std::mem::replace(&mut *n.state.lock().unwrap(), NodeState::Taken);
        let NodeState::Pending { inputs, build } = state else {
            crate::bail!("xla: node {} changed state during eager flush", n.op)
        };
        let mut args: Vec<Arc<xla::PjRtBuffer>> = Vec::with_capacity(inputs.len());
        let mut meta: Vec<(usize, ElementType)> = Vec::with_capacity(inputs.len());
        for i in inputs.iter() {
            match i {
                Input::Scalar(v) => {
                    args.push(Arc::new(dev.upload(ElementType::S64, &v.to_le_bytes(), 1)?));
                    meta.push((1, ElementType::S64));
                }
                Input::Value(Value::Buffer { buf, len, ty }) => {
                    args.push(buf.clone());
                    meta.push((*len, *ty));
                }
                Input::Value(Value::Node(c)) => {
                    let buf = match c.done_buffer() {
                        Some(b) => b,
                        None => crate::bail!("xla: unmaterialized input {} of {}", c.op, n.op),
                    };
                    args.push(buf);
                    meta.push((c.out_len, c.out_ty));
                }
            }
        }
        let cache_key = (n.op, n.key.clone());
        let cached = dev.inner.node_cache.lock().unwrap().get(&cache_key).cloned();
        let exe = match cached {
            Some(exe) => exe,
            None => {
                let b = XlaBuilder::new(n.op);
                let mut xops = Vec::with_capacity(meta.len());
                for (i, (len, ty)) in meta.iter().enumerate() {
                    let p = b
                        .parameter(i as i64, *ty, &[*len as i64], &format!("p{i}"))
                        .map_err(xerr)?;
                    xops.push(p);
                }
                let root = build(&b, &xops).map_err(xerr)?;
                let comp = b.build(&root).map_err(xerr)?;
                let exe = Arc::new(dev.inner.client.compile(&comp).map_err(xerr)?);
                dev.inner.node_cache.lock().unwrap().insert(cache_key, exe.clone());
                exe
            }
        };
        let arg_refs: Vec<&xla::PjRtBuffer> = args.iter().map(|a| a.as_ref()).collect();
        let mut outs = exe.execute_b(&arg_refs).map_err(xerr)?;
        let buf = match outs.pop().and_then(|mut r| r.pop()) {
            Some(b) => b,
            None => crate::bail!("xla: execution of {} returned no buffer", n.op),
        };
        *n.state.lock().unwrap() = NodeState::Done(Arc::new(buf));
    }
    Ok(())
}

use crate::{Backend, Result, Shape, Tensor, TensorView, WithDTypeF, shape::Dim};

pub struct Cache<T: WithDTypeF, B: Backend> {
    all_data: Tensor<T, B>,
    dim: usize,
    current_seq_len: usize,
    max_seq_len: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: WithDTypeF, B: Backend> Cache<T, B> {
    pub fn new<S: Into<Shape>, D: Dim>(dim: D, shape: S, dev: &B) -> Result<Self> {
        let shape = shape.into();
        let dim = dim.to_index(&shape, "kv-cache")?;
        let max_seq_len = shape.dims()[dim];
        let all_data = Tensor::zeros(shape, dev)?;
        Ok(Self { all_data, dim, current_seq_len: 0, max_seq_len, _phantom: Default::default() })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn all_data(&self) -> &Tensor<T, B> {
        &self.all_data
    }

    pub fn current_data(&self) -> Result<TensorView<T, B>> {
        let view = TensorView::from(&self.all_data);
        view.narrow(self.dim, ..self.current_seq_len)
    }

    pub fn append(&mut self, src: &Tensor<T, B>) -> Result<()> {
        let seq_len = src.dim(self.dim)?;
        if self.current_seq_len + seq_len > self.max_seq_len {
            crate::bail!(
                "kv-cache: above max-seq-len {}+{seq_len}>{}",
                self.current_seq_len,
                self.max_seq_len
            )
        }
        self.all_data.slice_assign(src, self.dim, self.current_seq_len)?;
        self.current_seq_len += seq_len;
        Ok(())
    }
}

pub struct KvCache<T: WithDTypeF, B: Backend> {
    k: Cache<T, B>,
    v: Cache<T, B>,
}

impl<T: WithDTypeF, B: Backend> KvCache<T, B> {
    pub fn new<S: Into<Shape>, D: Dim>(dim: D, shape: S, dev: &B) -> Result<Self> {
        let shape = shape.into();
        let dim = dim.to_index(&shape, "kv-cache")?;
        let k = Cache::new(dim, &shape, dev)?;
        let v = Cache::new(dim, &shape, dev)?;
        Ok(Self { k, v })
    }

    pub fn k(&self) -> &Cache<T, B> {
        &self.k
    }

    pub fn v(&self) -> &Cache<T, B> {
        &self.v
    }

    pub fn append(
        &mut self,
        k: &Tensor<T, B>,
        v: &Tensor<T, B>,
    ) -> Result<(TensorView<T, B>, TensorView<T, B>)> {
        self.k.append(k)?;
        self.v.append(v)?;
        let k = self.k.current_data()?;
        let v = self.v.current_data()?;
        Ok((k, v))
    }
}

#[derive(Debug, Clone)]
pub struct RotatingCache<T: WithDTypeF, B: Backend> {
    all_data: Option<Tensor<T, B>>,
    dim: usize,
    // `offset` is the current write index in the buffer
    offset: usize,
    // The total size of the sequence seen so far.
    current_seq_len: usize,
    // max_seq_len is the size of the rotating buffer, it is actually allowed for the full
    // sequence to grow past this limit.
    max_seq_len: usize,
}

impl<T: WithDTypeF, B: Backend> RotatingCache<T, B> {
    pub fn new(dim: usize, max_seq_len: usize) -> Self {
        Self { all_data: None, dim, offset: 0, current_seq_len: 0, max_seq_len }
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn all_data(&self) -> &Option<Tensor<T, B>> {
        &self.all_data
    }

    pub fn current_data(&self) -> Result<Option<Tensor<T, B>>> {
        let data = match self.all_data.as_ref() {
            None => None,
            Some(d) => {
                if self.current_seq_len >= self.max_seq_len {
                    Some(d.clone())
                } else {
                    Some(d.narrow(self.dim, ..self.current_seq_len)?.contiguous()?)
                }
            }
        };
        Ok(data)
    }

    pub fn reset(&mut self) {
        self.offset = 0;
        self.current_seq_len = 0;
        self.all_data = None;
    }

    pub fn append(&mut self, src: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let seq_len = src.dim(self.dim)?;
        // This doesn't seem very idiomatic but because the creation can fail, it's tricky to use
        // self.all_data.get_or_insert_with.
        if self.all_data.is_none() {
            let mut shape = src.dims().to_vec();
            shape[self.dim] = self.max_seq_len;
            let ad = Tensor::<T, B>::zeros(shape, src.device())?;
            self.all_data = Some(ad)
        };
        let ad = self.all_data.as_mut().unwrap();

        self.current_seq_len += seq_len;
        if seq_len >= self.max_seq_len {
            let to_copy =
                src.narrow(self.dim, seq_len - self.max_seq_len..seq_len)?.contiguous()?;
            ad.slice_set(&to_copy, self.dim, 0)?;
            self.offset = 0;
            // Here we return `src` rather than `ad` so that all the past can be used.
            Ok(src.clone())
        } else {
            let rem_len = self.max_seq_len - self.offset;
            if seq_len <= rem_len {
                ad.slice_set(src, self.dim, self.offset)?;
                self.offset = (self.offset + seq_len) % self.max_seq_len;
            } else {
                // We have to make two copies here as we go over the boundary of the cache.
                if rem_len > 0 {
                    let src1 = src.narrow(self.dim, ..rem_len)?.contiguous()?;
                    ad.slice_set(&src1, self.dim, self.offset)?;
                }
                let src2 = src.narrow(self.dim, rem_len..seq_len)?.contiguous()?;
                ad.slice_set(&src2, self.dim, 0)?;
                self.offset = seq_len - rem_len;
            }
            if self.current_seq_len >= self.max_seq_len {
                Ok(ad.clone())
            } else {
                Ok(ad.narrow(self.dim, ..self.current_seq_len)?.contiguous()?)
            }
        }
    }

    fn get_mask_abs(&self, size1: usize, size2: usize, device: &B) -> Result<Tensor<u8, B>> {
        let context = self.max_seq_len;
        let mask: Vec<_> = (0..size1)
            .flat_map(|i| {
                (0..size2).map(move |j| {
                    u8::from(size1 + j > size2 + i || size1 + j + context < size2 + i)
                })
            })
            .collect();
        Tensor::from_vec(mask, (size1, size2), device)
    }

    fn get_mask_rel(&self, size1: usize, size2: usize, device: &B) -> Result<Tensor<u8, B>> {
        let context = self.max_seq_len;
        let upd_offset = (self.offset + size1) % self.max_seq_len;
        let mask: Vec<_> = (0..size1)
            .flat_map(|pos_src| {
                // The absolute position of the elements that will get added to the cache.
                let pos_src = self.current_seq_len + pos_src;
                (0..size2).map(move |pos_cache_rel| {
                    // The absolute position of the cache elements after the addition.
                    let pos_cache = self.current_seq_len + size1 + pos_cache_rel - upd_offset;
                    let pos_cache = if pos_cache_rel < upd_offset {
                        pos_cache
                    } else {
                        pos_cache - self.max_seq_len
                    };
                    u8::from(pos_cache > pos_src || pos_cache + context < pos_src)
                })
            })
            .collect();
        Tensor::from_vec(mask, (size1, size2), device)
    }

    /// Returns the positions corresponding to all the elements that will be returned
    /// *after* adding `seq_len` to the cache.
    pub fn positions(&self, seq_len: usize) -> Vec<usize> {
        if seq_len <= self.max_seq_len {
            let upd_offset = (self.offset + seq_len) % self.max_seq_len;
            let cache_out_len = (self.current_seq_len + seq_len).min(self.max_seq_len);
            (0..cache_out_len)
                .map(|i| {
                    let pos_cache = self.current_seq_len + seq_len + i - upd_offset;
                    if i < upd_offset { pos_cache } else { pos_cache - self.max_seq_len }
                })
                .collect()
        } else {
            (self.current_seq_len..(self.current_seq_len + seq_len)).collect()
        }
    }

    /// Returns the attn_mask to be applied *after* adding `seq_len` to the cache.
    pub fn attn_mask(&self, seq_len: usize, device: &B) -> Result<Option<Tensor<u8, B>>> {
        let mask = if seq_len == 1 {
            None
        } else {
            let mask = if seq_len < self.max_seq_len {
                let cache_out_len = (self.current_seq_len + seq_len).min(self.max_seq_len);
                self.get_mask_rel(seq_len, cache_out_len, device)?
            } else {
                self.get_mask_abs(seq_len, seq_len, device)?
            };
            Some(mask)
        };
        Ok(mask)
    }
}

#[derive(Debug, Clone)]
pub struct RotatingKvCache<T: WithDTypeF, B: Backend> {
    k: RotatingCache<T, B>,
    v: RotatingCache<T, B>,
}

impl<T: WithDTypeF, B: Backend> RotatingKvCache<T, B> {
    pub fn new(dim: usize, max_seq_len: usize) -> Self {
        let k = RotatingCache::new(dim, max_seq_len);
        let v = RotatingCache::new(dim, max_seq_len);
        Self { k, v }
    }

    pub fn k_cache(&self) -> &RotatingCache<T, B> {
        &self.k
    }

    pub fn v_cache(&self) -> &RotatingCache<T, B> {
        &self.v
    }

    pub fn k_cache_mut(&mut self) -> &mut RotatingCache<T, B> {
        &mut self.k
    }

    pub fn v_cache_mut(&mut self) -> &mut RotatingCache<T, B> {
        &mut self.v
    }

    pub fn k(&self) -> Result<Option<Tensor<T, B>>> {
        self.k.current_data()
    }

    pub fn v(&self) -> Result<Option<Tensor<T, B>>> {
        self.v.current_data()
    }

    pub fn append(
        &mut self,
        k: &Tensor<T, B>,
        v: &Tensor<T, B>,
    ) -> Result<(Tensor<T, B>, Tensor<T, B>)> {
        let out_k = self.k.append(k)?;
        let out_v = self.v.append(v)?;
        Ok((out_k, out_v))
    }

    pub fn offset(&self) -> usize {
        self.k.offset()
    }

    pub fn current_seq_len(&self) -> usize {
        self.k.current_seq_len()
    }

    /// Returns the attn_mask to be applied *after* adding `seq_len` to the cache.
    pub fn attn_mask(&self, seq_len: usize, device: &B) -> Result<Option<Tensor<u8, B>>> {
        self.k.attn_mask(seq_len, device)
    }

    /// Returns the positions corresponding to all the elements that will be returned
    /// *after* adding `seq_len` to the cache.
    pub fn positions(&self, seq_len: usize) -> Vec<usize> {
        self.k.positions(seq_len)
    }

    pub fn reset(&mut self) {
        self.k.reset();
        self.v.reset();
    }
}

#[derive(Debug, Clone)]
pub struct IndicesAndMask<T: WithDTypeF, B: Backend> {
    indices: Tensor<i64, B>,
    mask: Tensor<T, B>,
}

impl<T: WithDTypeF, B: Backend> IndicesAndMask<T, B> {
    pub fn mask(&self) -> &Tensor<T, B> {
        &self.mask
    }
}

#[derive(Debug, Clone)]
pub struct ScatteredKvCache<T: WithDTypeF, B: Backend> {
    k: Tensor<T, B>,
    v: Tensor<T, B>,
    context: usize,
}

impl<T: WithDTypeF, B: Backend> ScatteredKvCache<T, B> {
    pub fn append(
        &mut self,
        k: &Tensor<T, B>,
        v: &Tensor<T, B>,
        iam: &IndicesAndMask<T, B>,
    ) -> Result<(Tensor<T, B>, Tensor<T, B>)> {
        if self.context <= k.dim(2)? {
            return Ok((k.clone(), v.clone()));
        }
        let indices = iam.indices.unsqueeze(2)?.unsqueeze(1)?;
        let indices = indices.broadcast_as(k.shape())?.contiguous()?;
        self.k.scatter_set(&indices, k, 2)?;
        self.v.scatter_set(&indices, v, 2)?;
        Ok((self.k.clone(), self.v.clone()))
    }

    pub fn k(&self) -> &Tensor<T, B> {
        &self.k
    }

    pub fn v(&self) -> &Tensor<T, B> {
        &self.v
    }
}

#[derive(Debug)]
pub struct ScatteredCacheBuilder<B: Backend> {
    context: usize,
    // The current position in the stream, this can be larger than context.
    positions: Vec<usize>,
    // The index where the next element will be stored.
    indices: Vec<usize>,
    device: B,
}

impl<B: Backend> ScatteredCacheBuilder<B> {
    pub fn new(batch_size: usize, context: usize, device: &B) -> Result<Self> {
        let positions = vec![0; batch_size];
        let indices = vec![0; batch_size];
        Ok(Self { positions, indices, context, device: device.clone() })
    }

    pub fn make_cache<T: WithDTypeF>(
        &self,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<ScatteredKvCache<T, B>> {
        let batch_size = self.batch_size();
        let shape = (batch_size, num_heads, self.context, head_dim);
        let k = Tensor::<T, B>::zeros(shape, self.device())?;
        let v = Tensor::<T, B>::zeros(shape, self.device())?;
        Ok(ScatteredKvCache { k, v, context: self.context })
    }

    pub fn positions(&self) -> &[usize] {
        &self.positions
    }

    pub fn reset(&mut self) {
        self.positions.fill(0);
        self.indices.fill(0);
    }

    pub fn batch_size(&self) -> usize {
        self.positions.len()
    }

    pub fn reset_batch_index(&mut self, batch_index: usize) {
        self.positions[batch_index] = 0;
        self.indices[batch_index] = 0;
    }

    #[allow(clippy::needless_range_loop)]
    pub fn indices_and_mask<T: WithDTypeF>(
        &mut self,
        seq_len: usize,
        batch_mask: &[bool],
    ) -> Result<IndicesAndMask<T, B>> {
        // mask shape is (b, h, t, k)
        let context = self.context;
        if self.context <= seq_len {
            return self.indices_and_mask_abs(seq_len, batch_mask);
        }
        let mut attention_masks = Vec::with_capacity(self.batch_size());
        let mut cache_indices = Vec::with_capacity(self.batch_size());
        for (batch_i, &batch_mask) in batch_mask.iter().enumerate() {
            if !batch_mask {
                let masks: Vec<Vec<T>> = vec![vec![T::zero(); context]; seq_len];
                let indices = vec![self.indices[batch_i] as i64; seq_len];
                attention_masks.push(masks);
                cache_indices.push(indices);
            } else {
                let start_index = self.indices[batch_i];
                let start_pos = self.positions[batch_i];
                let mut masks: Vec<Vec<T>> = Vec::with_capacity(seq_len);
                let mut indices = Vec::with_capacity(seq_len);
                let mut all_pos = vec![usize::MAX; context];
                if start_pos < context {
                    for i in 0..start_pos {
                        all_pos[i] = i;
                    }
                } else {
                    let offset = start_pos - start_index;
                    for i in 0..context {
                        all_pos[i] =
                            if i < start_index { i + offset } else { i + offset - context };
                    }
                }
                for seq_i in 0..seq_len {
                    let index = self.indices[batch_i];
                    all_pos[index] = seq_i + start_pos;
                    indices.push(index as i64);
                    self.indices[batch_i] += 1;
                    self.positions[batch_i] += 1;
                    if self.indices[batch_i] >= self.context {
                        self.indices[batch_i] = 0;
                    }
                }

                for seq_i in 0..seq_len {
                    let my_pos = seq_i + start_pos;
                    let mask =
                        all_pos
                            .iter()
                            .map(|&pos| {
                                if pos <= my_pos {
                                    T::zero()
                                } else {
                                    T::from_f32(f32::NEG_INFINITY)
                                }
                            })
                            .collect::<Vec<T>>();
                    masks.push(mask);
                }

                attention_masks.push(masks);
                cache_indices.push(indices);
            }
        }
        let attention_masks =
            attention_masks.into_iter().flat_map(|m| m.into_iter().flatten()).collect::<Vec<T>>();
        let mask = Tensor::from_vec(
            attention_masks,
            (self.batch_size(), 1, seq_len, context),
            self.device(),
        )?;
        let cache_indices: Vec<_> = cache_indices.into_iter().flatten().collect();
        let indices = Tensor::from_vec(cache_indices, (self.batch_size(), seq_len), self.device())?;
        Ok(IndicesAndMask { indices, mask })
    }

    pub fn device(&self) -> &B {
        &self.device
    }

    #[allow(clippy::needless_range_loop)]
    fn indices_and_mask_abs<T: WithDTypeF>(
        &mut self,
        seq_len: usize,
        batch_mask: &[bool],
    ) -> Result<IndicesAndMask<T, B>> {
        let mask = self.get_mask_abs(seq_len, seq_len)?.reshape((
            self.batch_size(),
            1,
            seq_len,
            seq_len,
        ))?;
        let mut cache_indices = Vec::with_capacity(self.batch_size());
        for (batch_i, &batch_mask) in batch_mask.iter().enumerate() {
            if !batch_mask {
                let indices = vec![self.indices[batch_i] as i64; seq_len];
                cache_indices.push(indices);
            } else {
                let mut indices = Vec::with_capacity(seq_len);
                for _ in 0..seq_len {
                    let index = self.indices[batch_i];
                    indices.push(index as i64);
                    self.indices[batch_i] += 1;
                    self.positions[batch_i] += 1;
                    if self.indices[batch_i] >= self.context {
                        self.indices[batch_i] = 0;
                    }
                }
                cache_indices.push(indices);
            }
        }
        let cache_indices = cache_indices.into_iter().flatten().collect();
        let indices = Tensor::from_vec(cache_indices, (self.batch_size(), seq_len), self.device())?;
        Ok(IndicesAndMask { indices, mask })
    }

    fn get_mask_abs<T: WithDTypeF>(&self, size1: usize, size2: usize) -> Result<Tensor<T, B>> {
        let context = self.context;
        let mask: Vec<_> = (0..size1)
            .flat_map(|i| {
                (0..size2).map(move |j| {
                    if size1 + j > size2 + i || size1 + j + context < size2 + i {
                        T::from_f32(f32::NEG_INFINITY)
                    } else {
                        T::zero()
                    }
                })
            })
            .collect();
        Tensor::<T, B>::from_vec(mask, (size1, size2), self.device())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Squeeze dim 1 from the mask (shape: (b, 1, t, c) → (b, t, c)) and return as flat vec.
    fn squeeze_mask<B: Backend>(mask: &Tensor<f32, B>, b: usize, t: usize, c: usize) -> Vec<f32> {
        mask.reshape((b, t, c)).unwrap().to_vec().unwrap()
    }

    #[test]
    fn test_scattered_kv_cache() -> Result<()> {
        let device = crate::CPU;
        let mut cache = ScatteredCacheBuilder::new(2, 5, &device)?;
        let inf = f32::INFINITY;

        let iam = cache.indices_and_mask(1, &[true, false])?;
        assert_eq!(iam.indices.to_vec()?, [0i64, 0]);
        assert_eq!(
            squeeze_mask(&iam.mask, 2, 1, 5),
            [0.0, -inf, -inf, -inf, -inf, 0.0, 0.0, 0.0, 0.0, 0.0]
        );

        let iam = cache.indices_and_mask(1, &[true, false])?;
        assert_eq!(iam.indices.to_vec()?, [1i64, 0]);
        assert_eq!(
            squeeze_mask(&iam.mask, 2, 1, 5),
            [0.0, 0.0, -inf, -inf, -inf, 0.0, 0.0, 0.0, 0.0, 0.0]
        );

        let iam = cache.indices_and_mask(3, &[false, true])?;
        assert_eq!(iam.indices.to_vec()?, [2i64, 2, 2, 0, 1, 2]);
        #[rustfmt::skip]
        assert_eq!(
            squeeze_mask(&iam.mask, 2, 3, 5),
            [
                0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, -inf, -inf, -inf, -inf,  0.0, 0.0, -inf, -inf, -inf,  0.0, 0.0, 0.0, -inf, -inf,
            ]
        );

        let iam = cache.indices_and_mask(3, &[true, true])?;
        assert_eq!(iam.indices.to_vec()?, [2i64, 3, 4, 3, 4, 0]);
        #[rustfmt::skip]
        assert_eq!(
            squeeze_mask(&iam.mask, 2, 3, 5),
            [
                0.0, 0.0, 0.0, -inf, -inf,  0.0, 0.0, 0.0, 0.0, -inf,  0.0, 0.0, 0.0, 0.0, 0.0,
                -inf, 0.0, 0.0, 0.0, -inf,  -inf, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0,
            ]
        );

        let iam = cache.indices_and_mask(1, &[true, false])?;
        assert_eq!(iam.indices.to_vec()?, [0i64, 1]);
        assert_eq!(
            squeeze_mask(&iam.mask, 2, 1, 5),
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );

        let iam = cache.indices_and_mask(2, &[true, false])?;
        assert_eq!(iam.indices.to_vec()?, [1i64, 2, 1, 1]);
        #[rustfmt::skip]
        assert_eq!(
            squeeze_mask(&iam.mask, 2, 2, 5),
            [
                0.0, 0.0, -inf, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0,
            ]
        );

        Ok(())
    }
}

#![allow(clippy::type_complexity)]
use crate::nn::var_builder::Path;
use crate::nn::{Embedding, Linear, RmsNorm};
use crate::{Backend, Result, Tensor, WithDTypeF};

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
}

impl Config {
    /// Llama 3 8B configuration
    pub fn llama3_8b() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 14336,
            vocab_size: 128256,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            max_position_embeddings: 8192,
        }
    }

    /// TinyLlama 1.1B configuration
    /// https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
    pub fn tiny_llama_1_1b() -> Self {
        Self {
            hidden_size: 2048,
            intermediate_size: 5632,
            vocab_size: 32000,
            num_hidden_layers: 22,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            head_dim: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 2048,
        }
    }

    /// SmolLM 135M configuration
    /// https://huggingface.co/HuggingFaceTB/SmolLM-135M
    pub fn smol_lm_135m() -> Self {
        Self {
            hidden_size: 576,
            intermediate_size: 1536,
            vocab_size: 49152,
            num_hidden_layers: 30,
            num_attention_heads: 9,
            num_key_value_heads: 3,
            head_dim: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 2048,
        }
    }

    /// SmolLM 360M configuration
    /// https://huggingface.co/HuggingFaceTB/SmolLM-360M
    pub fn smol_lm_360m() -> Self {
        Self {
            hidden_size: 960,
            intermediate_size: 2560,
            vocab_size: 49152,
            num_hidden_layers: 32,
            num_attention_heads: 15,
            num_key_value_heads: 5,
            head_dim: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 2048,
        }
    }

    /// Tiny test configuration for quick testing (only ~1M params)
    pub fn tiny_test() -> Self {
        Self {
            hidden_size: 64,
            intermediate_size: 128,
            vocab_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 512,
        }
    }
}

pub struct Attention<T: WithDTypeF, B: Backend> {
    q_proj: Linear<T, B>,
    k_proj: Linear<T, B>,
    v_proj: Linear<T, B>,
    o_proj: Linear<T, B>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_kv_groups: usize,
}

impl<T: WithDTypeF, B: Backend> Attention<T, B> {
    pub fn new(
        q_proj: Linear<T, B>,
        k_proj: Linear<T, B>,
        v_proj: Linear<T, B>,
        o_proj: Linear<T, B>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let num_kv_groups = num_heads / num_kv_heads;
        Self { q_proj, k_proj, v_proj, o_proj, num_heads, num_kv_heads, head_dim, num_kv_groups }
    }

    pub fn load(vb: &Path<B>, config: &Config) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;

        let q_proj = Linear::load(vb.pp("q_proj"), hidden_size, num_heads * head_dim)?;
        let k_proj = Linear::load(vb.pp("k_proj"), hidden_size, num_kv_heads * head_dim)?;
        let v_proj = Linear::load(vb.pp("v_proj"), hidden_size, num_kv_heads * head_dim)?;
        let o_proj = Linear::load(vb.pp("o_proj"), num_heads * head_dim, hidden_size)?;

        Ok(Self::new(q_proj, k_proj, v_proj, o_proj, num_heads, num_kv_heads, head_dim))
    }

    #[tracing::instrument(name = "attn", skip_all)]
    pub fn forward(
        &self,
        x: &Tensor<T, B>,
        cos: &Tensor<T, B>,
        sin: &Tensor<T, B>,
        pos: usize,
        kv_cache: Option<(&Tensor<T, B>, &Tensor<T, B>)>,
        fixed_cache: bool,
    ) -> Result<(Tensor<T, B>, Tensor<T, B>, Tensor<T, B>)> {
        let (b, seq_len, _hidden) = x.dims3()?;

        // Project to Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape: (b, seq_len, num_heads * head_dim) -> (b, num_heads, seq_len, head_dim)
        let q = q.reshape((b, seq_len, self.num_heads, self.head_dim))?;
        let q = q.transpose(1, 2)?.contiguous()?;

        let k = k.reshape((b, seq_len, self.num_kv_heads, self.head_dim))?;
        let k = k.transpose(1, 2)?.contiguous()?;

        let v = v.reshape((b, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // Apply RoPE
        let q = q.rope(cos, sin, pos)?;
        let k = k.rope(cos, sin, pos)?;

        // Handle KV cache - cache BEFORE repeat_kv to save memory
        let (k_cache, v_cache, k, v) = match kv_cache {
            Some((prev_k, prev_v)) if fixed_cache => {
                // Preallocated cache: write the new keys/values in place at
                // `pos` and attend over the whole (fixed-size) cache. The
                // causality mask hides both future and not-yet-written
                // positions, and shapes stay identical across decoding steps.
                prev_k.slice_set(&k, 2, pos)?;
                prev_v.slice_set(&v, 2, pos)?;
                (prev_k.clone(), prev_v.clone(), prev_k.clone(), prev_v.clone())
            }
            Some((prev_k, prev_v)) => {
                let k_cat = Tensor::cat(&[prev_k, &k], 2)?;
                let v_cat = Tensor::cat(&[prev_v, &v], 2)?;
                (k_cat.clone(), v_cat.clone(), k_cat, v_cat)
            }
            None => (k.clone(), v.clone(), k, v),
        };

        // Repeat KV heads for grouped query attention
        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        // Scaled dot-product attention
        // Q: (b, num_heads, seq_len, head_dim)
        // K: (b, num_heads, kv_len, head_dim)
        // V: (b, num_heads, kv_len, head_dim)
        let scale = T::from_f32(1.0 / (self.head_dim as f32).sqrt());
        let k_t = k.transpose(2, 3)?;
        let attn_weights = q.matmul(&k_t)?;
        let attn_weights = attn_weights.scale(scale)?;

        // Apply causal mask and softmax
        // Reshape to (batch * heads, seq_q, seq_kv) for causality mask
        let (b, h, seq_q, seq_kv) = attn_weights.dims4()?;
        let attn_weights = attn_weights.reshape((b * h, seq_q, seq_kv))?;
        let attn_weights = attn_weights.apply_causality_mask(pos)?;
        let attn_weights = attn_weights.softmax()?;
        let attn_weights = attn_weights.reshape((b, h, seq_q, seq_kv))?;

        // Attention output
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back: (b, num_heads, seq_len, head_dim) -> (b, seq_len, hidden_size)
        let attn_output =
            attn_output.transpose(1, 2)?.reshape((b, seq_len, self.num_heads * self.head_dim))?;

        // Output projection
        let output = self.o_proj.forward(&attn_output)?;

        Ok((output, k_cache, v_cache))
    }

    fn repeat_kv(&self, x: Tensor<T, B>) -> Result<Tensor<T, B>> {
        if self.num_kv_groups == 1 {
            return Ok(x);
        }
        // x shape: (batch, num_kv_heads, seq_len, head_dim)
        // output shape: (batch, num_heads, seq_len, head_dim)
        // Repeat each KV head num_kv_groups times using index_select
        // indices: [0, 0, ..., 1, 1, ..., 2, 2, ...] with num_kv_groups repetitions each
        let indices: Vec<i64> = (0..self.num_kv_heads as i64)
            .flat_map(|i| std::iter::repeat_n(i, self.num_kv_groups))
            .collect();
        let indices =
            Tensor::from_vec(indices, self.num_kv_heads * self.num_kv_groups, x.device())?;
        x.index_select(&indices, 1)
    }
}

pub struct Mlp<T: WithDTypeF, B: Backend> {
    gate_proj: Linear<T, B>,
    up_proj: Linear<T, B>,
    down_proj: Linear<T, B>,
}

impl<T: WithDTypeF, B: Backend> Mlp<T, B> {
    pub fn new(gate_proj: Linear<T, B>, up_proj: Linear<T, B>, down_proj: Linear<T, B>) -> Self {
        Self { gate_proj, up_proj, down_proj }
    }

    pub fn load(vb: &Path<B>, config: &Config) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        let gate_proj = Linear::load(vb.pp("gate_proj"), hidden_size, intermediate_size)?;
        let up_proj = Linear::load(vb.pp("up_proj"), hidden_size, intermediate_size)?;
        let down_proj = Linear::load(vb.pp("down_proj"), intermediate_size, hidden_size)?;

        Ok(Self::new(gate_proj, up_proj, down_proj))
    }

    #[tracing::instrument(name = "mlp", skip_all)]
    pub fn forward(&self, x: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        // SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
        let gate = self.gate_proj.forward(x)?;
        let gate = gate.silu()?;
        let up = self.up_proj.forward(x)?;
        let hidden = gate.mul(&up)?;
        self.down_proj.forward(&hidden)
    }
}

pub struct TransformerBlock<T: WithDTypeF, B: Backend> {
    attn: Attention<T, B>,
    mlp: Mlp<T, B>,
    input_layernorm: RmsNorm<T, B>,
    post_attention_layernorm: RmsNorm<T, B>,
}

impl<T: WithDTypeF, B: Backend> TransformerBlock<T, B> {
    pub fn new(
        attn: Attention<T, B>,
        mlp: Mlp<T, B>,
        input_layernorm: RmsNorm<T, B>,
        post_attention_layernorm: RmsNorm<T, B>,
    ) -> Self {
        Self { attn, mlp, input_layernorm, post_attention_layernorm }
    }

    pub fn load(vb: &Path<B>, config: &Config) -> Result<Self> {
        let attn = Attention::load(&vb.pp("self_attn"), config)?;
        let mlp = Mlp::load(&vb.pp("mlp"), config)?;
        let input_layernorm =
            RmsNorm::load(vb.pp("input_layernorm"), config.hidden_size, config.rms_norm_eps)?;
        let post_attention_layernorm = RmsNorm::load(
            vb.pp("post_attention_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;

        Ok(Self::new(attn, mlp, input_layernorm, post_attention_layernorm))
    }

    #[tracing::instrument(name = "transformer-block", skip_all)]
    pub fn forward(
        &self,
        x: &Tensor<T, B>,
        cos: &Tensor<T, B>,
        sin: &Tensor<T, B>,
        pos: usize,
        kv_cache: Option<(&Tensor<T, B>, &Tensor<T, B>)>,
        fixed_cache: bool,
    ) -> Result<(Tensor<T, B>, Tensor<T, B>, Tensor<T, B>)> {
        // Pre-norm architecture
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let (attn_out, k_cache, v_cache) =
            self.attn.forward(&x, cos, sin, pos, kv_cache, fixed_cache)?;
        let x = residual.add(&attn_out)?;

        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&x)?;
        let x = residual.add(&mlp_out)?;

        Ok((x, k_cache, v_cache))
    }
}

pub struct Llama<T: WithDTypeF, B: Backend> {
    embed_tokens: Embedding<T, B>,
    layers: Vec<TransformerBlock<T, B>>,
    norm: RmsNorm<T, B>,
    lm_head: Linear<T, B>,
    cos_cache: Tensor<T, B>,
    sin_cache: Tensor<T, B>,
}

pub struct KvCache<T: WithDTypeF, B: Backend> {
    kvs: Vec<(Tensor<T, B>, Tensor<T, B>)>,
    /// Sequence capacity when the cache is preallocated, `None` for a cache
    /// growing by concatenation.
    max_seq_len: Option<usize>,
}

impl<T: WithDTypeF, B: Backend> KvCache<T, B> {
    /// Preallocate a fixed-capacity cache: keys and values are written in
    /// place and attention always runs over `max_seq_len` positions, so the
    /// tensor shapes are identical for every decoding step. This costs
    /// compute proportional to the capacity but keeps shape-specialized
    /// backends (e.g. xla) from recompiling at each step.
    ///
    /// Pass the cache to [`Llama::forward`] from the very first call on, and
    /// keep `pos + tokens.len() <= max_seq_len`.
    pub fn fixed(config: &Config, batch: usize, max_seq_len: usize, dev: &B) -> Result<Self> {
        let shape = (batch, config.num_key_value_heads, max_seq_len, config.head_dim);
        let mut kvs = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            let k = Tensor::zeros(shape, dev)?;
            let v = Tensor::zeros(shape, dev)?;
            kvs.push((k, v));
        }
        Ok(Self { kvs, max_seq_len: Some(max_seq_len) })
    }
}

impl<T: WithDTypeF, B: Backend> Llama<T, B> {
    pub fn new(
        embed_tokens: Embedding<T, B>,
        layers: Vec<TransformerBlock<T, B>>,
        norm: RmsNorm<T, B>,
        lm_head: Linear<T, B>,
        cos_cache: Tensor<T, B>,
        sin_cache: Tensor<T, B>,
    ) -> Self {
        Self { embed_tokens, layers, norm, lm_head, cos_cache, sin_cache }
    }

    pub fn load(vb: &Path<B>, config: &Config) -> Result<Self> {
        let model = vb.pp("model");

        let embed_tokens =
            Embedding::load(model.pp("embed_tokens"), config.vocab_size, config.hidden_size)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(TransformerBlock::load(&model.pp(format!("layers.{i}")), config)?);
        }

        let norm = RmsNorm::load(model.pp("norm"), config.hidden_size, config.rms_norm_eps)?;

        // lm_head might be tied to embed_tokens in some models
        let lm_head = match vb.contains("lm_head.weight") {
            true => Linear::load(vb.pp("lm_head"), config.hidden_size, config.vocab_size)?,
            false => Linear::new(embed_tokens.embeddings().clone()),
        };

        let (cos_cache, sin_cache) = precompute_freqs_cis(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            vb.device(),
        )?;

        Ok(Self::new(embed_tokens, layers, norm, lm_head, cos_cache, sin_cache))
    }

    #[tracing::instrument(name = "llama-forward", skip_all)]
    pub fn forward(
        &self,
        tokens: &[u32],
        pos: usize,
        kv_caches: Option<&KvCache<T, B>>,
    ) -> Result<(Tensor<T, B>, KvCache<T, B>)> {
        // Token embedding: (seq_len,) -> (1, seq_len, hidden_size)
        let token_ids = Tensor::from_vec(
            tokens.iter().map(|&t| t as i64).collect(),
            tokens.len(),
            self.embed_tokens.device(),
        )?;
        let mut x = self.embed_tokens.forward(&token_ids)?;
        x = x.reshape((1, tokens.len(), ()))?;

        // Run through transformer layers
        let max_seq_len = kv_caches.and_then(|c| c.max_seq_len);
        let mut kvs = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            let kv_cache = kv_caches.map(|c| (&c.kvs[i].0, &c.kvs[i].1));
            let (new_x, k_cache, v_cache) = layer.forward(
                &x,
                &self.cos_cache,
                &self.sin_cache,
                pos,
                kv_cache,
                max_seq_len.is_some(),
            )?;
            x = new_x;
            kvs.push((k_cache, v_cache));
        }

        // Final norm
        let x = self.norm.forward(&x)?;

        // LM head: (1, seq_len, hidden_size) -> (1, seq_len, vocab_size)
        let logits = self.lm_head.forward(&x)?;

        Ok((logits, KvCache { kvs, max_seq_len }))
    }
}

pub fn precompute_freqs_cis<T: WithDTypeF, B: Backend>(
    head_dim: usize,
    max_seq_len: usize,
    theta: f32,
    dev: &B,
) -> Result<(Tensor<T, B>, Tensor<T, B>)> {
    let half_dim = head_dim / 2;
    let mut freqs = Vec::with_capacity(half_dim);
    for i in 0..half_dim {
        let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
        freqs.push(freq);
    }

    let mut cos_data = Vec::with_capacity(max_seq_len * half_dim);
    let mut sin_data = Vec::with_capacity(max_seq_len * half_dim);

    for pos in 0..max_seq_len {
        for &freq in &freqs {
            let angle = pos as f32 * freq;
            cos_data.push(T::from_f32(angle.cos()));
            sin_data.push(T::from_f32(angle.sin()));
        }
    }

    let shape: crate::Shape = (max_seq_len, half_dim).into();
    let cos = Tensor::from_vec(cos_data, shape.clone(), dev)?;
    let sin = Tensor::from_vec(sin_data, shape, dev)?;
    Ok((cos, sin))
}

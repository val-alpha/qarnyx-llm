//! Qwen3.5-2B hybrid CPU reference model.
//! Implements the 24-layer 3:1 Gated DeltaNet / gated-attention text stack.

use qarnyx_core::{Backend, DType, TensorView};
use qarnyx_gguf::{GgufFile, MetadataValue, TensorInfo};
use qarnyx_quant::{dequantize, GgmlType};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};
use crate::{ModelError, TensorLoadError};

const H: usize = 2048;
const FF: usize = 6144;
const N: usize = 24;
const LH: usize = 16;
const LD: usize = 128;
const LIN: usize = LH * LD;
const FH: usize = 8;
const FHD: usize = 256;
const KV: usize = 2 * FHD;
const Q: usize = FH * FHD;
const ROPE: usize = 64;
const VOCAB: usize = 248_320;
const CONV: usize = 4;

#[derive(Debug, Clone, Default)]
pub struct Qwen35Profile {
    entries: BTreeMap<&'static str, ProfileEntry>,
}

#[derive(Debug, Clone, Copy, Default)]
struct ProfileEntry {
    calls: u64,
    nanos: u128,
}

impl Qwen35Profile {
    fn record(&mut self, label: &'static str, elapsed: Duration) {
        let entry = self.entries.entry(label).or_default();
        entry.calls += 1;
        entry.nanos += elapsed.as_nanos();
    }

    pub fn report(&self) -> String {
        let mut rows: Vec<_> = self.entries.iter().collect();
        rows.sort_by_key(|(_, e)| std::cmp::Reverse(e.nanos));
        let mut out = String::from("Qwen3.5 profile (aggregate):\n");
        out.push_str("  label                         calls      total_ms      avg_us\n");
        for (label, entry) in rows {
            let total_ms = entry.nanos as f64 / 1_000_000.0;
            let avg_us = entry.nanos as f64 / entry.calls as f64 / 1_000.0;
            out.push_str(&format!(
                "  {:<28} {:>5} {:>13.3} {:>12.3}\n",
                label, entry.calls, total_ms, avg_us
            ));
        }
        out
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35Config {
    pub block_count: usize,
    pub embedding_length: usize,
    pub feed_forward_length: usize,
    pub attention_head_count: usize,
    pub attention_head_count_kv: usize,
    pub attention_key_length: usize,
    pub attention_value_length: usize,
    pub rms_epsilon: f32,
    pub rope_theta: f32,
    pub rope_dimension: usize,
    pub conv_kernel: usize,
    pub state_size: usize,
    pub group_count: usize,
    pub time_step_rank: usize,
    pub inner_size: usize,
    pub recurrent_layers: Vec<bool>,
}

impl Qwen35Config {
    pub fn from_gguf(g: &GgufFile) -> Result<Self, TensorLoadError> {
        if g.metadata_string("general.architecture") != Some("qwen35") {
            return Err(TensorLoadError::Config("GGUF architecture is not qwen35".into()));
        }
        let c = Self {
            block_count: u(g, "qwen35.block_count")?,
            embedding_length: u(g, "qwen35.embedding_length")?,
            feed_forward_length: u(g, "qwen35.feed_forward_length")?,
            attention_head_count: u(g, "qwen35.attention.head_count")?,
            attention_head_count_kv: u(g, "qwen35.attention.head_count_kv")?,
            attention_key_length: u(g, "qwen35.attention.key_length")?,
            attention_value_length: u(g, "qwen35.attention.value_length")?,
            rms_epsilon: f(g, "qwen35.attention.layer_norm_rms_epsilon")?,
            rope_theta: f(g, "qwen35.rope.freq_base")?,
            rope_dimension: u(g, "qwen35.rope.dimension_count")?,
            conv_kernel: u(g, "qwen35.ssm.conv_kernel")?,
            state_size: u(g, "qwen35.ssm.state_size")?,
            group_count: u(g, "qwen35.ssm.group_count")?,
            time_step_rank: u(g, "qwen35.ssm.time_step_rank")?,
            inner_size: u(g, "qwen35.ssm.inner_size")?,
            recurrent_layers: match g.metadata_value("qwen35.attention.recurrent_layers") {
                Some(_) => bools(g, "qwen35.attention.recurrent_layers")?,
                None => (0..u(g, "qwen35.block_count")?)
                    .map(|i| (i + 1) % u(g, "qwen35.full_attention_interval").unwrap_or(4) != 0)
                    .collect(),
            },
        };
        if c.block_count != N || c.embedding_length != H || c.feed_forward_length != FF
            || c.attention_head_count != FH || c.attention_head_count_kv != 2
            || c.attention_key_length != FHD || c.attention_value_length != FHD
            || c.rope_dimension != ROPE || c.conv_kernel != CONV
            || c.state_size != LD || c.group_count != LH
            || c.time_step_rank != LH || c.inner_size != LIN
            || c.recurrent_layers.len() < N {
            return Err(TensorLoadError::Config("unsupported Qwen3.5 configuration; expected 2B text layout".into()));
        }
        Ok(c)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Qwen35Linear<'a> {
    pub attn_norm: &'a [f32], pub qkv: TensorView<'a>, pub gate: TensorView<'a>,
    pub conv: TensorView<'a>, pub dt: &'a [f32], pub a: &'a [f32],
    pub beta: TensorView<'a>, pub alpha: TensorView<'a>, pub ssm_norm: &'a [f32],
    pub out: TensorView<'a>, pub ffn_norm: &'a [f32],
    pub ffn_gate: TensorView<'a>, pub ffn_up: TensorView<'a>, pub ffn_down: TensorView<'a>,
}

#[derive(Debug, Clone, Copy)]
pub struct Qwen35Attention<'a> {
    pub attn_norm: &'a [f32], pub q: TensorView<'a>, pub k: TensorView<'a>,
    pub v: TensorView<'a>, pub out: TensorView<'a>, pub q_norm: &'a [f32],
    pub k_norm: &'a [f32], pub ffn_norm: &'a [f32],
    pub ffn_gate: TensorView<'a>, pub ffn_up: TensorView<'a>, pub ffn_down: TensorView<'a>,
}

#[derive(Debug, Clone, Copy)]
pub enum Qwen35Layer<'a> { Linear(Qwen35Linear<'a>), Attention(Qwen35Attention<'a>) }

#[derive(Debug, Clone)]
pub struct Qwen35Cache {
    pub sequence_len: usize,
    recurrent: Vec<Vec<f32>>,
    conv: Vec<Vec<f32>>,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}
impl Qwen35Cache {
    pub fn new(c: &Qwen35Config) -> Self {
        let state = LH * LD * LD;
        Self {
            sequence_len: 0,
            recurrent: (0..c.block_count).map(|_| vec![0.0; state]).collect(),
            conv: (0..c.block_count).map(|_| vec![0.0; LIN * 3 * (CONV-1)]).collect(),
            keys: (0..c.block_count).map(|_| Vec::new()).collect(),
            values: (0..c.block_count).map(|_| Vec::new()).collect(),
        }
    }
    pub fn clear(&mut self) {
        self.sequence_len = 0;
        for x in &mut self.recurrent { x.fill(0.0); }
        for x in &mut self.conv { x.fill(0.0); }
        for x in &mut self.keys { x.clear(); }
        for x in &mut self.values { x.clear(); }
    }
}

pub struct Qwen35Model<'a> {
    pub config: Qwen35Config,
    pub layers: Vec<Qwen35Layer<'a>>,
    pub token_embedding: TensorView<'a>,
    pub output_norm: Vec<f32>,
    pub output: TensorView<'a>,
}

impl<'a> Qwen35Model<'a> {
    pub fn from_gguf(g: &GgufFile, data: &'a [u8]) -> Result<Self, TensorLoadError> {
        let c = Qwen35Config::from_gguf(g)?;
        let vocab = token_count(g)?;
        if vocab != VOCAB { return Err(TensorLoadError::Config("expected Qwen3.5-2B vocabulary 248320".into())); }
        let emb = matrix(g, data, "token_embd.weight", &[vocab, H])?;
        let out = match find(g, "output.weight") {
            Ok(_) => matrix(g, data, "output.weight", &[vocab, H])?,
            Err(TensorLoadError::MissingTensor(_)) => emb,
            Err(e) => return Err(e),
        };
        let output_norm = vector(g, data, "output_norm.weight", &[H])?;
        let mut layers = Vec::with_capacity(N);
        for i in 0..N { layers.push(load_layer(g, data, &c, i)?); }
        Ok(Self { config:c, layers, token_embedding:emb, output_norm, output:out })
    }

    pub fn vocab_size(&self) -> usize { self.output.shape[0] }

    pub fn embed_token(&self, id: usize) -> Result<Vec<f32>, ModelError> {
        if id >= self.vocab_size() { return Err(ModelError::InvalidConfig("token ID outside vocabulary")); }
        row(self.token_embedding, id, H)
    }

    pub fn forward_logits<B: Backend>(&self, b: &B, id: usize, pos: usize, cache: &mut Qwen35Cache)
        -> Result<Vec<f32>, ModelError> {
        let x = self.embed_token(id)?;
        let h = self.forward_with_cache(b, &x, pos, cache)?;
        let n = rms0(&h, &self.output_norm, self.config.rms_epsilon)?;
        let mut logits = vec![0.0; self.vocab_size()];
        b.matvec(self.output, &n, &mut logits)?;
        Ok(logits)
    }

    pub fn forward_logits_profiled<B: Backend>(
        &self,
        b: &B,
        id: usize,
        pos: usize,
        cache: &mut Qwen35Cache,
        profile: &mut Qwen35Profile,
    ) -> Result<Vec<f32>, ModelError> {
        let x = self.embed_token(id)?;
        let h = self.forward_with_cache_profiled(b, &x, pos, cache, profile)?;
        let n = rms0(&h, &self.output_norm, self.config.rms_epsilon)?;
        let mut logits = vec![0.0; self.vocab_size()];
        let start = Instant::now();
        b.matvec(self.output, &n, &mut logits)?;
        profile.record("output_logits_matvec", start.elapsed());
        Ok(logits)
    }

    pub fn forward_with_cache<B: Backend>(&self, b: &B, input: &[f32], pos: usize, cache: &mut Qwen35Cache)
        -> Result<Vec<f32>, ModelError> {
        let mut profile = None;
        self.forward_with_cache_internal(b, input, pos, cache, &mut profile)
    }

    pub fn forward_with_cache_profiled<B: Backend>(
        &self,
        b: &B,
        input: &[f32],
        pos: usize,
        cache: &mut Qwen35Cache,
        profile: &mut Qwen35Profile,
    ) -> Result<Vec<f32>, ModelError> {
        let mut profile = Some(profile);
        self.forward_with_cache_internal(b, input, pos, cache, &mut profile)
    }

    fn forward_with_cache_internal<B: Backend>(
        &self,
        b: &B,
        input: &[f32],
        pos: usize,
        cache: &mut Qwen35Cache,
        profile: &mut Option<&mut Qwen35Profile>,
    ) -> Result<Vec<f32>, ModelError> {
        if pos != cache.sequence_len || input.len() != H {
            return Err(ModelError::InvalidConfig("Qwen3.5 cache position/input shape mismatch"));
        }
        let mut h = input.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            let start = Instant::now();
            h = match layer {
                Qwen35Layer::Linear(w) => linear(b, &self.config, w, &h, &mut cache.recurrent[i], &mut cache.conv[i], profile)?,
                Qwen35Layer::Attention(w) => attention(b, &self.config, w, &h, pos, &mut cache.keys[i], &mut cache.values[i], profile)?,
            };
            if let Some(p) = profile.as_deref_mut() {
                p.record(if matches!(layer, Qwen35Layer::Linear(_)) { "layer_linear_total" } else { "layer_attention_total" }, start.elapsed());
            }
        }
        cache.sequence_len += 1;
        Ok(h)
    }
}

fn linear<B: Backend>(b: &B, c: &Qwen35Config, w: &Qwen35Linear<'_>, input: &[f32],
    state: &mut [f32], conv_state: &mut [f32], profile: &mut Option<&mut Qwen35Profile>) -> Result<Vec<f32>, ModelError> {
    let x = rms0(input, w.attn_norm, c.rms_epsilon)?;
    let mut qkv = vec![0.0; LIN*3];
    let mut z = vec![0.0; LIN];
    let mut beta = vec![0.0; LH];
    let mut alpha = vec![0.0; LH];
    timed_matvec(b, w.qkv, &x, &mut qkv, profile, "linear_qkv_matvec")?;
    timed_matvec(b, w.gate, &x, &mut z, profile, "linear_gate_matvec")?;
    timed_matvec(b, w.beta, &x, &mut beta, profile, "linear_beta_matvec")?;
    timed_matvec(b, w.alpha, &x, &mut alpha, profile, "linear_alpha_matvec")?;

    let conv_start = Instant::now();
    let mut conv = vec![0.0; LIN*3];
    // GGUF layout is q(2048), k(2048), v(2048) for the 16x128 linear heads.
    let mut kernel = vec![0.0; CONV];
    for ch in 0..qkv.len() {
        decode_into(w.conv, ch, &mut kernel)?;
        let base = ch * (CONV-1);
        let current = qkv[ch];
        let y = kernel[0]*conv_state[base] + kernel[1]*conv_state[base+1]
            + kernel[2]*conv_state[base+2] + kernel[3]*current;
        conv[ch] = silu(y);
        conv_state[base] = conv_state[base+1];
        conv_state[base+1] = conv_state[base+2];
        conv_state[base+2] = current;
    }

    if let Some(p) = profile.as_deref_mut() { p.record("linear_conv_and_state", conv_start.elapsed()); }

    let mut q = vec![0.0; LIN];
    let mut k = vec![0.0; LIN];
    let mut v = vec![0.0; LIN];
    q.copy_from_slice(&conv[..LIN]);
    k.copy_from_slice(&conv[LIN..2*LIN]);
    v.copy_from_slice(&conv[2*LIN..3*LIN]);
    let recurrent_start = Instant::now();
    let core = recurrent_heads(
        b, &mut q, &mut k, &v, &z, &beta, &alpha, w.a, w.dt, state, w.ssm_norm,
        c.rms_epsilon,
    )?;
    if let Some(p) = profile.as_deref_mut() {
        p.record("linear_recurrent_heads", recurrent_start.elapsed());
    }
    let mut y=vec![0.0;H];
    timed_matvec(b, w.out, &core, &mut y, profile, "linear_out_matvec")?;
    add(&mut y,input);
    ffn(b,c,w.ffn_norm,w.ffn_gate,w.ffn_up,w.ffn_down,&y, profile)
}

fn recurrent_heads<B: Backend>(
    b: &B,
    q: &mut [f32],
    k: &mut [f32],
    v: &[f32],
    z: &[f32],
    beta: &[f32],
    alpha: &[f32],
    a: &[f32],
    dt: &[f32],
    state: &mut [f32],
    ssm_norm: &[f32],
    rms_epsilon: f32,
) -> Result<Vec<f32>, ModelError> {
    if q.len() != LIN
        || k.len() != LIN
        || v.len() != LIN
        || z.len() != LIN
        || beta.len() != LH
        || alpha.len() != LH
        || a.len() != LH
        || dt.len() != LH
        || state.len() != LH * LD * LD
        || ssm_norm.len() != LD
    {
        return Err(ModelError::ShapeMismatch(
            "Qwen3.5 recurrent head buffers have incompatible lengths".into(),
        ));
    }

    // Each head owns its q/k vectors. Normalize them before dispatch so the
    // parallel state update only reads these buffers.
    for head in 0..LH {
        let q_head = &mut q[head * LD..(head + 1) * LD];
        l2(q_head);
        for value in q_head {
            *value /= (LD as f32).sqrt();
        }
        l2(&mut k[head * LD..(head + 1) * LD]);
    }

    // Recurrent time order is preserved within a head. Only independent heads
    // are scheduled concurrently, and each task receives one disjoint state block.
    b.parallel_for_each_mut(state, LD * LD, &|head, head_state| {
        let key_head = &k[head * LD..(head + 1) * LD];
        let value_head = &v[head * LD..(head + 1) * LD];
        let decay = (-a[head].exp() * softplus(alpha[head] + dt[head])).exp();
        let beta_value = sigmoid(beta[head]);

        for column in 0..LD {
            for row in 0..LD {
                head_state[row * LD + column] *= decay;
            }
        }

        let mut memory = [0.0_f32; LD];
        for column in 0..LD {
            for row in 0..LD {
                memory[column] += head_state[row * LD + column] * key_head[row];
            }
        }

        for row in 0..LD {
            let delta = (value_head[row] - memory[row]) * beta_value;
            for column in 0..LD {
                head_state[column * LD + row] += key_head[column] * delta;
            }
        }
    })?;

    // Each output head reads only its updated state and writes to its own slice.
    // The per-head reduction order is unchanged.
    let mut core = vec![0.0_f32; LIN];
    b.parallel_for_each_mut(&mut core, LD, &|head, head_core| {
        let head_state = &state[head * LD * LD..(head + 1) * LD * LD];
        let query_head = &q[head * LD..(head + 1) * LD];
        let gate_head = &z[head * LD..(head + 1) * LD];

        for column in 0..LD {
            for row in 0..LD {
                head_core[column] += head_state[row * LD + column] * query_head[row];
            }
        }

        let mean_square = head_core
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            / LD as f64;
        let rms = (mean_square + f64::from(rms_epsilon))
            .sqrt()
            .recip() as f32;
        for column in 0..LD {
            head_core[column] *= rms * ssm_norm[column] * silu(gate_head[column]);
        }
    })?;

    Ok(core)
}

fn attention<B: Backend>(b:&B,c:&Qwen35Config,w:&Qwen35Attention<'_>,input:&[f32],pos:usize,
    keys:&mut Vec<f32>,values:&mut Vec<f32>, profile:&mut Option<&mut Qwen35Profile>) -> Result<Vec<f32>,ModelError> {
    let x=rms0(input,w.attn_norm,c.rms_epsilon)?;
    let mut qg=vec![0.0;Q*2]; let mut k=vec![0.0;KV]; let mut v=vec![0.0;KV];
    timed_matvec(b, w.q, &x, &mut qg, profile, "attention_q_matvec")?;
    timed_matvec(b, w.k, &x, &mut k, profile, "attention_k_matvec")?;
    timed_matvec(b, w.v, &x, &mut v, profile, "attention_v_matvec")?;
    // GGUF q_proj stores each attention head as [query | gate], not
    // [all queries | all gates]. Deinterleave before applying q_norm/RoPE.
    let mut q = vec![0.0; Q];
    let mut gate = vec![0.0; Q];
    for h in 0..FH {
        let src = h * FHD * 2;
        let dst = h * FHD;
        q[dst..dst + FHD].copy_from_slice(&qg[src..src + FHD]);
        gate[dst..dst + FHD].copy_from_slice(&qg[src + FHD..src + FHD * 2]);
        headnorm(&mut q[dst..dst + FHD], w.q_norm, c.rms_epsilon);
        rope(&mut q[dst..dst + ROPE], c.rope_theta, pos);
    }
    for h in 0..2 {
        headnorm(&mut k[h*FHD..(h+1)*FHD], w.k_norm, c.rms_epsilon);
        rope(&mut k[h*FHD..h*FHD+ROPE], c.rope_theta, pos);
    }
    let attn_start = Instant::now();
    keys.extend_from_slice(&k); values.extend_from_slice(&v);
    let seq=keys.len()/KV; let mut a=vec![0.0;Q];
    for qh in 0..FH {
        let kh=qh/4; let qb=qh*FHD; let mut s=vec![0.0;seq];
        for t in 0..seq { let kb=t*KV+kh*FHD; s[t]=dot(&q[qb..qb+FHD],&keys[kb..kb+FHD])/(FHD as f32).sqrt(); }
        softmax(&mut s);
        for t in 0..seq { let vb=t*KV+kh*FHD; for j in 0..FHD { a[qb+j]+=s[t]*values[vb+j]; } }
    }
    for i in 0..Q { a[i] *= sigmoid(gate[i]); }
    if let Some(p) = profile.as_deref_mut() { p.record("attention_core_softmax", attn_start.elapsed()); }
    let mut y=vec![0.0;H];
    timed_matvec(b, w.out, &a, &mut y, profile, "attention_out_matvec")?;
    add(&mut y,input);
    ffn(b,c,w.ffn_norm,w.ffn_gate,w.ffn_up,w.ffn_down,&y, profile)
}

fn ffn<B:Backend>(b:&B,c:&Qwen35Config,norm:&[f32],gate:TensorView<'_>,up:TensorView<'_>,down:TensorView<'_>,input:&[f32], profile:&mut Option<&mut Qwen35Profile>)->Result<Vec<f32>,ModelError>{
    let x=rms0(input,norm,c.rms_epsilon)?; let mut g=vec![0.0;FF]; let mut u=vec![0.0;FF];
    timed_matvec(b, gate, &x, &mut g, profile, "ffn_gate_matvec")?;
    timed_matvec(b, up, &x, &mut u, profile, "ffn_up_matvec")?;
    for i in 0..FF { g[i]=silu(g[i])*u[i]; }
    let mut y=vec![0.0;H]; timed_matvec(b, down, &g, &mut y, profile, "ffn_down_matvec")?;
    add(&mut y,input); Ok(y)
}

fn timed_matvec<B: Backend>(
    b: &B,
    weights: TensorView<'_>,
    input: &[f32],
    output: &mut [f32],
    profile: &mut Option<&mut Qwen35Profile>,
    label: &'static str,
) -> Result<(), ModelError> {
    let start = Instant::now();
    b.matvec(weights, input, output)?;
    if let Some(p) = profile.as_deref_mut() {
        p.record(label, start.elapsed());
    }
    Ok(())
}

fn load_layer<'a>(g:&GgufFile,d:&'a[u8],c:&Qwen35Config,i:usize)->Result<Qwen35Layer<'a>,TensorLoadError>{
    let p=format!("blk.{i}");
    let an=Box::leak(vector(g,d,&format!("{p}.attn_norm.weight"),&[H])?.into_boxed_slice());
    let fnorm=Box::leak(vector(g,d,&format!("{p}.post_attention_norm.weight"),&[H])?.into_boxed_slice());
    let fg=matrix(g,d,&format!("{p}.ffn_gate.weight"),&[FF,H])?;
    let fu=matrix(g,d,&format!("{p}.ffn_up.weight"),&[FF,H])?;
    let fd=matrix(g,d,&format!("{p}.ffn_down.weight"),&[H,FF])?;
    if c.recurrent_layers[i] {
        Ok(Qwen35Layer::Linear(Qwen35Linear{
            attn_norm:an,qkv:matrix(g,d,&format!("{p}.attn_qkv.weight"),&[LIN*3,H])?,
            gate:matrix(g,d,&format!("{p}.attn_gate.weight"),&[LIN,H])?,
            conv:matrix(g,d,&format!("{p}.ssm_conv1d.weight"),&[LIN*3,CONV])?,
            dt:Box::leak(vector(g,d,&format!("{p}.ssm_dt.bias"),&[LH])?.into_boxed_slice()),
            a:Box::leak(vector(g,d,&format!("{p}.ssm_a"),&[LH])?.into_boxed_slice()),
            beta:matrix(g,d,&format!("{p}.ssm_beta.weight"),&[LH,H])?,
            alpha:matrix(g,d,&format!("{p}.ssm_alpha.weight"),&[LH,H])?,
            ssm_norm:Box::leak(vector(g,d,&format!("{p}.ssm_norm.weight"),&[LD])?.into_boxed_slice()),
            out:matrix(g,d,&format!("{p}.ssm_out.weight"),&[H,LIN])?,
            ffn_norm:fnorm,ffn_gate:fg,ffn_up:fu,ffn_down:fd,
        }))
    } else {
        Ok(Qwen35Layer::Attention(Qwen35Attention{
            attn_norm:an,q:matrix(g,d,&format!("{p}.attn_q.weight"),&[Q*2,H])?,
            k:matrix(g,d,&format!("{p}.attn_k.weight"),&[KV,H])?,
            v:matrix(g,d,&format!("{p}.attn_v.weight"),&[KV,H])?,
            out:matrix(g,d,&format!("{p}.attn_output.weight"),&[H,Q])?,
            q_norm:Box::leak(vector(g,d,&format!("{p}.attn_q_norm.weight"),&[FHD])?.into_boxed_slice()),
            k_norm:Box::leak(vector(g,d,&format!("{p}.attn_k_norm.weight"),&[FHD])?.into_boxed_slice()),
            ffn_norm:fnorm,ffn_gate:fg,ffn_up:fu,ffn_down:fd,
        }))
    }
}

fn rms0(x:&[f32],w:&[f32],eps:f32)->Result<Vec<f32>,ModelError>{
    if x.len()!=w.len(){return Err(ModelError::ShapeMismatch("Qwen3.5 RMSNorm shape mismatch".into()));}
    // Qwen3.5 GGUF stores the effective RMSNorm scale. The +1 offset used
    // by the source model is folded into the GGUF norm weights, so runtime
    // inference must apply the weight directly (matching llama.cpp).
    let m=x.iter().map(|v|f64::from(*v)*f64::from(*v)).sum::<f64>()/x.len() as f64;
    let inv=(m+f64::from(eps)).sqrt().recip() as f32;
    Ok(x.iter().zip(w).map(|(a,b)|a*inv*b).collect())
}
fn headnorm(x:&mut[f32],w:&[f32],eps:f32){
    let m=x.iter().map(|v|f64::from(*v)*f64::from(*v)).sum::<f64>()/x.len() as f64;
    let inv=(m+f64::from(eps)).sqrt().recip() as f32;
    for(i,a)in x.iter_mut().enumerate(){*a*=inv*w[i];}
}
fn l2(x:&mut[f32]){let s=x.iter().map(|v|f64::from(*v)*f64::from(*v)).sum::<f64>();let inv=(s+1e-6).sqrt().recip() as f32;for v in x{*v*=inv;}}
fn rope(x: &mut [f32], theta: f32, pos: usize) {
    // Qwen3.5 uses split-half rotate_half on its 64-dimensional rotary slice.
    // The MRoPE frequency layout is interleaved; vector rotation is not adjacent-pair RoPE.
    let half = ROPE / 2;
    for i in 0..half {
        let angle = (pos as f32) * theta.powf(-((2 * i) as f32) / ROPE as f32);
        let (cos, sin) = (angle.cos(), angle.sin());
        let u = x[i];
        let v = x[i + half];
        x[i] = u * cos - v * sin;
        x[i + half] = u * sin + v * cos;
    }
}

fn dot(a:&[f32],b:&[f32])->f32{a.iter().zip(b).map(|(x,y)|x*y).sum()}
fn add(a:&mut[f32],b:&[f32]){for(i,x)in a.iter_mut().enumerate(){*x+=b[i];}}
fn softmax(x:&mut[f32]){let m=x.iter().copied().fold(f32::NEG_INFINITY,f32::max);let mut s=0.0f64;for v in x.iter_mut(){*v=(*v-m).exp();s+=f64::from(*v);}for v in x{*v=(f64::from(*v)/s)as f32;}}
fn silu(x:f32)->f32{if x>=0.0{x/(1.0+(-x).exp())}else{let e=x.exp();x*e/(1.0+e)}}
fn sigmoid(x:f32)->f32{if x>=0.0{1.0/(1.0+(-x).exp())}else{let e=x.exp();e/(1.0+e)}}
fn softplus(x:f32)->f32{if x>20.0{x}else if x< -20.0{x.exp()}else{(1.0+x.exp()).ln()}}

fn row(t:TensorView<'_>,r:usize,n:usize)->Result<Vec<f32>,ModelError>{let mut o=vec![0.0;n];decode_into(t,r,&mut o)?;Ok(o)}
fn decode_into(t:TensorView<'_>,r:usize,o:&mut[f32])->Result<(),ModelError>{
    let ty=dtype_id(t.dtype).ok_or(ModelError::InvalidConfig("unsupported tensor dtype"))?;
    let rb=t.data.len()/t.shape[0];let s=r.checked_mul(rb).ok_or(ModelError::InvalidConfig("row overflow"))?;
    let e=s.checked_add(rb).ok_or(ModelError::InvalidConfig("row overflow"))?;
    let v=dequantize(ty,&[o.len()as u64],t.data.get(s..e).ok_or(ModelError::InvalidConfig("row out of bounds"))?)
        .map_err(|_|ModelError::InvalidConfig("dequantization failed"))?;o.copy_from_slice(&v);Ok(())
}
fn matrix<'a>(g:&GgufFile,d:&'a[u8],name:&str,e:&[usize])->Result<TensorView<'a>,TensorLoadError>{let t=find(g,name)?;if t.dimensions.len()!=2||t.dimensions[0]!=e[1]as u64||t.dimensions[1]!=e[0]as u64{return Err(TensorLoadError::expected_shape(name,&t.dimensions,&[e[1],e[0]]));}let dt=dtype(t)?;Ok(TensorView::new(bytes(t,d)?,Box::leak(e.to_vec().into_boxed_slice()),dt))}
fn vector(g:&GgufFile,d:&[u8],name:&str,e:&[usize])->Result<Vec<f32>,TensorLoadError>{let t=find(g,name)?;if t.dimensions!=e.iter().map(|x|*x as u64).collect::<Vec<_>>(){return Err(TensorLoadError::expected_shape(name,&t.dimensions,e));}dequantize(t.ggml_type,&t.dimensions,bytes(t,d)?).map_err(|source|TensorLoadError::Decode{name:name.into(),source})}
fn find<'a>(g:&'a GgufFile,n:&str)->Result<&'a TensorInfo,TensorLoadError>{g.tensors.iter().find(|t|t.name==n).ok_or_else(||TensorLoadError::MissingTensor(n.into()))}
fn bytes<'a>(t:&TensorInfo,d:&'a[u8])->Result<&'a[u8],TensorLoadError>{let s=usize::try_from(t.file_offset).map_err(|_|TensorLoadError::OffsetTooLarge{name:t.name.clone(),value:t.file_offset})?;let n=usize::try_from(t.byte_len).map_err(|_|TensorLoadError::ByteLengthTooLarge{name:t.name.clone(),value:t.byte_len})?;let e=s.checked_add(n).ok_or(TensorLoadError::PayloadOutOfBounds{name:t.name.clone(),offset:s,end:usize::MAX,len:d.len()})?;d.get(s..e).ok_or(TensorLoadError::PayloadOutOfBounds{name:t.name.clone(),offset:s,end:e,len:d.len()})}
fn dtype(t:&TensorInfo)->Result<DType,TensorLoadError>{match GgmlType::from_raw(t.ggml_type){Some(GgmlType::F32)=>Ok(DType::F32),Some(GgmlType::F16)=>Ok(DType::F16),Some(GgmlType::Q4_0)=>Ok(DType::Q4_0),Some(GgmlType::Q8_0)=>Ok(DType::Q8_0),Some(GgmlType::Q4_K)=>Ok(DType::Q4K),Some(GgmlType::Q5_K)=>Ok(DType::Q5K),Some(GgmlType::Q6_K)=>Ok(DType::Q6K),_=>Err(TensorLoadError::Config(format!("unsupported GGML type {}",t.ggml_type)))}}
fn dtype_id(d:DType)->Option<u32>{match d{DType::F32=>Some(GgmlType::F32 as u32),DType::F16=>Some(GgmlType::F16 as u32),DType::Q4_0=>Some(GgmlType::Q4_0 as u32),DType::Q8_0=>Some(GgmlType::Q8_0 as u32),DType::Q4K=>Some(GgmlType::Q4_K as u32),DType::Q5K=>Some(GgmlType::Q5_K as u32),DType::Q6K=>Some(GgmlType::Q6_K as u32)}}
fn token_count(g:&GgufFile)->Result<usize,TensorLoadError>{let t=find(g,"token_embd.weight")?;usize::try_from(*t.dimensions.get(1).ok_or(TensorLoadError::Config("token embedding must be rank 2".into()))?).map_err(|_|TensorLoadError::Config("vocab exceeds usize".into()))}
fn u(g:&GgufFile,k:&'static str)->Result<usize,TensorLoadError>{let v=g.metadata_value(k).ok_or_else(||TensorLoadError::Config(format!("missing metadata {k}")))?;let n=match v{MetadataValue::Uint32(x)=>u64::from(*x),MetadataValue::Uint64(x)=>*x,MetadataValue::Int32(x)if *x>=0=>*x as u64,MetadataValue::Int64(x)if *x>=0=>*x as u64,_=>return Err(TensorLoadError::Config(format!("{k} is not an integer")))};usize::try_from(n).map_err(|_|TensorLoadError::Config(format!("{k} exceeds usize")))}
fn f(g:&GgufFile,k:&'static str)->Result<f32,TensorLoadError>{match g.metadata_value(k){Some(MetadataValue::Float32(x))=>Ok(*x),Some(MetadataValue::Float64(x))=>Ok(*x as f32),_=>Err(TensorLoadError::Config(format!("missing or invalid metadata {k}")))}}
fn bools(g:&GgufFile,k:&'static str)->Result<Vec<bool>,TensorLoadError>{let MetadataValue::Array{values,..}=g.metadata_value(k).ok_or_else(||TensorLoadError::Config(format!("missing metadata {k}")))?else{return Err(TensorLoadError::Config(format!("{k} is not an array")))};values.iter().map(|v|match v{MetadataValue::Bool(x)=>Ok(*x),_=>Err(TensorLoadError::Config(format!("{k} is not BOOL[]")))}).collect()}


#[cfg(test)]
mod recurrent_head_tests {
    use super::*;
    use qarnyx_cpu::CpuBackend;

    fn signal(length: usize, offset: usize) -> Vec<f32> {
        (0..length)
            .map(|index| {
                let value = ((index * 37 + offset * 19) % 251) as i32 - 125;
                value as f32 / 1_000.0
            })
            .collect()
    }

    fn serial_reference(
        q: &mut [f32],
        k: &mut [f32],
        v: &[f32],
        z: &[f32],
        beta: &[f32],
        alpha: &[f32],
        a: &[f32],
        dt: &[f32],
        state: &mut [f32],
        ssm_norm: &[f32],
        rms_epsilon: f32,
    ) -> Vec<f32> {
        for head in 0..LH {
            l2(&mut q[head * LD..(head + 1) * LD]);
            for value in &mut q[head * LD..(head + 1) * LD] {
                *value /= (LD as f32).sqrt();
            }
            l2(&mut k[head * LD..(head + 1) * LD]);

            let state_start = head * LD * LD;
            let head_state = &mut state[state_start..state_start + LD * LD];
            let key_head = &k[head * LD..(head + 1) * LD];
            let value_head = &v[head * LD..(head + 1) * LD];
            let decay = (-a[head].exp() * softplus(alpha[head] + dt[head])).exp();
            let beta_value = sigmoid(beta[head]);

            for column in 0..LD {
                for row in 0..LD {
                    head_state[row * LD + column] *= decay;
                }
            }

            let mut memory = [0.0_f32; LD];
            for column in 0..LD {
                for row in 0..LD {
                    memory[column] += head_state[row * LD + column] * key_head[row];
                }
            }

            for row in 0..LD {
                let delta = (value_head[row] - memory[row]) * beta_value;
                for column in 0..LD {
                    head_state[column * LD + row] += key_head[column] * delta;
                }
            }
        }

        let mut core = vec![0.0_f32; LIN];
        for head in 0..LH {
            let state_start = head * LD * LD;
            let head_state = &state[state_start..state_start + LD * LD];
            let query_head = &q[head * LD..(head + 1) * LD];
            let gate_head = &z[head * LD..(head + 1) * LD];
            let head_core = &mut core[head * LD..(head + 1) * LD];

            for column in 0..LD {
                for row in 0..LD {
                    head_core[column] += head_state[row * LD + column] * query_head[row];
                }
            }

            let mean_square = head_core
                .iter()
                .map(|value| f64::from(*value) * f64::from(*value))
                .sum::<f64>()
                / LD as f64;
            let rms = (mean_square + f64::from(rms_epsilon))
                .sqrt()
                .recip() as f32;
            for column in 0..LD {
                head_core[column] *= rms * ssm_norm[column] * silu(gate_head[column]);
            }
        }
        core
    }

    #[test]
    fn parallel_recurrent_heads_match_serial_reference_exactly() {
        let q = signal(LIN, 1);
        let k = signal(LIN, 2);
        let v = signal(LIN, 3);
        let z = signal(LIN, 4);
        let beta = (0..LH)
            .map(|head| 0.1 + head as f32 * 0.015)
            .collect::<Vec<_>>();
        let alpha = signal(LH, 5);
        let a = (0..LH)
            .map(|head| -0.4 - head as f32 * 0.015)
            .collect::<Vec<_>>();
        let dt = (0..LH)
            .map(|head| 0.02 + head as f32 * 0.001)
            .collect::<Vec<_>>();
        let ssm_norm = (0..LD)
            .map(|index| 0.75 + (index % 17) as f32 * 0.01)
            .collect::<Vec<_>>();
        let initial_state = signal(LH * LD * LD, 6);
        let rms_epsilon = 1.0e-5_f32;

        let mut expected_q = q.clone();
        let mut expected_k = k.clone();
        let mut expected_state = initial_state.clone();
        let expected_core = serial_reference(
            &mut expected_q,
            &mut expected_k,
            &v,
            &z,
            &beta,
            &alpha,
            &a,
            &dt,
            &mut expected_state,
            &ssm_norm,
            rms_epsilon,
        );

        for workers in [1, 4, 10] {
            let backend = CpuBackend::new(workers).expect("CPU worker pool");
            let mut actual_q = q.clone();
            let mut actual_k = k.clone();
            let mut actual_state = initial_state.clone();
            let actual_core = recurrent_heads(
                &backend,
                &mut actual_q,
                &mut actual_k,
                &v,
                &z,
                &beta,
                &alpha,
                &a,
                &dt,
                &mut actual_state,
                &ssm_norm,
                rms_epsilon,
            )
            .expect("parallel recurrent heads");

            assert_eq!(actual_q, expected_q, "q differs with {workers} workers");
            assert_eq!(actual_k, expected_k, "k differs with {workers} workers");
            assert_eq!(
                actual_state, expected_state,
                "recurrent state differs with {workers} workers"
            );
            assert_eq!(
                actual_core, expected_core,
                "head output differs with {workers} workers"
            );
        }
    }
}

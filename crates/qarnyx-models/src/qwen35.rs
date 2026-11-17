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

#[derive(Debug, Clone, Default)]
struct Qwen35Scratch {
    hidden_a: Vec<f32>,
    hidden_b: Vec<f32>,
    workspace: Qwen35Workspace,
}

#[derive(Debug, Clone, Default)]
struct Qwen35Workspace {
    norm: Vec<f32>,
    qkv: Vec<f32>,
    z: Vec<f32>,
    beta: Vec<f32>,
    alpha: Vec<f32>,
    conv_output: Vec<f32>,
    core: Vec<f32>,
    layer_hidden: Vec<f32>,
    ffn_gate: Vec<f32>,
    ffn_up: Vec<f32>,
    qg: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    q: Vec<f32>,
    gate: Vec<f32>,
    attention: Vec<f32>,
    scores: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Qwen35Cache {
    pub sequence_len: usize,
    recurrent: Vec<Vec<f32>>,
    conv: Vec<Vec<f32>>,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    scratch: Qwen35Scratch,
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
            scratch: Qwen35Scratch::default(),
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

        let Qwen35Cache {
            sequence_len,
            recurrent,
            conv,
            keys,
            values,
            scratch,
        } = cache;
        let Qwen35Scratch {
            hidden_a,
            hidden_b,
            workspace,
        } = scratch;

        ensure_buffer(hidden_a, H);
        hidden_a.copy_from_slice(input);
        ensure_buffer(hidden_b, H);

        for (index, layer) in self.layers.iter().enumerate() {
            let start = Instant::now();
            match layer {
                Qwen35Layer::Linear(weights) => linear(
                    b,
                    &self.config,
                    weights,
                    hidden_a,
                    hidden_b,
                    &mut recurrent[index],
                    &mut conv[index],
                    workspace,
                    profile,
                )?,
                Qwen35Layer::Attention(weights) => attention(
                    b,
                    &self.config,
                    weights,
                    hidden_a,
                    pos,
                    &mut keys[index],
                    &mut values[index],
                    hidden_b,
                    workspace,
                    profile,
                )?,
            }

            if let Some(profile) = profile.as_deref_mut() {
                profile.record(
                    if matches!(layer, Qwen35Layer::Linear(_)) {
                        "layer_linear_total"
                    } else {
                        "layer_attention_total"
                    },
                    start.elapsed(),
                );
            }
            std::mem::swap(hidden_a, hidden_b);
        }

        *sequence_len += 1;
        Ok(hidden_a.clone())
    }
}

fn linear<B: Backend>(
    b: &B,
    c: &Qwen35Config,
    w: &Qwen35Linear<'_>,
    input: &[f32],
    output: &mut [f32],
    state: &mut [f32],
    conv_state: &mut [f32],
    scratch: &mut Qwen35Workspace,
    profile: &mut Option<&mut Qwen35Profile>,
) -> Result<(), ModelError> {
    let Qwen35Workspace {
        norm,
        qkv,
        z,
        beta,
        alpha,
        conv_output,
        core,
        layer_hidden,
        ffn_gate,
        ffn_up,
        ..
    } = scratch;

    ensure_buffer(norm, H);
    rms0_into(input, w.attn_norm, c.rms_epsilon, norm)?;

    ensure_buffer(qkv, LIN * 3);
    ensure_buffer(z, LIN);
    ensure_buffer(beta, LH);
    ensure_buffer(alpha, LH);
    let projection_start = Instant::now();
    let weights = [w.qkv, w.gate, w.beta, w.alpha];
    let mut outputs: [&mut [f32]; 4] = [
        qkv.as_mut_slice(),
        z.as_mut_slice(),
        beta.as_mut_slice(),
        alpha.as_mut_slice(),
    ];
    b.matvec_batch(&weights, norm, &mut outputs)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.record("linear_projection_batch", projection_start.elapsed());
    }

    let conv_start = Instant::now();
    ensure_buffer(conv_output, LIN * 3);
    let mut kernel = [0.0_f32; CONV];
    // GGUF layout is q(2048), k(2048), v(2048) for the 16x128 linear heads.
    for channel in 0..qkv.len() {
        decode_into(w.conv, channel, &mut kernel)?;
        let base = channel * (CONV - 1);
        let current = qkv[channel];
        let value = kernel[0] * conv_state[base]
            + kernel[1] * conv_state[base + 1]
            + kernel[2] * conv_state[base + 2]
            + kernel[3] * current;
        conv_output[channel] = silu(value);
        conv_state[base] = conv_state[base + 1];
        conv_state[base + 1] = conv_state[base + 2];
        conv_state[base + 2] = current;
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.record("linear_conv_and_state", conv_start.elapsed());
    }

    let (q, rest) = conv_output.split_at_mut(LIN);
    let (k, v) = rest.split_at_mut(LIN);
    ensure_buffer(core, LIN);
    let recurrent_start = Instant::now();
    recurrent_heads(
        b,
        q,
        k,
        v,
        z,
        beta,
        alpha,
        w.a,
        w.dt,
        state,
        w.ssm_norm,
        c.rms_epsilon,
        core,
    )?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.record("linear_recurrent_heads", recurrent_start.elapsed());
    }

    ensure_buffer(layer_hidden, H);
    timed_matvec(b, w.out, core, layer_hidden, profile, "linear_out_matvec")?;
    add(layer_hidden, input);
    ffn(
        b,
        c,
        w.ffn_norm,
        w.ffn_gate,
        w.ffn_up,
        w.ffn_down,
        layer_hidden,
        output,
        norm,
        ffn_gate,
        ffn_up,
        profile,
    )
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
    core: &mut [f32],
) -> Result<(), ModelError> {
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
        || core.len() != LIN
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
    core.fill(0.0);
    b.parallel_for_each_mut(core, LD, &|head, head_core| {
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

    Ok(())
}

fn attention<B: Backend>(
    b: &B,
    c: &Qwen35Config,
    w: &Qwen35Attention<'_>,
    input: &[f32],
    pos: usize,
    keys: &mut Vec<f32>,
    values: &mut Vec<f32>,
    output: &mut [f32],
    scratch: &mut Qwen35Workspace,
    profile: &mut Option<&mut Qwen35Profile>,
) -> Result<(), ModelError> {
    let Qwen35Workspace {
        norm,
        qg,
        k,
        v,
        q,
        gate,
        attention,
        scores,
        layer_hidden,
        ffn_gate,
        ffn_up,
        ..
    } = scratch;

    ensure_buffer(norm, H);
    rms0_into(input, w.attn_norm, c.rms_epsilon, norm)?;
    ensure_buffer(qg, Q * 2);
    ensure_buffer(k, KV);
    ensure_buffer(v, KV);

    let projection_start = Instant::now();
    let weights = [w.q, w.k, w.v];
    let mut outputs: [&mut [f32]; 3] = [
        qg.as_mut_slice(),
        k.as_mut_slice(),
        v.as_mut_slice(),
    ];
    b.matvec_batch(&weights, norm, &mut outputs)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.record("attention_projection_batch", projection_start.elapsed());
    }

    ensure_buffer(q, Q);
    ensure_buffer(gate, Q);
    // GGUF q_proj stores each attention head as [query | gate], not
    // [all queries | all gates]. Deinterleave before applying q_norm/RoPE.
    for head in 0..FH {
        let source = head * FHD * 2;
        let destination = head * FHD;
        q[destination..destination + FHD]
            .copy_from_slice(&qg[source..source + FHD]);
        gate[destination..destination + FHD]
            .copy_from_slice(&qg[source + FHD..source + FHD * 2]);
        headnorm(&mut q[destination..destination + FHD], w.q_norm, c.rms_epsilon);
        rope(&mut q[destination..destination + ROPE], c.rope_theta, pos);
    }
    for head in 0..2 {
        headnorm(&mut k[head * FHD..(head + 1) * FHD], w.k_norm, c.rms_epsilon);
        rope(&mut k[head * FHD..head * FHD + ROPE], c.rope_theta, pos);
    }

    let attention_start = Instant::now();
    keys.extend_from_slice(k);
    values.extend_from_slice(v);
    let sequence_length = keys.len() / KV;
    ensure_buffer(attention, Q);
    attention.fill(0.0);
    ensure_buffer(scores, sequence_length);

    for query_head in 0..FH {
        let key_head = query_head / 4;
        let query_base = query_head * FHD;
        scores.fill(0.0);
        for token in 0..sequence_length {
            let key_base = token * KV + key_head * FHD;
            scores[token] = dot(
                &q[query_base..query_base + FHD],
                &keys[key_base..key_base + FHD],
            ) / (FHD as f32).sqrt();
        }
        softmax(scores);
        for token in 0..sequence_length {
            let value_base = token * KV + key_head * FHD;
            for index in 0..FHD {
                attention[query_base + index] +=
                    scores[token] * values[value_base + index];
            }
        }
    }
    for index in 0..Q {
        attention[index] *= sigmoid(gate[index]);
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.record("attention_core_softmax", attention_start.elapsed());
    }

    ensure_buffer(layer_hidden, H);
    timed_matvec(
        b,
        w.out,
        attention,
        layer_hidden,
        profile,
        "attention_out_matvec",
    )?;
    add(layer_hidden, input);
    ffn(
        b,
        c,
        w.ffn_norm,
        w.ffn_gate,
        w.ffn_up,
        w.ffn_down,
        layer_hidden,
        output,
        norm,
        ffn_gate,
        ffn_up,
        profile,
    )
}

fn ffn<B: Backend>(
    b: &B,
    c: &Qwen35Config,
    norm_weights: &[f32],
    gate: TensorView<'_>,
    up: TensorView<'_>,
    down: TensorView<'_>,
    input: &[f32],
    output: &mut [f32],
    norm_scratch: &mut Vec<f32>,
    gate_scratch: &mut Vec<f32>,
    up_scratch: &mut Vec<f32>,
    profile: &mut Option<&mut Qwen35Profile>,
) -> Result<(), ModelError> {
    ensure_buffer(norm_scratch, H);
    rms0_into(input, norm_weights, c.rms_epsilon, norm_scratch)?;
    ensure_buffer(gate_scratch, FF);
    ensure_buffer(up_scratch, FF);

    let projection_start = Instant::now();
    let weights = [gate, up];
    let mut outputs: [&mut [f32]; 2] = [
        gate_scratch.as_mut_slice(),
        up_scratch.as_mut_slice(),
    ];
    b.matvec_batch(&weights, norm_scratch, &mut outputs)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.record("ffn_projection_batch", projection_start.elapsed());
    }

    // Preserve the model reference's exact SiLU multiplication order while
    // combining gate/up in the reusable gate buffer.
    for (gate_value, up_value) in gate_scratch.iter_mut().zip(up_scratch.iter()) {
        *gate_value = silu(*gate_value) * *up_value;
    }
    timed_matvec(b, down, gate_scratch, output, profile, "ffn_down_matvec")?;
    add(output, input);
    Ok(())
}

fn ensure_buffer(buffer: &mut Vec<f32>, length: usize) {
    if buffer.len() != length {
        buffer.resize(length, 0.0);
    }
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

fn rms0(x: &[f32], w: &[f32], eps: f32) -> Result<Vec<f32>, ModelError> {
    let mut output = vec![0.0; x.len()];
    rms0_into(x, w, eps, &mut output)?;
    Ok(output)
}

fn rms0_into(
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
    output: &mut [f32],
) -> Result<(), ModelError> {
    if input.is_empty()
        || input.len() != weight.len()
        || output.len() != input.len()
    {
        return Err(ModelError::ShapeMismatch(
            "Qwen3.5 RMSNorm shape mismatch".into(),
        ));
    }
    // Qwen3.5 GGUF stores the effective RMSNorm scale. The +1 offset used
    // by the source model is folded into the GGUF norm weights, so runtime
    // inference must apply the weight directly (matching llama.cpp).
    let mean_square = input
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        / input.len() as f64;
    let inverse_rms = (mean_square + f64::from(epsilon)).sqrt().recip() as f32;
    for ((output_value, input_value), weight_value) in
        output.iter_mut().zip(input).zip(weight)
    {
        *output_value = input_value * inverse_rms * weight_value;
    }
    Ok(())
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
    fn qwen_workspace_buffers_reuse_allocated_capacity() {
        let mut buffer = Vec::with_capacity(32);
        ensure_buffer(&mut buffer, 16);
        buffer.fill(2.0);
        let capacity = buffer.capacity();
        ensure_buffer(&mut buffer, 16);
        assert_eq!(buffer.capacity(), capacity);
        assert_eq!(buffer, vec![2.0; 16]);

        let mut normalized = vec![0.0_f32; 3];
        rms0_into(&[3.0, 4.0, 0.0], &[1.0, 1.0, 1.0], 1.0e-5, &mut normalized)
            .unwrap();
        let expected_inverse = (25.0_f64 / 3.0 + 1.0e-5).sqrt().recip() as f32;
        assert_eq!(normalized, [3.0 * expected_inverse, 4.0 * expected_inverse, 0.0]);
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
            let mut actual_core = vec![0.0_f32; LIN];
            recurrent_heads(
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
                &mut actual_core,
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

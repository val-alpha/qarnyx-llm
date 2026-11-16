//! Qwen3.5-2B hybrid CPU reference model.
//! Implements the 24-layer 3:1 Gated DeltaNet / gated-attention text stack.

use qarnyx_core::{Backend, DType, TensorView};
use qarnyx_gguf::{GgufFile, MetadataValue, TensorInfo};
use qarnyx_quant::{dequantize, GgmlType};
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
            conv: (0..c.block_count).map(|_| vec![0.0; LIN * (CONV-1)]).collect(),
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

    pub fn forward_with_cache<B: Backend>(&self, b: &B, input: &[f32], pos: usize, cache: &mut Qwen35Cache)
        -> Result<Vec<f32>, ModelError> {
        if pos != cache.sequence_len || input.len() != H {
            return Err(ModelError::InvalidConfig("Qwen3.5 cache position/input shape mismatch"));
        }
        let mut h = input.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            h = match layer {
                Qwen35Layer::Linear(w) => linear(b, &self.config, w, &h, &mut cache.recurrent[i], &mut cache.conv[i])?,
                Qwen35Layer::Attention(w) => attention(b, &self.config, w, &h, &mut cache.keys[i], &mut cache.values[i])?,
            };
        }
        cache.sequence_len += 1;
        Ok(h)
    }
}

fn linear<B: Backend>(b: &B, c: &Qwen35Config, w: &Qwen35Linear<'_>, input: &[f32],
    state: &mut [f32], conv_state: &mut [f32]) -> Result<Vec<f32>, ModelError> {
    let x = rms0(input, w.attn_norm, c.rms_epsilon)?;
    let mut qkv = vec![0.0; LIN*3];
    let mut z = vec![0.0; LIN];
    let mut beta = vec![0.0; LH];
    let mut alpha = vec![0.0; LH];
    b.matvec(w.qkv, &x, &mut qkv)?;
    b.matvec(w.gate, &x, &mut z)?;
    b.matvec(w.beta, &x, &mut beta)?;
    b.matvec(w.alpha, &x, &mut alpha)?;

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

    let mut q = vec![0.0; LIN];
    let mut k = vec![0.0; LIN];
    let mut v = vec![0.0; LIN];
    q.copy_from_slice(&conv[..LIN]);
    k.copy_from_slice(&conv[LIN..2*LIN]);
    v.copy_from_slice(&conv[2*LIN..3*LIN]);
    for h in 0..LH {
        l2(&mut q[h*LD..(h+1)*LD]); l2(&mut k[h*LD..(h+1)*LD]);
        let sb = h*LD*LD;
        let decay = (-w.a[h].exp() * softplus(alpha[h] + w.dt[h])).exp();
        let beta = sigmoid(beta[h]);
        for j in 0..LD { for i in 0..LD { state[sb+i*LD+j] *= decay; } }
        let mut mem = [0.0f32; LD];
        for j in 0..LD {
            for i in 0..LD { mem[j] += state[sb+i*LD+j] * k[h*LD+i]; }
        }
        for i in 0..LD {
            let d = (v[h*LD+i] - mem[i]) * beta;
            for j in 0..LD { state[sb+j*LD+i] += k[h*LD+j] * d; }
        }
    }
    let mut core = vec![0.0; LIN];
    for h in 0..LH {
        let sb=h*LD*LD;
        for j in 0..LD { for i in 0..LD { core[h*LD+j] += state[sb+i*LD+j]*q[h*LD+i]; } }
        let sl=&mut core[h*LD..(h+1)*LD];
        let rms=(sl.iter().map(|x| f64::from(*x)*f64::from(*x)).sum::<f64>()/LD as f64+f64::from(c.rms_epsilon)).sqrt().recip() as f32;
        for j in 0..LD { sl[j] *= rms*w.ssm_norm[j]*silu(z[h*LD+j]); }
    }
    let mut y=vec![0.0;H]; b.matvec(w.out,&core,&mut y)?; add(&mut y,input);
    ffn(b,c,w.ffn_norm,w.ffn_gate,w.ffn_up,w.ffn_down,&y)
}

fn attention<B: Backend>(b:&B,c:&Qwen35Config,w:&Qwen35Attention<'_>,input:&[f32],
    keys:&mut Vec<f32>,values:&mut Vec<f32>) -> Result<Vec<f32>,ModelError> {
    let x=rms0(input,w.attn_norm,c.rms_epsilon)?;
    let mut qg=vec![0.0;Q*2]; let mut k=vec![0.0;KV]; let mut v=vec![0.0;KV];
    b.matvec(w.q,&x,&mut qg)?; b.matvec(w.k,&x,&mut k)?; b.matvec(w.v,&x,&mut v)?;
    let mut q=qg[..Q].to_vec();
    for h in 0..FH { headnorm(&mut q[h*FHD..(h+1)*FHD],w.q_norm,c.rms_epsilon); rope(&mut q[h*FHD..h*FHD+ROPE],c.rope_theta,0); }
    for h in 0..2 { headnorm(&mut k[h*FHD..(h+1)*FHD],w.k_norm,c.rms_epsilon); rope(&mut k[h*FHD..h*FHD+ROPE],c.rope_theta,0); }
    keys.extend_from_slice(&k); values.extend_from_slice(&v);
    let seq=keys.len()/KV; let mut a=vec![0.0;Q];
    for qh in 0..FH {
        let kh=qh/4; let qb=qh*FHD; let mut s=vec![0.0;seq];
        for t in 0..seq { let kb=t*KV+kh*FHD; s[t]=dot(&q[qb..qb+FHD],&keys[kb..kb+FHD])/(FHD as f32).sqrt(); }
        softmax(&mut s);
        for t in 0..seq { let vb=t*KV+kh*FHD; for j in 0..FHD { a[qb+j]+=s[t]*values[vb+j]; } }
    }
    for i in 0..Q { a[i]*=sigmoid(qg[Q+i]); }
    let mut y=vec![0.0;H]; b.matvec(w.out,&a,&mut y)?; add(&mut y,input);
    ffn(b,c,w.ffn_norm,w.ffn_gate,w.ffn_up,w.ffn_down,&y)
}

fn ffn<B:Backend>(b:&B,c:&Qwen35Config,norm:&[f32],gate:TensorView<'_>,up:TensorView<'_>,down:TensorView<'_>,input:&[f32])->Result<Vec<f32>,ModelError>{
    let x=rms0(input,norm,c.rms_epsilon)?; let mut g=vec![0.0;FF]; let mut u=vec![0.0;FF];
    b.matvec(gate,&x,&mut g)?; b.matvec(up,&x,&mut u)?; for i in 0..FF { g[i]=silu(g[i])*u[i]; }
    let mut y=vec![0.0;H]; b.matvec(down,&g,&mut y)?; add(&mut y,input); Ok(y)
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
            conv:matrix(g,d,&format!("{p}.ssm_conv1d.weight"),&[LIN,CONV])?,
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
    let m=x.iter().map(|v|f64::from(*v)*f64::from(*v)).sum::<f64>()/x.len() as f64;
    let inv=(m+f64::from(eps)).sqrt().recip() as f32;
    Ok(x.iter().zip(w).map(|(a,b)|a*inv*(1.0+b)).collect())
}
fn headnorm(x:&mut[f32],w:&[f32],eps:f32){let m=x.iter().map(|v|f64::from(*v)*f64::from(*v)).sum::<f64>()/x.len() as f64;let inv=(m+f64::from(eps)).sqrt().recip() as f32;for(i,a)in x.iter_mut().enumerate(){*a*=inv*(1.0+w[i]);}}
fn l2(x:&mut[f32]){let s=x.iter().map(|v|f64::from(*v)*f64::from(*v)).sum::<f64>();let inv=(s+1e-6).sqrt().recip() as f32;for v in x{*v*=inv;}}
fn rope(x:&mut[f32],theta:f32,pos:usize){for i in 0..ROPE/2{let a=(pos as f32)*theta.powf(-((2*i)as f32)/ROPE as f32);let(c,s)=(a.cos(),a.sin());let u=x[i];let v=x[i+ROPE/2];x[i]=u*c-v*s;x[i+ROPE/2]=u*s+v*c;}}
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

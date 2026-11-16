use std::fs::File;
use std::io::BufReader;
use std::fs;
use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use qarnyx_gguf::GgufFile;
use qarnyx_models::{LlamaModel, Qwen35Model};
use qarnyx_runtime::{GenerationConfig, LlamaGenerator, Qwen35Generator};
use qarnyx_tokenizer::GgufTokenizer;

/// Qarnyx command-line interface.
#[derive(Debug, Parser)]
#[command(
    name = "qarnyx",
    version,
    about = "Rust-native CPU-first GGUF inference runtime"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Print the current implementation status.
    Status,
    /// Read GGUF metadata and the tensor directory without loading tensor weights.
    Inspect {
        /// Path to a GGUF v3 model file.
        model: PathBuf,
        /// Print every tensor descriptor (may produce a large amount of output).
        #[arg(long)]
        list_tensors: bool,
    },
    /// Run CPU autoregressive generation from a supported GGUF architecture.
    Generate {
        /// Path to a supported GGUF v3 model.
        model: PathBuf,
        /// Prompt text.
        prompt: String,
        /// Maximum number of generated tokens.
        #[arg(long, default_value_t = 128)]
        max_new_tokens: usize,
        /// Sampling temperature; zero selects greedy decoding.
        #[arg(long, default_value_t = 0.8)]
        temperature: f32,
        /// Top-k sampling cutoff; zero disables top-k.
        #[arg(long, default_value_t = 40)]
        top_k: usize,
        /// Nucleus sampling cutoff.
        #[arg(long, default_value_t = 0.95)]
        top_p: f32,
        /// Worker count for the CPU backend.
        #[arg(long, default_value_t = 1)]
        workers: usize,
        /// CPU kernel implementation: automatic SIMD dispatch or scalar reference.
        #[arg(long, value_enum, default_value = "auto")]
        cpu_kernel: CpuKernelMode,
        /// RNG seed.
        #[arg(long, default_value_t = 0x51_4e_52_59_58)]
        seed: u64,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CpuKernelMode {
    Auto,
    Scalar,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Command::Status) | None => {
            println!("Qarnyx 0.1.0 - Rust-native CPU-first GGUF inference runtime");
            println!("GGUF v3 parsing, quantized CPU kernels, experimental Qwen3.5 generation, and runtime AVX2/FMA dispatch are available.");
            println!("Use 'qarnyx generate <model.gguf> <prompt> --cpu-kernel auto|scalar' to select CPU kernels.");
        }
        Some(Command::Inspect {
            model,
            list_tensors,
        }) => inspect_model(&model, list_tensors)?,
        Some(Command::Generate {
            model,
            prompt,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            workers,
            cpu_kernel,
            seed,
        }) => generate_model(
            &model,
            &prompt,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            workers,
            cpu_kernel,
            seed,
        )?,
    }

    Ok(())
}

fn inspect_model(path: &std::path::Path, list_tensors: bool) -> anyhow::Result<()> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let gguf = GgufFile::parse_reader(&mut reader)?;

    println!("GGUF version:       {}", gguf.version);
    println!("File size:          {} bytes", gguf.file_size);
    println!("Metadata entries:   {}", gguf.metadata.len());
    println!("Tensor count:       {}", gguf.tensors.len());
    println!("Tensor alignment:   {} bytes", gguf.alignment);
    println!("Tensor data starts: byte {}", gguf.tensor_data_offset);

    if let Some(architecture) = gguf.metadata_string("general.architecture") {
        println!("Architecture:       {architecture}");
    }
    if let Some(name) = gguf.metadata_string("general.name") {
        println!("Model name:         {name}");
    }

    if list_tensors {
        println!();
        println!("Tensor directory:");
        for tensor in &gguf.tensors {
            println!(
                "  {:<36} shape={:?} ggml_type={} bytes={} offset={} file_offset={}",
                tensor.name, tensor.dimensions, tensor.ggml_type, tensor.byte_len, tensor.offset, tensor.file_offset
            );
        }
    } else if !gguf.tensors.is_empty() {
        println!();
        println!(
            "First tensor:        {} {:?} (ggml_type {}, {} bytes, file offset {})",
            gguf.tensors[0].name,
            gguf.tensors[0].dimensions,
            gguf.tensors[0].ggml_type,
            gguf.tensors[0].byte_len,
            gguf.tensors[0].file_offset
        );
        println!("Use --list-tensors to print all tensor descriptors.");
    }

    println!();
    println!("Tensor payloads were not read into memory.");
    Ok(())
}

fn generate_model(
    path: &std::path::Path,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    workers: usize,
    cpu_kernel: CpuKernelMode,
    seed: u64,
) -> anyhow::Result<()> {
    let data = fs::read(path)?;
    let gguf = GgufFile::parse(&data)?;
    let tokenizer = GgufTokenizer::from_gguf(&gguf)?;
    let backend = match cpu_kernel {
        CpuKernelMode::Auto => qarnyx_cpu::CpuBackend::new(workers)?,
        CpuKernelMode::Scalar => qarnyx_cpu::CpuBackend::new_scalar(workers)?,
    };
    let config = GenerationConfig {
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        seed,
    };

    match gguf.metadata_string("general.architecture") {
        Some("qwen35") => {
            let model = Qwen35Model::from_gguf(&gguf, &data)?;
            let mut generator = Qwen35Generator::new(seed);
            let result = generator.generate(&model, &tokenizer, &backend, prompt, config)?;
            println!("{}", result.text);
        }
        Some("llama") => {
            let model = LlamaModel::from_gguf(&gguf, &data)?;
            let mut generator = LlamaGenerator::new(seed);
            let result = generator.generate(&model, &tokenizer, &backend, prompt, config)?;
            println!("{}", result.text);
        }
        Some(other) => anyhow::bail!("unsupported model architecture: {other}"),
        None => anyhow::bail!("GGUF is missing general.architecture metadata"),
    }

    Ok(())
}

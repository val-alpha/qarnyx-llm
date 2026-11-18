use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use qarnyx_gguf::GgufFile;
use qarnyx_models::{LlamaModel, Qwen35Model};
use qarnyx_runtime::{
    CancellationToken, GenerationConfig, GenerationError, LlamaGenerator, Qwen35Generator,
};
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
        /// Maximum combined prompt and generated sequence length.
        #[arg(long, default_value_t = 4096)]
        max_context_tokens: usize,
        /// Maximum per-session cache/workspace budget in MiB.
        #[arg(long, default_value_t = 512)]
        max_cache_mib: usize,
        /// Maximum accounted model-file plus session-cache budget in MiB.
        #[arg(long, default_value_t = 8192)]
        max_memory_mib: usize,
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
        /// CPU kernel implementation: automatic SIMD, fused Q4_K / K-quants, or scalar reference.
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
    FusedQ4k,
    FusedQk,
    Scalar,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Command::Status) | None => {
            println!("Qarnyx 0.1.0 - Rust-native CPU-first GGUF inference runtime");
            println!("GGUF v3 parsing, quantized CPU kernels, experimental Qwen3.5 generation, and runtime AVX2/FMA dispatch are available.");
            println!("Use --cpu-kernel auto|fused-q4k|fused-qk|scalar to select CPU kernels.");
        }
        Some(Command::Inspect {
            model,
            list_tensors,
        }) => inspect_model(&model, list_tensors)?,
        Some(Command::Generate {
            model,
            prompt,
            max_new_tokens,
            max_context_tokens,
            max_cache_mib,
            max_memory_mib,
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
            max_context_tokens,
            max_cache_mib,
            max_memory_mib,
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

fn model_file_budget_bytes(
    max_memory_mib: usize,
    max_cache_mib: usize,
) -> anyhow::Result<usize> {
    const MIB: usize = 1024 * 1024;

    if max_memory_mib == 0 || max_cache_mib == 0 {
        anyhow::bail!("--max-memory-mib and --max-cache-mib must be non-zero");
    }

    let total_budget_bytes = max_memory_mib
        .checked_mul(MIB)
        .ok_or_else(|| anyhow::anyhow!("--max-memory-mib does not fit in bytes"))?;
    let cache_budget_bytes = max_cache_mib
        .checked_mul(MIB)
        .ok_or_else(|| anyhow::anyhow!("--max-cache-mib does not fit in bytes"))?;

    total_budget_bytes
        .checked_sub(cache_budget_bytes)
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "--max-memory-mib must be greater than --max-cache-mib so the model file has a non-zero budget"
            )
        })
}

fn read_model_with_memory_budget(
    path: &std::path::Path,
    max_memory_mib: usize,
    max_cache_mib: usize,
) -> anyhow::Result<Vec<u8>> {
    let model_budget_bytes = model_file_budget_bytes(max_memory_mib, max_cache_mib)?;
    let model_budget_u64 = u64::try_from(model_budget_bytes)?;
    let mut file = File::open(path)?;
    let expected_size = file.metadata()?.len();

    if expected_size > model_budget_u64 {
        anyhow::bail!(
            "model file is {expected_size} bytes; --max-memory-mib {max_memory_mib} with --max-cache-mib {max_cache_mib} leaves {model_budget_bytes} bytes for model data"
        );
    }

    // Allocate only the size checked above, then verify EOF to detect a file
    // that changes size between metadata inspection and the read.
    let expected_len = usize::try_from(expected_size)?;
    let mut data = Vec::new();
    data.try_reserve_exact(expected_len)
        .map_err(|error| anyhow::anyhow!("could not reserve model file buffer: {error}"))?;
    data.resize(expected_len, 0);
    file.read_exact(&mut data)?;

    let mut extra = [0_u8; 1];
    if file.read(&mut extra)? != 0 {
        anyhow::bail!("model file size changed while it was being read");
    }

    Ok(data)
}

fn generate_model(
    path: &std::path::Path,
    prompt: &str,
    max_new_tokens: usize,
    max_context_tokens: usize,
    max_cache_mib: usize,
    max_memory_mib: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    workers: usize,
    cpu_kernel: CpuKernelMode,
    seed: u64,
) -> anyhow::Result<()> {
    let cancellation = CancellationToken::new();
    let signal_cancellation = cancellation.clone();
    ctrlc::set_handler(move || signal_cancellation.cancel())?;

    let data = read_model_with_memory_budget(path, max_memory_mib, max_cache_mib)?;
    let gguf = GgufFile::parse(&data)?;
    let tokenizer = GgufTokenizer::from_gguf(&gguf)?;
    let backend = match cpu_kernel {
        CpuKernelMode::Auto => qarnyx_cpu::CpuBackend::new(workers)?,
        CpuKernelMode::FusedQ4k => qarnyx_cpu::CpuBackend::new_fused_q4k(workers)?,
        CpuKernelMode::FusedQk => qarnyx_cpu::CpuBackend::new_fused_qk(workers)?,
        CpuKernelMode::Scalar => qarnyx_cpu::CpuBackend::new_scalar(workers)?,
    };
    let config = GenerationConfig {
        max_new_tokens,
        max_context_tokens,
        max_cache_mib,
        temperature,
        top_k,
        top_p,
        seed,
    };

    eprintln!("Press Ctrl+C to cancel generation cleanly.");
    match gguf.metadata_string("general.architecture") {
        Some("qwen35") => {
            let model = Qwen35Model::from_gguf(&gguf, &data)?;
            let mut generator = Qwen35Generator::new(seed);
            let prompt_tokens = tokenizer.encode_qwen35_chat_prompt(prompt)?;
            let result = match generator.generate_ids_with_cancellation(
                &model,
                &tokenizer,
                &backend,
                &prompt_tokens,
                config,
                &cancellation,
            ) {
                Ok(result) => result,
                Err(GenerationError::Cancelled) => {
                    eprintln!("\nGeneration cancelled; session cache released.");
                    return Ok(());
                }
                Err(error) => return Err(error.into()),
            };
            let generated_ids = &result.token_ids[prompt_tokens.len()..];
            let generated_text = tokenizer.decode(generated_ids)?;
            println!("{generated_text}");
        }
        Some("llama") => {
            let model = LlamaModel::from_gguf(&gguf, &data)?;
            let mut generator = LlamaGenerator::new(seed);
            let result = match generator.generate_with_cancellation(
                &model,
                &tokenizer,
                &backend,
                prompt,
                config,
                &cancellation,
            ) {
                Ok(result) => result,
                Err(GenerationError::Cancelled) => {
                    eprintln!("\nGeneration cancelled; session cache released.");
                    return Ok(());
                }
                Err(error) => return Err(error.into()),
            };
            println!("{}", result.text);
        }
        Some(other) => anyhow::bail!("unsupported model architecture: {other}"),
        None => anyhow::bail!("GGUF is missing general.architecture metadata"),
    }

    Ok(())
}

#[cfg(test)]
mod memory_budget_tests {
    use super::*;

    #[test]
    fn model_budget_reserves_the_configured_session_cache() {
        assert_eq!(
            model_file_budget_bytes(8192, 512).unwrap(),
            (8192 - 512) * 1024 * 1024
        );
    }

    #[test]
    fn rejects_a_cache_reservation_that_consumes_the_total_budget() {
        assert!(model_file_budget_bytes(512, 512).is_err());
        assert!(model_file_budget_bytes(128, 512).is_err());
    }

    #[test]
    fn rejects_zero_and_overflowing_budgets() {
        assert!(model_file_budget_bytes(0, 1).is_err());
        assert!(model_file_budget_bytes(8192, 0).is_err());
        assert!(model_file_budget_bytes(usize::MAX, 1).is_err());
    }
}

use std::fs::File;
use std::io::{self, BufReader, Read, Write};
use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use qarnyx_gguf::GgufFile;
use qarnyx_models::{LlamaModel, Qwen35Model};
use qarnyx_runtime::{
    CancellationToken, GenerationConfig, GenerationError, GenerationEvent, LlamaGenerator,
    Qwen35Generator,
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
        /// Enable Qwen3.5 reasoning output (disabled by default).
        #[arg(long, action = clap::ArgAction::SetTrue)]
        thinking: bool,
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
        /// CPU kernel implementation: automatic dispatch or explicit fused K-quant AVX2/FMA.
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
    FusedKqFma,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Command::Status) | None => {
            println!("Qarnyx 0.1.0 - Rust-native CPU-first GGUF inference runtime");
            println!(
                "GGUF v3 parsing, quantized CPU kernels, experimental Qwen3.5 generation, and runtime AVX2/FMA dispatch are available."
            );
            println!("Use --cpu-kernel auto|fused-kq-fma to select CPU kernels.");
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
            thinking,
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
            thinking,
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
                tensor.name,
                tensor.dimensions,
                tensor.ggml_type,
                tensor.byte_len,
                tensor.offset,
                tensor.file_offset
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

fn model_file_budget_bytes(max_memory_mib: usize, max_cache_mib: usize) -> anyhow::Result<usize> {
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
    thinking: bool,
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
        CpuKernelMode::FusedKqFma => qarnyx_cpu::CpuBackend::new_fused_kq_fma(workers)?,
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
            let prompt_tokens = tokenizer.encode_qwen35_chat_prompt(prompt, thinking)?;
            run_streaming_generation(
                &cancellation,
                thinking.then_some("<think>\n"),
                |cancellation, on_event| {
                    generator
                        .generate_ids_with_events(
                            &model,
                            &tokenizer,
                            &backend,
                            &prompt_tokens,
                            config,
                            cancellation,
                            on_event,
                        )
                        .map(|_| ())
                },
            )?;
        }
        Some("llama") => {
            if thinking {
                anyhow::bail!("--thinking is currently supported only for Qwen3.5");
            }
            let model = LlamaModel::from_gguf(&gguf, &data)?;
            let mut generator = LlamaGenerator::new(seed);
            run_streaming_generation(&cancellation, None, |cancellation, on_event| {
                generator
                    .generate_with_events(
                        &model,
                        &tokenizer,
                        &backend,
                        prompt,
                        config,
                        cancellation,
                        on_event,
                    )
                    .map(|_| ())
            })?;
        }
        Some(other) => anyhow::bail!("unsupported model architecture: {other}"),
        None => anyhow::bail!("GGUF is missing general.architecture metadata"),
    }

    Ok(())
}

/// Runs generation while writing each incremental text event to stdout.
///
/// The callback is synchronous, so flushing after each event makes decoded
/// text visible immediately instead of waiting for the full generation.
fn run_streaming_generation<F>(
    cancellation: &CancellationToken,
    prefix: Option<&str>,
    run: F,
) -> anyhow::Result<()>
where
    F: FnOnce(&CancellationToken, &mut dyn FnMut(GenerationEvent)) -> Result<(), GenerationError>,
{
    let stdout = std::io::stdout();
    let mut output = stdout.lock();

    if let Some(prefix) = prefix {
        output.write_all(prefix.as_bytes())?;
        output.flush()?;
    }

    let mut write_error = None;
    let result = {
        let mut on_event = |event| {
            if write_error.is_some() {
                return;
            }

            if let Err(error) = write_generation_event(&mut output, event) {
                write_error = Some(error);
                cancellation.cancel();
            }
        };

        run(cancellation, &mut on_event)
    };

    if let Some(error) = write_error {
        return Err(error.into());
    }

    match result {
        Ok(()) => {
            writeln!(output)?;
            output.flush()?;
            Ok(())
        }
        Err(GenerationError::Cancelled) => {
            writeln!(output)?;
            output.flush()?;
            eprintln!("Generation cancelled; session cache released.");
            Ok(())
        }
        Err(error) => {
            writeln!(output)?;
            output.flush()?;
            Err(error.into())
        }
    }
}

/// Writes one runtime event and flushes it so terminal output is incremental.
fn write_generation_event<W: Write>(writer: &mut W, event: GenerationEvent) -> io::Result<()> {
    match event {
        GenerationEvent::Token { text_delta, .. } => {
            writer.write_all(text_delta.as_bytes())?;
        }
        GenerationEvent::TextDelta { text } => {
            writer.write_all(text.as_bytes())?;
        }
    }
    writer.flush()
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

#[cfg(test)]
mod cli_tests {
    use super::*;

    #[test]
    fn generation_events_write_incremental_utf8_text() {
        let mut output = Vec::new();

        write_generation_event(
            &mut output,
            GenerationEvent::Token {
                token_id: 1,
                text_delta: "Halo ".to_owned(),
            },
        )
        .unwrap();
        write_generation_event(
            &mut output,
            GenerationEvent::Token {
                token_id: 2,
                text_delta: "dunia".to_owned(),
            },
        )
        .unwrap();
        write_generation_event(
            &mut output,
            GenerationEvent::TextDelta {
                text: " 🌏".to_owned(),
            },
        )
        .unwrap();

        assert_eq!(String::from_utf8(output).unwrap(), "Halo dunia 🌏");
    }

    #[test]
    fn thinking_is_disabled_by_default_and_can_be_enabled() {
        let default_cli =
            Cli::try_parse_from(["qarnyx", "generate", "model.gguf", "hello"]).unwrap();
        assert!(matches!(
            default_cli.command,
            Some(Command::Generate {
                thinking: false,
                ..
            })
        ));

        let thinking_cli =
            Cli::try_parse_from(["qarnyx", "generate", "model.gguf", "hello", "--thinking"])
                .unwrap();
        assert!(matches!(
            thinking_cli.command,
            Some(Command::Generate { thinking: true, .. })
        ));
    }
}

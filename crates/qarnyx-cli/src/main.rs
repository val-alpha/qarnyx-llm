use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use qarnyx_gguf::GgufFile;

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
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Command::Status) | None => {
            println!("Qarnyx 0.1.0 - GGUF v3 reader implemented");
            println!("Quantized tensor decoding and model inference are not implemented yet.");
        }
        Some(Command::Inspect {
            model,
            list_tensors,
        }) => inspect_model(&model, list_tensors)?,
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
                "  {:<36} shape={:?} ggml_type={} offset={} file_offset={}",
                tensor.name,
                tensor.dimensions,
                tensor.ggml_type,
                tensor.offset,
                tensor.file_offset
            );
        }
    } else if !gguf.tensors.is_empty() {
        println!();
        println!(
            "First tensor:        {} {:?} (ggml_type {}, file offset {})",
            gguf.tensors[0].name,
            gguf.tensors[0].dimensions,
            gguf.tensors[0].ggml_type,
            gguf.tensors[0].file_offset
        );
        println!("Use --list-tensors to print all tensor descriptors.");
    }

    println!();
    println!("Tensor payloads were not read into memory.");
    Ok(())
}

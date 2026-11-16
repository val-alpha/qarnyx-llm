use clap::{Parser, Subcommand};

/// Kernforge command-line interface.
#[derive(Debug, Parser)]
#[command(name = "kernforge", version, about = "Rust-native CPU-first GGUF inference runtime")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Print the current implementation status.
    Status,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Command::Status) | None => {
            println!("Kernforge 0.1.0 - early development scaffold");
            println!("GGUF parsing, tensor kernels, and inference are not implemented yet.");
        }
    }

    Ok(())
}

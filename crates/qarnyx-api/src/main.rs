use std::{
    net::{IpAddr, SocketAddr},
    path::PathBuf,
};

use clap::Parser;
use qarnyx_api::{ApiState, router};
use tokio::net::TcpListener;

#[derive(Debug, Parser)]
#[command(
    name = "qarnyx-server",
    version,
    about = "OpenAI-compatible HTTP server for the Qarnyx CPU inference runtime"
)]
struct Args {
    /// Path to a supported GGUF v3 model.
    #[arg(long)]
    model: PathBuf,

    /// Interface to bind. Defaults to loopback for local-only access.
    #[arg(long, default_value = "127.0.0.1")]
    host: IpAddr,

    /// TCP port for the HTTP API.
    #[arg(long, default_value_t = 8000)]
    port: u16,

    /// Number of persistent CPU worker threads.
    #[arg(long, default_value_t = 1)]
    workers: usize,

    /// Accounted model-file plus one-session-cache memory budget, in MiB.
    #[arg(long, default_value_t = 8192)]
    max_memory_mib: usize,

    /// Maximum combined prompt and completion token count.
    #[arg(long, default_value_t = 4096)]
    max_context_tokens: usize,

    /// Per-request KV/recurrent cache budget, in MiB.
    #[arg(long, default_value_t = 512)]
    max_cache_mib: usize,

    /// Maximum simultaneous inference requests. Defaults to one to bound cache memory.
    #[arg(long, default_value_t = 1)]
    max_concurrent_requests: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let state = ApiState::load(
        &args.model,
        args.workers,
        args.max_memory_mib,
        args.max_context_tokens,
        args.max_cache_mib,
        args.max_concurrent_requests,
    )?;

    let address = SocketAddr::from((args.host, args.port));
    let listener = TcpListener::bind(address).await?;
    println!(
        "Qarnyx API ready: model={} architecture={} address=http://{}",
        state.model_id(),
        state.architecture(),
        listener.local_addr()?
    );
    println!("Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions");

    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            eprintln!("failed to listen for Ctrl+C: {error}");
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                let _ = signal.recv().await;
            }
            Err(error) => eprintln!("failed to listen for SIGTERM: {error}"),
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

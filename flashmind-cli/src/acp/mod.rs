//! `flsh acp`: flsh as an Agent Client Protocol agent on stdio.
//!
//! Each ACP session is a flsh session working in the session's directory,
//! with the configured model and every tool. Its turns are saved like any
//! other session, so `flsh resume` and `session/resume` can continue it.

mod background;
mod history;
mod media;
mod open;
mod server;
mod steering;
mod system;
mod updates;
mod usage;

#[cfg(test)]
mod tests;

use agent_client_protocol::{ConnectTo, Stdio};
use anyhow::Context;

pub(crate) use open::FlshOpener;
use server::Server;

/// Serves ACP on stdin and stdout until the client closes stdin.
pub(crate) async fn run(opener: FlshOpener) -> anyhow::Result<()> {
    let server = Server::new(opener);
    let served = server.agent().connect_to(Stdio::new()).await;
    server.shutdown().await;
    served.context("the ACP connection failed")
}

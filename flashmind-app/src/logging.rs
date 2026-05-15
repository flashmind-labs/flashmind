//! Log initialisation with daily file rotation.

use tracing::Level;
use tracing_subscriber::filter::{LevelFilter, Targets};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::Layer;

use crate::config::AppConfig;

fn log_filter() -> Targets {
    Targets::new()
        .with_default(Level::DEBUG)
        .with_target("hyper_util", LevelFilter::OFF)
        .with_target("h2", LevelFilter::OFF)
        .with_target("rustls", LevelFilter::OFF)
}

/// Initialise the global tracing subscriber.
///
/// When `write_to_file` is true, logs are written to daily-rotated files under
/// `~/.flashmind/logs/<prefix>.log.YYYY-MM-DD`.  Otherwise logs go to stderr.
pub fn setup_logging(prefix: &str, write_to_file: bool) {
    let log_dir = AppConfig::log_dir();
    let _ = std::fs::create_dir_all(&log_dir);

    if write_to_file {
        let file_appender =
            tracing_appender::rolling::daily(&log_dir, format!("{prefix}.log"));
        let file_layer = tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_writer(file_appender)
            .with_filter(log_filter());

        tracing_subscriber::registry().with(file_layer).init();
    } else {
        let console_layer = tracing_subscriber::fmt::layer()
            .with_writer(std::io::stderr)
            .with_filter(log_filter());

        tracing_subscriber::registry().with(console_layer).init();
    }
}

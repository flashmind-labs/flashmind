//! Tools for Meshy features beyond text and image to 3D: retexture, remesh,
//! convert, resize, UV unwrap, image generation, 3D printing, Creative Lab,
//! the animation library and account information.
//!
//! Task tools report the Meshy task ID so later tools can chain with
//! `input_task_id` instead of uploading the model again.

mod files;
mod query;
mod specs;
mod task;

use std::path::PathBuf;
use std::sync::Arc;

use flashmind_llm::three_d::{ThreeDClient, ThreeDProvider};
use flashmind_types::ApiKey;

use crate::Tool;

pub use query::MeshyQueryTool;
pub use task::MeshyTaskTool;

/// Every Meshy tool for the configured key; empty without a Meshy key.
pub fn tools_from_keys(
    keys: impl Fn(ApiKey) -> Option<String>,
    output_dir: PathBuf,
) -> anyhow::Result<Vec<Arc<dyn Tool>>> {
    let Some(key) = keys(ApiKey::Meshy).filter(|key| !key.trim().is_empty()) else {
        return Ok(Vec::new());
    };
    let client = ThreeDClient::new(ThreeDProvider::Meshy, key)?;
    let mut tools: Vec<Arc<dyn Tool>> = Vec::new();
    for spec in specs::TASKS {
        tools.push(Arc::new(MeshyTaskTool::new(
            client.clone(),
            output_dir.clone(),
            spec,
        )));
    }
    for spec in specs::QUERIES {
        tools.push(Arc::new(MeshyQueryTool::new(client.clone(), spec)));
    }
    Ok(tools)
}

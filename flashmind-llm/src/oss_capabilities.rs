//! Open-source model capabilities registry.
//!
//! This module provides a const registry of open-source models (primarily vLLM-compatible)
//! with their detected capabilities. Use this for OpenAI-compatible endpoints serving
//! open-source models like Qwen, MiniMax, DeepSeek, Llama, etc.
//!
//! ## Usage
//!
//! ```ignore
//! use crate::llm::oss_capabilities::get_oss_capabilities;
//!
//! let caps = get_oss_capabilities("qwen/qwen3.5-27b-instruct");
//! assert!(caps.tool_calling);
//! ```

use flashmind_types::ModelCapabilities;

/// Registry of open-source model capabilities.
/// Keys are lowercase model identifiers (with or without organization prefix).
///
/// Capabilities are based on model architecture and known support:
/// - Tool calling: Based on whether the model supports function calling via chat templates
/// - Vision: For VL (vision-language) models
/// - Reasoning: For reasoning-focused models (e.g., DeepSeek-R1, Qwen3-32B)
pub static OSS_MODEL_CAPABILITIES: &[(&str, ModelCapabilities)] = &[
    // ============ Gemma 4 Series ============
    // Gemma 4 - Multimodal frontier models with native reasoning and tool calling
    // Official sizes: E2B (2B), E4B (4B), 26B MoE (activates 4B), 31B Dense
    (
        "google/gemma-4-e2b-it",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: true,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "google/gemma-4-e4b-it",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: true,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "google/gemma-4-26b-a4b-it",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "google/gemma-4-31b-it",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ Qwen3.6 Series ============
    // Qwen3.6-35B-A3B - First open-weight variant of Qwen3.6, MoE (35B total, 3B activated)
    // Vision-language model with native 262K context (extensible to 1M via YaRN)
    // Strong agentic coding, tool calling, thinking mode by default
    // Supports: images, documents (scanned PDFs), video
    (
        "qwen/qwen3.6-35b-a3b",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen3.6-35b-a3b-fp8",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ Qwen3.5 Series ============
    // Qwen3.5 base models - excellent tool calling support, up to 262K context
    // All Qwen3.5 models are VLMs with early fusion training
    // Support: images, documents (scanned PDFs), video
    (
        "qwen/qwen3.5-0.5b",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen3.5-1.8b",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen3.5-4b",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen3.5-7b",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen3.5-14b",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen3.5-27b",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen3.5-32b",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ Llama 4 Series ============
    // Llama 4 - Next-gen open models with native multimodal (text, image, video) support
    (
        "meta/llama-4-8b",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "meta/llama-4-70b",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "meta/llama-4-400b",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: true,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ DeepSeek V3.2 Series ============
    // DeepSeek V3.2 - High-efficiency reasoning models (NOT multimodal)
    (
        "deepseek/deepseek-v3.2-exp",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "deepseek/deepseek-v3.2",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen3.5-72b",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen3.5-122b",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen3.5-397b",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // Qwen3.5 Instruct variants - all support vision, docs, video
    (
        "qwen/qwen3.5-0.5b-instruct",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen3.5-1.8b-instruct",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen3.5-4b-instruct",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen3.5-7b-instruct",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen3.5-14b-instruct",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen3.5-27b-instruct",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen3.5-32b-instruct",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen3.5-72b-instruct",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen3.5-122b-instruct",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // Qwen3.5 Flash variants (fast versions) - also support vision, docs, video
    (
        "qwen/qwen3.5-flash-02-24",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen3.5-flash-08-24",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ Qwen3 Series (newer generation) ============
    (
        "qwen/qwen3-8b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen3-14b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen3-32b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen3-72b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ Qwen3 Coder Series ============
    // Code-focused Qwen3 models — text-only, no vision
    (
        "qwen/qwen3-coder",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // Qwen3-Coder-Next - 80B MoE model (3B active) with 256K context for agentic coding
    // Text-only: no vision, audio, or video support
    (
        "qwen/qwen3-coder-next",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen3-coder-next",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ Qwen2.5 VL Series (Vision) ============
    (
        "qwen/qwen2.5-vl-3b",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen2.5-vl-7b",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen2.5-vl-72b",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen2.5-vl-3b-instruct",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen2.5-vl-7b-instruct",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen2.5-vl-72b-instruct",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ Qwen2 VL Series ============
    (
        "qwen/qwen2-vl-2b",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: false,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen2-vl-7b",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: false,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen2-vl-72b",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: false,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ MiniMax Series ============
    // MiniMax-M2.7: Latest generation, excellent coding and agentic capabilities
    // MiniMax-M2.5: Excellent coding, agentic tool use, SWE-Bench verified
    (
        "minimaxai/minimax-m2.5",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "minimaxai/minimax-m2.5-fp8",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // MiniMax-M2.7
    (
        "minimaxai/minimax-m2.7",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // MiniMax-M2.1
    (
        "minimaxai/minimax-m2.1",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // MiniMax-M2
    (
        "minimaxai/minimax-m2",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // MiniMax-M1
    (
        "minimaxai/minimax-m1",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // MiniMax-Text series
    (
        "minimaxai/minimax-text-01",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ DeepSeek Series ============
    // DeepSeek V3 - 128K context, strong reasoning
    (
        "deepseek-ai/deepseek-v3",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "deepseek-ai/deepseek-v3.1",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "deepseek-ai/deepseek-v3.2",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // DeepSeek Coder V2
    (
        "deepseek-ai/deepseek-coder-v2",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "deepseek-ai/deepseek-coder-v2-lite",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // DeepSeek V2 series
    (
        "deepseek-ai/deepseek-v2",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "deepseek-ai/deepseek-v2-chat",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "deepseek-ai/deepseek-v2.5",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // DeepSeek LLM series
    (
        "deepseek-ai/deepseek-llm-67b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "deepseek-ai/deepseek-llm-67b-chat",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ Meta Llama Series ============
    // Llama 4 - Vision support
    (
        "meta-llama/llama-4-scout",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "meta-llama/llama-4-maverick",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // Llama 3.2 Vision
    (
        "meta-llama/llama-3.2-1b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "meta-llama/llama-3.2-3b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "meta-llama/llama-3.2-11b-vision-instruct",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "meta-llama/llama-3.2-90b-vision-instruct",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // Llama 3.1
    (
        "meta-llama/llama-3.1-8b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "meta-llama/llama-3.1-70b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "meta-llama/llama-3.1-405b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // Llama 3
    (
        "meta-llama/llama-3-8b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "meta-llama/llama-3-70b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ Mistral Series ============
    // Mistral Large
    (
        "mistralai/mistral-large",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "mistralai/mistral-large-2411",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // Mistral Nemo
    (
        "mistralai/mistral-nemo",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // Mixtral (mixture of experts)
    (
        "mistralai/mixtral-8x7b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "mistralai/mixtral-8x22b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // Mistral Small
    (
        "mistralai/mistral-small",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "mistralai/mistral-small-24b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ Cohere Series ============
    (
        "cohere/command-a",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "cohere/command-r",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "cohere/command-r-plus",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "cohere/command-r7b-12-2024",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ Google Gemma Series ============
    (
        "google/gemma-2-2b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "google/gemma-2-9b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "google/gemma-2-27b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "google/gemma-3-1b",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "google/gemma-3-4b",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "google/gemma-3-12b",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ AI21 Labs Jurassic Series ============
    (
        "ai21/jamba-1.5-large",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "ai21/jamba-1.5-mini",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ Snowflake Arctic ============
    (
        "snowflake/snowflake-arctic-instruct",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ ByteDance Series ============
    (
        "bytedance/豆包",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ Alibaba Qwen2.5 Code ============
    (
        "qwen/qwen2.5-coder-1.5b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen2.5-coder-7b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen2.5-coder-14b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen2.5-coder-32b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ Nous-Hermes Series ============
    (
        "nousresearch/hermes-3",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "nousresearch/hermes-2-pro",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ Fireworks AI Models ============
    (
        "fireworks/firefunction-v2",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ Abacus AI ============
    (
        "abacusai/smaug-72b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ 01.AI Yi Series ============
    (
        "01-ai/yi-1.5-6b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "01-ai/yi-1.5-9b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "01-ai/yi-1.5-34b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ Phi Series (Microsoft) ============
    (
        "microsoft/phi-4",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "microsoft/phi-4-mini",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "microsoft/phi-3.5-mini",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "microsoft/phi-3-medium",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ Falcon Series ============
    (
        "tiiuae/falcon3-10b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "tiiuae/falcon3-72b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "tiiuae/falcon-m2-10b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ OLMo Series (Allen AI) ============
    (
        "allenai/olmo-2-13b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "allenai/olmo-2-32b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ SmolVLM (IBM) ============
    (
        "ibm/smolvlm-256m",
        ModelCapabilities {
            tool_calling: false,
            images: true,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "ibm/smolvlm-1.7b",
        ModelCapabilities {
            tool_calling: false,
            images: true,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ NVLM (NVIDIA) ============
    (
        "nvidia/nvlm-1.0-72b",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ Aya Series (Cohere) ============
    (
        "cohere/aya-23-8b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "cohere/aya-23-35b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ DeepSeek R1 & V4 Series ============
    (
        "deepseek-ai/deepseek-r1",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "deepseek-ai/deepseek-v4-pro",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "deepseek-ai/deepseek-v4-flash",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "deepseek-ai/deepseek-v3.2-speciale",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "deepseek-ai/deepseek-ocr",
        ModelCapabilities {
            tool_calling: false,
            images: true,
            documents: true,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "deepseek-ai/deepseek-ocr-2",
        ModelCapabilities {
            tool_calling: false,
            images: true,
            documents: true,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "deepseek-ai/deepseek-math-v2",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ Qwen 3.6 Series ============
    (
        "qwen/qwen3.6-35b-a3b",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen3.6-27b",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "qwen/qwen3.5-9b",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ Mistral Medium & Small 4 Series ============
    (
        "mistralai/mistral-medium-3.5-128b",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "mistralai/mistral-small-4-119b",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "mistralai/devstral-small-2-24b-instruct",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "mistralai/devstral-2-123b-instruct",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ GLM Series (Zhipu AI) ============
    (
        "zai-org/glm-5.1",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "zai-org/glm-5",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "zai-org/glm-4.7",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "zai-org/glm-4.7-flash",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ Kimi / Moonshot Series ============
    (
        "moonshotai/kimi-k2.6",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "moonshotai/kimi-k2.5",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "moonshotai/kimi-k2-instruct",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "moonshotai/kimi-vl-a3b-thinking",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "moonshotai/kimi-vl-a3b-instruct",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "moonshotai/moonlight-16b-a3b-instruct",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ NVIDIA Nemotron Series ============
    (
        "nvidia/nemotron-3-nano-30b-a3b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "nvidia/nemotron-3-super-120b-a12b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "nvidia/llama-3.3-nemotron-super-49b-v1",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "nvidia/llama-3.1-nemotron-nano-8b-v1",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "nvidia/nemotron-nano-12b-v2-vl",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ OpenAI OSS Series ============
    (
        "openai/gpt-oss-120b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "openai/gpt-oss-20b",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ InternLM Series ============
    (
        "internlm/intern-s2-preview",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "internlm/intern-s1-pro",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "internlm/intern-s1",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ Xiaomi MiMo Series ============
    (
        "xiaomimimo/mimo-v2.5-pro",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "xiaomimimo/mimo-v2.5",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "xiaomimimo/mimo-v2-flash",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ OpenBMB MiniCPM Series ============
    (
        "openbmb/minicpm-v-4.6",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "openbmb/minicpm-v-4.6-thinking",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "openbmb/minicpm-o-4.5",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: true,
            reasoning: true,
            audio_output: true,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ Tencent Hunyuan Series ============
    (
        "tencent/hy3-preview",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ AIDC-AI Ovis Series ============
    (
        "aidc-ai/ovis2.6-80b-a3b",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ inclusionAI Ling Series ============
    (
        "inclusionai/ling-2.6-1t",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: true,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "inclusionai/ling-2.6-flash",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ Poolside Laguna Series ============
    (
        "poolside/laguna-xs.2",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ ByteDance Seed Series ============
    (
        "bytedance-seed/seed1.5-vl",
        ModelCapabilities {
            tool_calling: true,
            images: true,
            documents: true,
            video: true,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ Meta Llama 3.2 Text Instruct ============
    (
        "meta-llama/llama-3.2-1b-instruct",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    (
        "meta-llama/llama-3.2-3b-instruct",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ Google Gemma 3 Small ============
    (
        "google/gemma-3-270m",
        ModelCapabilities {
            tool_calling: false,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
    // ============ Zed Industries ============
    (
        "zed-industries/zeta-2.1",
        ModelCapabilities {
            tool_calling: true,
            images: false,
            documents: false,
            video: false,
            audio: false,
            reasoning: false,
            audio_output: false,
            image_generation: false,
            video_generation: false,
        },
    ),
];

/// Look up capabilities for an open-source model.
///
/// Matching is done case-insensitively and handles both full model IDs
/// (e.g., "Qwen/Qwen3.5-27B-Instruct") and bare model names (e.g., "qwen3.5-27b").
///
/// Returns `None` if the model is not in the registry.
pub fn get_oss_capabilities(model_id: &str) -> Option<ModelCapabilities> {
    let normalized = model_id.to_lowercase();

    // Try exact match first
    for (key, caps) in OSS_MODEL_CAPABILITIES {
        if normalized == key.to_lowercase() {
            return Some(*caps);
        }
    }

    // Try matching just the model name part (after last /)
    let model_name = normalized.rsplit('/').next().unwrap_or(&normalized);

    for (key, caps) in OSS_MODEL_CAPABILITIES {
        let key_name = key.rsplit('/').next().unwrap_or(key);
        if model_name == key_name.to_lowercase() {
            return Some(*caps);
        }
    }

    // Try partial match (model name contains the key)
    for (key, caps) in OSS_MODEL_CAPABILITIES {
        let key_name = key.rsplit('/').next().unwrap_or(key);
        if model_name.contains(&key_name.to_lowercase())
            || key_name.to_lowercase().contains(model_name)
        {
            return Some(*caps);
        }
    }

    None
}

/// Get capabilities with fallback defaults for unknown models.
///
/// If the model is not in the registry, returns a conservative default
/// assuming basic capabilities (tool calling enabled, no vision/reasoning).
pub fn get_oss_capabilities_or_default(model_id: &str) -> ModelCapabilities {
    get_oss_capabilities(model_id).unwrap_or(ModelCapabilities {
        tool_calling: true,
        images: false,
        documents: false,
        video: false,
        audio: false,
        reasoning: false,
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match() {
        let caps = get_oss_capabilities("qwen/qwen3.5-27b-instruct");
        assert!(caps.is_some());
        assert!(caps.unwrap().tool_calling);
    }

    #[test]
    fn test_bare_model_name() {
        let caps = get_oss_capabilities("qwen3.5-27b");
        assert!(caps.is_some());
    }

    #[test]
    fn test_case_insensitive() {
        let caps = get_oss_capabilities("QWEN/QWEN3.5-27B-INSTRUCT");
        assert!(caps.is_some());
    }

    #[test]
    fn test_unknown_model() {
        let caps = get_oss_capabilities("unknown/model-xyz");
        assert!(caps.is_none());
    }

    #[test]
    fn test_default_for_unknown() {
        let caps = get_oss_capabilities_or_default("unknown/model-xyz");
        assert!(caps.tool_calling);
        assert!(!caps.images);
    }

    #[test]
    fn test_vision_model() {
        let caps = get_oss_capabilities("qwen/qwen2.5-vl-7b-instruct").unwrap();
        assert!(caps.images);
        assert!(caps.video);
    }

    #[test]
    fn test_reasoning_model() {
        let caps = get_oss_capabilities("deepseek-ai/deepseek-v3").unwrap();
        assert!(caps.reasoning);
    }
}

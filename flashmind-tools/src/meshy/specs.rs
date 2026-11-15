//! Descriptions and JSON schemas for each Meshy tool.

use serde_json::{Map, Value, json};

use super::files::FileKind::{Image, Model};
use super::query::{Query, QuerySpec};
use super::task::{CREATIVE_LAB_PRODUCTS, Route, TaskSpec};

const FORMATS: [&str; 7] = ["glb", "fbx", "obj", "usdz", "blend", "stl", "3mf"];

pub(super) static TASKS: &[TaskSpec] = &[
    TaskSpec {
        name: "retexture_3d",
        description: "Re-texture a 3D model with Meshy from a text style prompt, a style image, or 1 to 4 view images. Give the model as input_task_id (a succeeded Meshy task) or model_url. Saves GLB unless target_formats says otherwise and reports the task ID. Paid Meshy plan required.",
        route: Route::Fixed("openapi/v1/retexture"),
        schema: retexture,
        files: &[
            ("model_url", Model),
            ("image_style_url", Image),
            ("multiview_image_urls", Image),
        ],
        default_glb: true,
        extras: &[],
    },
    TaskSpec {
        name: "remesh_3d",
        description: "Remesh a 3D model with Meshy to a target triangle or quad count. Give the model as input_task_id or model_url. Saves GLB unless target_formats says otherwise and reports the task ID.",
        route: Route::Fixed("openapi/v1/remesh"),
        schema: remesh,
        files: &[("model_url", Model)],
        default_glb: false,
        extras: &[],
    },
    TaskSpec {
        name: "convert_3d",
        description: "Convert a 3D model to other formats (GLB, FBX, OBJ, USDZ, Blender, STL, 3MF) with Meshy for 1 credit. Give the model as input_task_id or model_url.",
        route: Route::Fixed("openapi/v1/convert"),
        schema: convert,
        files: &[("model_url", Model)],
        default_glb: false,
        extras: &[],
    },
    TaskSpec {
        name: "resize_3d",
        description: "Scale a 3D model to a real-world size with Meshy for 1 credit: set resize_height, resize_longest_side (meters) or auto_size, and exactly one of them. Saves GLB.",
        route: Route::Fixed("openapi/v1/resize"),
        schema: resize,
        files: &[("model_url", Model)],
        default_glb: false,
        extras: &[],
    },
    TaskSpec {
        name: "uv_unwrap_3d",
        description: "Generate clean UVs for a GLB with Meshy for 5 credits. The mesh may have at most 40,000 faces. Give the model as input_task_id or a GLB model_url. Saves GLB.",
        route: Route::Fixed("openapi/v1/uv-unwrap"),
        schema: uv_unwrap,
        files: &[("model_url", Model)],
        default_glb: false,
        extras: &[],
    },
    TaskSpec {
        name: "meshy_text_to_image",
        description: "Generate an image from text with Meshy, optionally in a T or A pose or from several angles, as a reference for 3D generation. Saves the images and reports the task ID, which meshy_image_to_image accepts.",
        route: Route::Fixed("openapi/v1/text-to-image"),
        schema: text_to_image,
        files: &[],
        default_glb: false,
        extras: &[],
    },
    TaskSpec {
        name: "meshy_image_to_image",
        description: "Edit or transform images with Meshy from a prompt. Give 1 to 5 reference_image_urls (URLs or local images) or the input_task_id of an earlier Meshy image task. Saves the images and reports the task ID.",
        route: Route::Fixed("openapi/v1/image-to-image"),
        schema: image_to_image,
        files: &[("reference_image_urls", Image)],
        default_glb: false,
        extras: &[],
    },
    TaskSpec {
        name: "print_multicolor_3d",
        description: "Turn a textured model into a multi-color 3MF for a color 3D printer with Meshy. Give the model as input_task_id or a GLB or FBX model_url. Saves the 3MF.",
        route: Route::Fixed("openapi/v1/print/multi-color"),
        schema: multicolor,
        files: &[("model_url", Model)],
        default_glb: false,
        extras: &[],
    },
    TaskSpec {
        name: "analyze_printability_3d",
        description: "Check whether a 3D model can be printed with Meshy, for free: watertightness, holes, non-manifold edges and, if printer_type or a print size is given, thin walls. Reports the printability report.",
        route: Route::Fixed("openapi/v1/print/analyze"),
        schema: analyze,
        files: &[("model_url", Model)],
        default_glb: false,
        extras: &["printability"],
    },
    TaskSpec {
        name: "repair_printability_3d",
        description: "Repair a 3D model so it prints, with Meshy for 10 credits: closes holes and fixes topology, and thickens thin walls when printer_type is given. Saves the repaired model and reports the repair report.",
        route: Route::Fixed("openapi/v1/print/repair"),
        schema: repair,
        files: &[("model_url", Model)],
        default_glb: false,
        extras: &["print_repair_report", "intended_print"],
    },
    TaskSpec {
        name: "split_3d",
        description: "Split a Meshy-generated model into printable parts with Meshy for 10 credits, automatically, by named parts, or by color, optionally with connectors. input_task_id must be a Meshy 6 or 7 task. Saves GLB unless target_formats says otherwise.",
        route: Route::Fixed("openapi/v1/print/split"),
        schema: split,
        files: &[],
        default_glb: false,
        extras: &["part_count", "prompt_ignored"],
    },
    TaskSpec {
        name: "relief_3d",
        description: "Turn an image into a relief model (a plaque or engraving) with Meshy for 20 credits. Saves GLB, STL and the depth map. Set should_generate_model false for the depth map only.",
        route: Route::Fixed("openapi/v1/relief"),
        schema: relief,
        files: &[("image_url", Image)],
        default_glb: false,
        extras: &[],
    },
    TaskSpec {
        name: "creative_lab_3d",
        description: "Meshy Creative Lab: turn a photo into a printable product. Most products have two calls: stage prototype (image_url to a concept image; reports the task ID, and candidate_ids for keycap), then stage build (input_task_id of that prototype to 3D files). Omit stage for fidget-collapsible, which is one call. Products and extras: keychain (prototype name_text up to 10 characters; build options badge_shape, size_mm, relief_height_mm, base_thickness_mm), fridge-magnet (same build options), figure, vinyl-figure and brick-figure (no options), lamp (prototype image_subject character or landscape; build options diameter_mm, thickness_mm, cut_amount_percent, light_source_preset; output.format stl or zip), keycap (build needs candidate_id; options head_size_mm, vertical_offset_mm), fidget-pixel (prototype needs type person or other; build options shape, grid_size, piece_size_mm, color_count, piece_height_mm). Build options go in options and the file format in output.format. Paid Meshy plan required.",
        route: Route::CreativeLab,
        schema: creative_lab,
        files: &[("image_url", Image)],
        default_glb: false,
        extras: &["candidate_ids", "name_text"],
    },
];

pub(super) static QUERIES: &[QuerySpec] = &[
    QuerySpec {
        name: "animation_library_3d",
        description: "List Meshy's preset character animations for free, as `action_id name [category/sub_category]`. Pass action_ids from here to animate_3d. Categories: WalkAndRun, BodyMovements, DailyActions, Fighting, Dancing.",
        query: Query::AnimationLibrary,
        schema: animation_library,
    },
    QuerySpec {
        name: "meshy_account",
        description: "Read Meshy account data: kind balance for remaining credits, or kind usage for recent API tasks with credits spent (usage needs a Studio or Enterprise team).",
        query: Query::Account,
        schema: account,
    },
];

/// Merge object-valued schema fragments into one object schema.
fn object(parts: Vec<Value>, required: &[&str]) -> Value {
    let mut properties = Map::new();
    for part in parts {
        if let Value::Object(map) = part {
            properties.extend(map);
        }
    }
    json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false
    })
}

/// `input_task_id` and `model_url`, of which at least one is needed.
fn model_input(types: &str) -> Value {
    json!({
        "input_task_id": {"type": "string", "minLength": 1, "description": "ID of a succeeded Meshy task that made a model; every Meshy tool reports its task ID. Takes precedence over model_url."},
        "model_url": {"type": "string", "minLength": 1, "description": format!("{types}: an HTTP(S) URL, or a local path relative to the working directory (up to 100 MB). Use this or input_task_id.")}
    })
}

const ANY_MODEL: &str = "The model (GLB, glTF, OBJ, FBX or STL)";

fn formats(default: &str) -> Value {
    json!({"target_formats": {"type": "array", "items": {"type": "string", "enum": FORMATS}, "minItems": 1, "description": format!("Formats to produce and save. Default: {default}.")}})
}

fn retexture() -> Value {
    object(
        vec![
            model_input(ANY_MODEL),
            json!({
                "text_style_prompt": {"type": "string", "maxLength": 800, "description": "Describes the new texture. Use this, image_style_url or multiview_image_urls."},
                "image_style_url": {"type": "string", "description": "Style reference image: URL or local path (JPG, PNG or WebP, up to 20 MB)"},
                "multiview_image_urls": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 4, "description": "1 to 4 images of the object, front view first. Needs ai_model meshy-7 or meshy-n1."},
                "ai_model": {"type": "string", "enum": ["meshy-6-lite", "meshy-6", "meshy-7", "meshy-n1", "latest"], "default": "latest"},
                "enable_original_uv": {"type": "boolean", "default": false, "description": "Keep the model's UV layout"},
                "enable_pbr": {"type": "boolean", "default": false, "description": "Also generate metallic, roughness and normal maps"},
                "texture_resolution": {"type": "string", "enum": ["2k", "4k", "8k"], "default": "2k", "description": "4k and 8k need meshy-6, meshy-7 or latest"},
                "remove_lighting": {"type": "boolean", "default": true, "description": "Remove baked highlights and shadows (meshy-6 only)"}
            }),
            formats("glb"),
        ],
        &[],
    )
}

fn remesh() -> Value {
    object(
        vec![
            model_input(ANY_MODEL),
            formats("glb"),
            json!({
                "topology": {"type": "string", "enum": ["quad", "triangle"], "default": "triangle"},
                "target_polycount": {"type": "integer", "minimum": 100, "maximum": 300000, "default": 30000},
                "decimation_mode": {"type": "integer", "enum": [1, 2, 3, 4], "description": "1 ultra, 2 high, 3 medium, 4 low. Overrides target_polycount."}
            }),
        ],
        &[],
    )
}

fn convert() -> Value {
    object(
        vec![
            model_input(ANY_MODEL),
            json!({"target_formats": {"type": "array", "items": {"type": "string", "enum": FORMATS}, "minItems": 1, "description": "Formats to produce and save"}}),
        ],
        &["target_formats"],
    )
}

fn resize() -> Value {
    object(
        vec![
            model_input(ANY_MODEL),
            json!({
                "resize_height": {"type": "number", "exclusiveMinimum": 0, "description": "New height in meters"},
                "resize_longest_side": {"type": "number", "exclusiveMinimum": 0, "description": "New longest side in meters, keeping proportions"},
                "auto_size": {"type": "boolean", "description": "Estimate the real-world height with AI"},
                "origin_at": {"type": "string", "enum": ["bottom", "center"], "default": "bottom"}
            }),
        ],
        &[],
    )
}

fn uv_unwrap() -> Value {
    object(vec![model_input("The GLB model")], &[])
}

fn text_to_image() -> Value {
    object(
        vec![json!({
            "ai_model": {"type": "string", "enum": ["nano-banana", "nano-banana-2", "nano-banana-2-lite", "nano-banana-2-1", "nano-banana-pro", "gpt-image-2", "gpt-image-2-5-flare", "gpt-image-2-5-sunburst", "gpt-image-2-6-flare", "gpt-image-2-6-sunburst"], "description": "nano-banana-pro needs a Pro plan"},
            "prompt": {"type": "string", "minLength": 1},
            "generate_multi_view": {"type": "boolean", "default": false, "description": "Several angles of the subject; cannot combine with aspect_ratio"},
            "pose_mode": {"type": "string", "enum": ["a-pose", "t-pose"], "description": "Fixes the output to 3:4 on nano-banana models"},
            "aspect_ratio": {"type": "string", "enum": ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"], "default": "1:1", "description": "3:2 and 2:3 only on gpt-image models"},
            "remove_background": {"type": "boolean", "default": false}
        })],
        &["ai_model", "prompt"],
    )
}

fn image_to_image() -> Value {
    object(
        vec![json!({
            "ai_model": {"type": "string", "enum": ["nano-banana", "nano-banana-2", "nano-banana-2-lite", "nano-banana-2-1", "nano-banana-pro", "gpt-image-2", "gpt-image-2-5-flare", "gpt-image-2-5-sunburst", "gpt-image-2-6-flare", "gpt-image-2-6-sunburst"], "description": "nano-banana-pro needs a Pro plan"},
            "prompt": {"type": "string", "minLength": 1, "description": "The edit or transformation to apply"},
            "reference_image_urls": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5, "description": "1 to 5 images: URLs or local paths (JPG, PNG or WebP, up to 20 MB each)"},
            "input_task_id": {"type": "string", "minLength": 1, "description": "A succeeded Meshy text-to-image or image-to-image task; takes precedence over reference_image_urls"},
            "generate_multi_view": {"type": "boolean", "default": false},
            "aspect_ratio": {"type": "string", "enum": ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"], "default": "1:1"},
            "remove_background": {"type": "boolean", "default": false}
        })],
        &["ai_model", "prompt"],
    )
}

fn multicolor() -> Value {
    object(
        vec![
            model_input("A GLB or FBX model"),
            json!({
                "max_colors": {"type": "integer", "minimum": 1, "maximum": 16, "default": 4},
                "style": {"type": "string", "enum": ["realistic", "cartoon"], "default": "realistic", "description": "realistic needs a base color texture; cartoon also accepts vertex colors"},
                "max_depth": {"type": "integer", "minimum": 3, "maximum": 6, "default": 4, "description": "Realistic style only"},
                "printer_brand": {"type": "string", "enum": ["bambu", "anycubic", "creality", "elegoo", "flashforge", "prusa", "qidi", "snapmaker", "ankermake"], "default": "bambu", "description": "Slicer preset in the 3MF"}
            }),
        ],
        &[],
    )
}

/// Printer and size fields shared by analysis and repair.
fn print_size() -> Value {
    json!({
        "printer_type": {"type": "string", "enum": ["fdm", "sla", "full_color"], "description": "Enables the wall thickness check (minimum 1.0, 0.8 and 2.0 mm)"},
        "intended_print_height_mm": {"type": "number", "minimum": 1, "maximum": 1000, "description": "GLB input only; not with intended_print_longest_side_mm"},
        "intended_print_longest_side_mm": {"type": "number", "minimum": 1, "maximum": 1000}
    })
}

fn analyze() -> Value {
    object(vec![model_input(ANY_MODEL), print_size()], &[])
}

fn repair() -> Value {
    object(vec![model_input(ANY_MODEL), print_size()], &[])
}

fn split() -> Value {
    object(
        vec![json!({
            "input_task_id": {"type": "string", "minLength": 1, "description": "A succeeded Meshy 6 or 7 task with at most 8,000,000 faces"},
            "mode": {"type": "string", "enum": ["auto", "by_parts", "by_color"], "default": "auto", "description": "by_color needs an image-to-3D task"},
            "prompt": {"type": "string", "maxLength": 600, "description": "Names 1 to 10 parts; required for by_parts and by_color"},
            "target_formats": {"type": "array", "items": {"type": "string", "enum": FORMATS}, "description": "Default glb; glb is always produced"},
            "layout": {"type": "string", "enum": ["assembled", "on_plate"], "default": "assembled"},
            "keep_texture": {"type": "boolean", "default": false, "description": "Keep the texture on each part; takes twice as long"},
            "connectors": {"type": "boolean", "default": false, "description": "Add mortise-and-tenon connectors at each cut"},
            "connector_type": {"type": "string", "enum": ["cube", "cylinder"], "default": "cube"},
            "connector_size": {"type": "number", "minimum": 0.1, "maximum": 0.8, "default": 0.5},
            "connector_height": {"type": "number", "minimum": 0.1, "maximum": 0.8, "default": 0.1}
        })],
        &["input_task_id"],
    )
}

fn relief() -> Value {
    object(
        vec![json!({
            "image_url": {"type": "string", "minLength": 1, "description": "The image: URL or local path (JPG, PNG or WebP, up to 20 MB)"},
            "variant": {"type": "string", "enum": ["high", "low"], "default": "low", "description": "high is deep, for printing; low is shallow, for engraving or CNC"},
            "should_generate_model": {"type": "boolean", "default": true, "description": "false returns only the depth map"},
            "size_mm": {"type": "number", "minimum": 20, "maximum": 250, "default": 60, "description": "Length of the longer side"},
            "carving_mode": {"type": "string", "enum": ["emboss", "engrave"], "default": "emboss"},
            "carving_depth_mm": {"type": "number", "minimum": 0, "maximum": 60, "description": "Default 6 for high, 3 for low"},
            "remove_base": {"type": "boolean", "default": false, "description": "Cut away the background and build no base plate"},
            "removal_threshold": {"type": "number", "minimum": 0.01, "maximum": 1, "default": 0.01, "description": "Only with remove_base"},
            "base_thickness_mm": {"type": "number", "minimum": 0.1, "maximum": 10, "default": 1},
            "bend_mode": {"type": "string", "enum": ["off", "convex", "concave"], "default": "off"},
            "bend_axis": {"type": "string", "enum": ["xz", "yz"], "default": "xz"},
            "bend_angle": {"type": "number", "minimum": 0, "maximum": 360, "default": 60},
            "mesh_resolution": {"type": "integer", "minimum": 512, "maximum": 2048, "default": 1024}
        })],
        &["image_url"],
    )
}

fn creative_lab() -> Value {
    object(
        vec![json!({
            "product": {"type": "string", "enum": CREATIVE_LAB_PRODUCTS},
            "stage": {"type": "string", "enum": ["prototype", "build"], "description": "Omit for fidget-collapsible"},
            "image_url": {"type": "string", "description": "Prototype stage, and fidget-collapsible: URL or local path (JPG, PNG or WebP, up to 20 MB)"},
            "input_task_id": {"type": "string", "description": "Build stage: a succeeded prototype task of the same product made through this API"},
            "name": {"type": "string", "maxLength": 100, "description": "Dashboard label, not printed"},
            "name_text": {"type": "string", "maxLength": 10, "description": "Keychain prototype: text printed on the keychain"},
            "remove_background": {"type": "boolean", "description": "Prototype stage: return a transparent concept image"},
            "image_subject": {"type": "string", "enum": ["character", "landscape"], "description": "Lamp prototype"},
            "type": {"type": "string", "enum": ["person", "other"], "description": "Fidget-pixel prototype"},
            "candidate_id": {"type": "string", "description": "Keycap build: one of the prototype's candidate_ids"},
            "options": {"type": "object", "description": "Build options for the product, such as size_mm or badge_shape"},
            "output": {"type": "object", "description": "Build output, such as {\"format\": \"glb\"}"}
        })],
        &["product"],
    )
}

fn animation_library() -> Value {
    object(
        vec![json!({
            "search": {"type": "string", "description": "Case-insensitive match on name or key"},
            "category": {"type": "string", "enum": ["WalkAndRun", "BodyMovements", "DailyActions", "Fighting", "Dancing"]},
            "sub_category": {"type": "string"},
            "action_ids": {"type": "array", "items": {"type": "integer"}, "maxItems": 200}
        })],
        &[],
    )
}

fn account() -> Value {
    object(
        vec![json!({
            "kind": {"type": "string", "enum": ["balance", "usage"]},
            "page_num": {"type": "integer", "minimum": 1, "maximum": 255, "description": "usage only"},
            "page_size": {"type": "integer", "minimum": 1, "maximum": 100, "description": "usage only"},
            "sort_by": {"type": "string", "enum": ["+created_at", "-created_at"], "description": "usage only; default newest first"},
            "start_time": {"type": "string", "description": "usage only: RFC 3339, default 30 days before end_time"},
            "end_time": {"type": "string", "description": "usage only: RFC 3339, default now"},
            "endpoints": {"type": "array", "items": {"type": "string"}, "description": "usage only: endpoint names such as image-to-3d or rig"},
            "status": {"type": "string", "enum": ["SUCCEEDED", "FAILED"], "description": "usage only"}
        })],
        &["kind"],
    )
}

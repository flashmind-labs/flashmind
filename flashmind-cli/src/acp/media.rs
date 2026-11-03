//! Images from tools and replies as ACP content, so the client can show them.

use std::path::{Path, PathBuf};

use agent_client_protocol::schema::v1::{
    ContentBlock, ImageContent, ResourceLink, ToolCallContent, ToolCallLocation,
};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use flashmind_types::ContentPart;
use flashmind_types::llm::mime_from_extension;
use url::Url;

/// The largest image sent inline, in decoded bytes. A larger one on disk is
/// only linked; a larger one held in memory alone is not sent.
pub(super) const MAX_INLINE_IMAGE: usize = 5 * 1024 * 1024;

/// What a tool call's result adds for the client besides its text.
#[derive(Default)]
pub(super) struct ToolMedia {
    pub(super) content: Vec<ToolCallContent>,
    pub(super) locations: Vec<ToolCallLocation>,
}

impl ToolMedia {
    /// The images among `parts` small enough to send inline, a link to each
    /// image among `files`, and every file as a location.
    pub(super) fn new(parts: &[ContentPart], files: &[PathBuf]) -> Self {
        let images = parts.iter().filter_map(inline_image);
        let links = files.iter().filter_map(|path| image_link(path));
        Self {
            content: images.chain(links).map(ToolCallContent::from).collect(),
            locations: files.iter().cloned().map(ToolCallLocation::new).collect(),
        }
    }
}

/// `part` as an image block, when it is an image of at most
/// [`MAX_INLINE_IMAGE`] bytes.
fn inline_image(part: &ContentPart) -> Option<ContentBlock> {
    let ContentPart::Image { media_type, data } = part else {
        return None;
    };
    // Base64 takes four characters for every three bytes.
    if data.len() / 4 * 3 > MAX_INLINE_IMAGE {
        return None;
    }
    Some(ContentBlock::Image(ImageContent::new(
        data.as_str(),
        media_type.as_str(),
    )))
}

/// A `file://` link to `path`, when it names an image by its extension and
/// is absolute.
fn image_link(path: &Path) -> Option<ContentBlock> {
    let mime = mime_from_extension(path);
    if !mime.starts_with("image/") {
        return None;
    }
    let uri = Url::from_file_path(path).ok()?;
    let name = path.file_name().map_or_else(
        || uri.to_string(),
        |name| name.to_string_lossy().into_owned(),
    );
    Some(ContentBlock::ResourceLink(
        ResourceLink::new(name, uri.as_str()).mime_type(mime.to_owned()),
    ))
}

/// A file the model put in its reply as an image block, when it is an image
/// of at most [`MAX_INLINE_IMAGE`] bytes.
pub(super) fn reply_image(media_type: &str, data: &[u8]) -> Option<ContentBlock> {
    if !media_type.starts_with("image/") || data.len() > MAX_INLINE_IMAGE {
        return None;
    }
    Some(ContentBlock::Image(ImageContent::new(
        STANDARD.encode(data),
        media_type,
    )))
}

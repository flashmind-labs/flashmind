//! Lossless reasoning blocks returned by a provider.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Native wire format retained alongside an assistant response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, strum::AsRefStr)]
pub enum NativeReasoningFormat {
    #[strum(serialize = "anthropic.content")]
    Anthropic,
    #[strum(serialize = "responses.output")]
    Responses,
    #[strum(serialize = "mistral.content")]
    Mistral,
}

/// Ordered reasoning blocks, including signatures and opaque provider fields.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ReasoningDetails(Vec<Map<String, Value>>);

impl ReasoningDetails {
    /// Keep provider blocks without translating or dropping unknown fields.
    pub fn from_blocks(blocks: impl IntoIterator<Item = Map<String, Value>>) -> Self {
        Self(blocks.into_iter().collect())
    }

    pub fn iter(&self) -> impl Iterator<Item = &Map<String, Value>> {
        self.0.iter()
    }

    /// A provider's complete native content or output item sequence.
    pub fn native(format: NativeReasoningFormat, items: Vec<Value>) -> Self {
        let kind = format.as_ref();
        Self(vec![Map::from_iter([
            ("type".into(), Value::String(kind.into())),
            ("id".into(), Value::String(kind.into())),
            ("items".into(), Value::Array(items)),
        ])])
    }

    pub fn native_items(&self, format: NativeReasoningFormat) -> Option<&[Value]> {
        let kind = format.as_ref();
        self.0
            .iter()
            .find(|block| block.get("type").and_then(Value::as_str) == Some(kind))?
            .get("items")?
            .as_array()
            .map(Vec::as_slice)
    }

    /// Remember the exact field used by a compatible chat endpoint.
    pub fn chat_field(name: &str, text: &str) -> Self {
        Self(vec![Map::from_iter([
            ("type".into(), Value::String("chat.reasoning".into())),
            ("id".into(), Value::String(name.into())),
            ("name".into(), Value::String(name.into())),
            ("text".into(), Value::String(text.into())),
        ])])
    }

    pub fn chat_fields(&self) -> impl Iterator<Item = (&str, &str)> {
        self.0.iter().filter_map(|block| {
            if block.get("type")?.as_str()? != "chat.reasoning" {
                return None;
            }
            let name = block.get("name")?.as_str()?;
            matches!(name, "reasoning" | "reasoning_content")
                .then_some((name, block.get("text")?.as_str()?))
        })
    }

    /// Signed and encrypted state must stay attached to its original request prefix.
    pub fn is_context_bound(&self) -> bool {
        self.native_items(NativeReasoningFormat::Anthropic)
            .is_some()
            || self
                .native_items(NativeReasoningFormat::Responses)
                .is_some()
            || self.0.iter().any(|block| {
                block.get("signature").is_some_and(|value| !value.is_null())
                    || block.get("type").and_then(Value::as_str) == Some("reasoning.encrypted")
            })
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Assemble streamed fragments without changing the order of logical blocks.
    pub fn append(&mut self, delta: Self) {
        for fragment in delta.0 {
            let existing = self.0.iter_mut().rev().find(|block| {
                let same = |key| match (block.get(key), fragment.get(key)) {
                    (Some(a), Some(b)) if !a.is_null() && !b.is_null() => a == b,
                    _ => true,
                };
                let identity = ["id", "index"].iter().any(|key| {
                    block
                        .get(*key)
                        .is_some_and(|value| !value.is_null() && fragment.get(*key) == Some(value))
                });
                identity
                    && block.get("type") == fragment.get("type")
                    && same("id")
                    && same("index")
                    && same("format")
            });
            if let Some(block) = existing {
                for (key, value) in fragment {
                    if key == "items"
                        && block.get("type").and_then(Value::as_str) == Some("mistral.content")
                        && let (Some(Value::Array(current)), Value::Array(next)) =
                            (block.get_mut(&key), &value)
                    {
                        current.extend(next.iter().cloned());
                        continue;
                    }
                    if matches!(key.as_str(), "text" | "summary" | "signature" | "data")
                        && let Some(text) = value.as_str()
                        && let Some(Value::String(current)) = block.get_mut(&key)
                    {
                        current.push_str(text);
                    } else if !value.is_null() || !block.contains_key(&key) {
                        block.insert(key, value);
                    }
                }
            } else {
                self.0.push(fragment);
            }
        }
    }

    /// Readable deltas for display. Encrypted blocks remain opaque.
    pub fn text(&self) -> String {
        self.0
            .iter()
            .filter_map(|block| match block.get("type").and_then(Value::as_str) {
                Some("reasoning.text") => block.get("text").and_then(Value::as_str),
                Some("reasoning.summary") => block.get("summary").and_then(Value::as_str),
                _ => None,
            })
            .collect()
    }

    /// Conservative size estimate, including opaque blocks sent back to the API.
    pub fn estimated_bytes(&self) -> usize {
        self.0.iter().fold(0usize, |size, block| {
            size.saturating_add(Value::Object(block.clone()).to_string().len())
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn fragments_keep_signatures_unknown_fields_and_block_order() {
        let mut details = ReasoningDetails::default();
        for delta in [
            json!([{"type":"reasoning.text","index":0,"text":"Check ","signature":null,"format":"example-v1"}]),
            json!([{"type":"reasoning.encrypted","index":0,"id":"rs_example","data":"opaque","extra":{"keep":true}}]),
            json!([{"type":"reasoning.text","index":0,"text":"facts","signature":"sig-"}]),
            json!([{"type":"reasoning.text","index":0,"signature":"end"}]),
        ] {
            details.append(serde_json::from_value(delta).unwrap());
        }
        assert_eq!(
            serde_json::to_value(&details).unwrap(),
            json!([
                {"type":"reasoning.text","index":0,"text":"Check facts","signature":"sig-end","format":"example-v1"},
                {"type":"reasoning.encrypted","index":0,"id":"rs_example","data":"opaque","extra":{"keep":true}}
            ])
        );
        assert_eq!(details.text(), "Check facts");
    }

    #[test]
    fn distinct_ids_and_unidentified_blocks_are_not_combined() {
        let mut details = ReasoningDetails::default();
        for id in ["rs_first", "rs_second"] {
            details.append(
                serde_json::from_value(json!([
                    {"type":"reasoning.summary","index":0,"id":id,"summary":"summary"},
                    {"type":"reasoning.unknown","payload":"keep"}
                ]))
                .unwrap(),
            );
        }
        assert_eq!(
            serde_json::to_value(details)
                .unwrap()
                .as_array()
                .unwrap()
                .len(),
            4
        );
    }
}

use serde::{Deserialize, Serialize};

/// The user's preferred way of working with an agent.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    strum::Display,
    strum::EnumString,
)]
#[serde(rename_all = "lowercase")]
#[strum(ascii_case_insensitive)]
pub enum InteractionMode {
    Read,
    #[default]
    Act,
}

impl InteractionMode {
    pub fn toggled(self) -> Self {
        match self {
            Self::Read => Self::Act,
            Self::Act => Self::Read,
        }
    }

    /// Developer instruction for the current mode. Tool availability is unchanged.
    pub fn developer_instruction(self) -> &'static str {
        match self {
            Self::Read => {
                "You are in Read mode. Focus on investigating, understanding, explaining, comparing options, and planning. Use tools and shell commands to gather information. Avoid editing files, running commands that change state, starting services, or making external changes. Describe proposed changes instead of implementing them until the user switches to Act mode. This mode is behavioral guidance; tools remain available and normal permission rules still apply."
            }
            Self::Act => {
                "You are in Act mode. Carry out the user's requested work, including edits, shell commands, starting services, and authorized external changes. Continue until the requested work is complete, and verify the result. Normal permission rules still apply."
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_parse_toggle_and_round_trip() {
        assert_eq!(InteractionMode::default(), InteractionMode::Act);
        for mode in [InteractionMode::Read, InteractionMode::Act] {
            assert_eq!(mode.toggled().toggled(), mode);
            assert_eq!(
                mode.to_string()
                    .to_lowercase()
                    .parse::<InteractionMode>()
                    .unwrap(),
                mode
            );
            let json = serde_json::to_string(&mode).unwrap();
            assert_eq!(
                serde_json::from_str::<InteractionMode>(&json).unwrap(),
                mode
            );
            assert!(
                mode.developer_instruction()
                    .contains(&format!("{mode} mode"))
            );
        }
        assert!("unknown".parse::<InteractionMode>().is_err());
    }
}

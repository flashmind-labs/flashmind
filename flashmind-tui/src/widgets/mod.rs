//! Reusable interactive widgets for agent CLIs.
//!
//! These widgets are standalone state machines — each owns its state, handles
//! key events, and renders styled [`ratatui::text::Line`] output.  Consumers
//! decide where and how to display them.

pub mod choice_picker;
pub mod dropdown;
pub mod plan_picker;
pub mod repl;
pub mod spinner;
pub mod status_bar;
pub mod textarea;
pub mod tree;

pub use choice_picker::{ChoiceOption, ChoicePicker, ChoicePickerAction, ChoiceResponse};
pub use dropdown::{Dropdown, DropdownAction};
pub use plan_picker::{PlanPicker, PlanPickerAction, PlanResponse, PlanStep, PlanStepResponse};
pub use repl::{RawModeGuard, Repl, ReplConfig, ReplEvent};
pub use spinner::Spinner;
pub use status_bar::{StatusBar, StatusBarSection};
pub use textarea::TextArea;
pub use tree::{Tree, TreeItem};

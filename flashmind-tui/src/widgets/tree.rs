//! Tree view widget.
//!
//! Renders a hierarchical tree structure with box-drawing connectors:
//!
//! ```text
//! ├─ task A
//! │  ├─ subtask B
//! │  └─ subtask C
//! └─ task D
//!    └─ subtask E
//! ```
//!
//! The widget is display-only — build a tree of [`TreeItem`] nodes and call
//! [`Tree::lines`] to render them as styled [`ratatui::text::Line`] values.

use ratatui::text::{Line, Span};

use crate::styles::S_DIM;

// ---------------------------------------------------------------------------
// Public types

pub struct TreeItem {
    pub content: Vec<Span<'static>>,
    pub children: Vec<TreeItem>,
}

impl TreeItem {
    pub fn leaf(content: impl Into<Vec<Span<'static>>>) -> Self {
        Self {
            content: content.into(),
            children: Vec::new(),
        }
    }

    pub fn branch(content: impl Into<Vec<Span<'static>>>, children: Vec<TreeItem>) -> Self {
        Self {
            content: content.into(),
            children,
        }
    }

    pub fn child(mut self, child: TreeItem) -> Self {
        self.children.push(child);
        self
    }
}

impl<S: Into<Span<'static>>> From<S> for TreeItem {
    fn from(s: S) -> Self {
        Self::leaf(vec![s.into()])
    }
}

// ---------------------------------------------------------------------------
// Widget

pub struct Tree {
    pub roots: Vec<TreeItem>,
}

impl Tree {
    pub fn new(roots: Vec<TreeItem>) -> Self {
        Self { roots }
    }

    pub fn lines(&self) -> Vec<Line<'static>> {
        let mut out = Vec::new();
        let count = self.roots.len();
        for (i, root) in self.roots.iter().enumerate() {
            let is_last = i + 1 == count;
            render_item(&mut out, root, &[], is_last);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Rendering

fn render_item(
    out: &mut Vec<Line<'static>>,
    item: &TreeItem,
    prefix_parts: &[bool],
    is_last: bool,
) {
    let mut spans = Vec::new();

    // Build the prefix from ancestor continuation flags
    for &has_sibling_below in prefix_parts {
        let connector = if has_sibling_below { "│  " } else { "   " };
        spans.push(Span::styled(connector, S_DIM));
    }

    // Add the branch connector for this node
    let branch = if is_last { "└─ " } else { "├─ " };
    spans.push(Span::styled(branch, S_DIM));

    // Add the node content
    spans.extend(item.content.iter().cloned());

    out.push(Line::from(spans));

    // Recurse into children
    let child_count = item.children.len();
    let mut child_prefix = prefix_parts.to_vec();
    child_prefix.push(!is_last);

    for (i, child) in item.children.iter().enumerate() {
        let child_is_last = i + 1 == child_count;
        render_item(out, child, &child_prefix, child_is_last);
    }
}

// ---------------------------------------------------------------------------
// Tests

#[cfg(test)]
mod tests {
    use super::*;

    fn plain_text(line: &Line) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    #[test]
    fn single_root() {
        let tree = Tree::new(vec![TreeItem::from("root")]);
        let lines = tree.lines();
        assert_eq!(lines.len(), 1);
        assert_eq!(plain_text(&lines[0]), "└─ root");
    }

    #[test]
    fn two_roots() {
        let tree = Tree::new(vec![TreeItem::from("a"), TreeItem::from("b")]);
        let lines = tree.lines();
        assert_eq!(plain_text(&lines[0]), "├─ a");
        assert_eq!(plain_text(&lines[1]), "└─ b");
    }

    #[test]
    fn nested_tree() {
        let tree = Tree::new(vec![
            TreeItem::branch(
                vec![Span::raw("a")],
                vec![TreeItem::branch(
                    vec![Span::raw("b")],
                    vec![TreeItem::from("c")],
                )],
            ),
            TreeItem::branch(vec![Span::raw("d")], vec![TreeItem::from("e")]),
        ]);
        let lines = tree.lines();
        assert_eq!(plain_text(&lines[0]), "├─ a");
        assert_eq!(plain_text(&lines[1]), "│  └─ b");
        assert_eq!(plain_text(&lines[2]), "│     └─ c");
        assert_eq!(plain_text(&lines[3]), "└─ d");
        assert_eq!(plain_text(&lines[4]), "   └─ e");
    }

    #[test]
    fn builder_pattern() {
        let tree = Tree::new(vec![
            TreeItem::leaf(vec![Span::raw("root")])
                .child(TreeItem::from("child1"))
                .child(TreeItem::from("child2")),
        ]);
        let lines = tree.lines();
        assert_eq!(lines.len(), 3);
        assert_eq!(plain_text(&lines[0]), "└─ root");
        assert_eq!(plain_text(&lines[1]), "   ├─ child1");
        assert_eq!(plain_text(&lines[2]), "   └─ child2");
    }

    #[test]
    fn matches_user_example() {
        // |_ a
        // | |_ b
        // |   |_ c
        // |_ d
        //   |_ e
        let tree = Tree::new(vec![
            TreeItem::branch(
                vec![Span::raw("a")],
                vec![TreeItem::branch(
                    vec![Span::raw("b")],
                    vec![TreeItem::from("c")],
                )],
            ),
            TreeItem::branch(vec![Span::raw("d")], vec![TreeItem::from("e")]),
        ]);
        let lines = tree.lines();
        assert_eq!(lines.len(), 5);
        assert_eq!(plain_text(&lines[0]), "├─ a");
        assert_eq!(plain_text(&lines[1]), "│  └─ b");
        assert_eq!(plain_text(&lines[2]), "│     └─ c");
        assert_eq!(plain_text(&lines[3]), "└─ d");
        assert_eq!(plain_text(&lines[4]), "   └─ e");
    }
}

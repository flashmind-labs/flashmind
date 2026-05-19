//! Shared helpers for database query tools (SQLite, Postgres, MySQL, ClickHouse).

use crate::utils::truncate_utf8;

pub const DEFAULT_LIMIT: usize = 50;
pub const MAX_LIMIT: usize = 200;

const WRITE_KEYWORDS: &[&str] = &[
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH", "DETACH", "VACUUM",
    "REINDEX", "REPLACE", "TRUNCATE", "GRANT", "REVOKE",
];

/// Returns `true` if `sql` begins with a write/DDL keyword.
pub fn is_write_query(sql: &str) -> bool {
    let upper = sql.trim_start().to_uppercase();
    WRITE_KEYWORDS.iter().any(|kw| upper.starts_with(kw))
}

/// Clamp a user-supplied row limit to the allowed range.
pub fn clamp_limit(limit: Option<usize>) -> usize {
    limit.unwrap_or(DEFAULT_LIMIT).min(MAX_LIMIT)
}

/// Format columns + rows as an ASCII table.
///
/// Each cell is capped at `max_cell_width` characters.
pub fn format_table(columns: &[String], rows: &[Vec<String>], max_cell_width: usize) -> String {
    if rows.is_empty() {
        return "(no results)".to_string();
    }

    let mut widths: Vec<usize> = columns.iter().map(|n| n.len()).collect();
    for row in rows {
        for (i, val) in row.iter().enumerate() {
            if i < widths.len() {
                widths[i] = widths[i].max(val.len()).min(max_cell_width);
            }
        }
    }

    let mut out = String::new();

    // Header
    let header: Vec<String> = columns
        .iter()
        .zip(&widths)
        .map(|(n, w)| format!("{:<width$}", n, width = w))
        .collect();
    out.push_str(&header.join(" | "));
    out.push('\n');
    out.push_str(
        &widths
            .iter()
            .map(|w| "-".repeat(*w))
            .collect::<Vec<_>>()
            .join("-+-"),
    );
    out.push('\n');

    // Rows
    for row in rows {
        let formatted: Vec<String> = row
            .iter()
            .zip(&widths)
            .map(|(v, w)| {
                if v.chars().count() > *w {
                    format!("{}...", truncate_utf8(v, w.saturating_sub(3)))
                } else {
                    format!("{:<width$}", v, width = w)
                }
            })
            .collect();
        out.push_str(&formatted.join(" | "));
        out.push('\n');
    }

    out.push_str(&format!("\n({} rows)", rows.len()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_write_query() {
        assert!(is_write_query("INSERT INTO t VALUES (1)"));
        assert!(is_write_query("  DELETE FROM t"));
        assert!(is_write_query("drop table t"));
        assert!(!is_write_query("SELECT * FROM t"));
        assert!(!is_write_query("EXPLAIN SELECT 1"));
    }

    #[test]
    fn test_clamp_limit() {
        assert_eq!(clamp_limit(None), 50);
        assert_eq!(clamp_limit(Some(10)), 10);
        assert_eq!(clamp_limit(Some(500)), 200);
    }

    #[test]
    fn test_format_table_empty() {
        let cols = vec!["id".into()];
        assert_eq!(format_table(&cols, &[], 60), "(no results)");
    }

    #[test]
    fn test_format_table_basic() {
        let cols = vec!["id".into(), "name".into()];
        let rows = vec![vec!["1".into(), "alice".into()]];
        let out = format_table(&cols, &rows, 60);
        assert!(out.contains("id"));
        assert!(out.contains("alice"));
        assert!(out.contains("(1 rows)"));
    }
}

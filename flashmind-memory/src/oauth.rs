//! OAuth token storage for MCP servers.
//!
//! Stores per-user OAuth credentials (as JSON blobs) in the `oauth_tokens` table,
//! keyed by `(user_id, mcp_server)`. The `credentials_json` column holds the
//! serialized `rmcp::transport::auth::StoredCredentials`.

use rusqlite::{Connection, Result, params};

/// Save or update OAuth credentials for a user's MCP server.
///
/// `credentials_json` is the serialized `StoredCredentials` from rmcp.
pub fn save_credentials(
    conn: &Connection,
    user_id: i64,
    mcp_server: &str,
    credentials_json: &str,
) -> Result<()> {
    conn.execute(
        "INSERT INTO oauth_tokens (user_id, mcp_server, credentials_json, updated_at)
         VALUES (?1, ?2, ?3, datetime('now'))
         ON CONFLICT(user_id, mcp_server) DO UPDATE SET
            credentials_json = excluded.credentials_json,
            updated_at = datetime('now')",
        params![user_id, mcp_server, credentials_json],
    )?;
    Ok(())
}

/// Load OAuth credentials for a user's MCP server.
///
/// Returns the raw JSON string to be deserialized into `StoredCredentials`.
pub fn load_credentials(
    conn: &Connection,
    user_id: i64,
    mcp_server: &str,
) -> Result<Option<String>> {
    let result = conn.query_row(
        "SELECT credentials_json FROM oauth_tokens WHERE user_id = ?1 AND mcp_server = ?2",
        params![user_id, mcp_server],
        |row| row.get(0),
    );

    match result {
        Ok(json) => Ok(Some(json)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Delete OAuth credentials for a user's MCP server.
pub fn delete_credentials(conn: &Connection, user_id: i64, mcp_server: &str) -> Result<bool> {
    let rows = conn.execute(
        "DELETE FROM oauth_tokens WHERE user_id = ?1 AND mcp_server = ?2",
        params![user_id, mcp_server],
    )?;
    Ok(rows > 0)
}

/// List all MCP servers for which a user has stored credentials.
pub fn list_credentials(conn: &Connection, user_id: i64) -> Result<Vec<String>> {
    let mut stmt =
        conn.prepare("SELECT mcp_server FROM oauth_tokens WHERE user_id = ?1 ORDER BY mcp_server")?;
    let servers = stmt
        .query_map(params![user_id], |row| row.get(0))?
        .collect::<Result<Vec<String>>>()?;
    Ok(servers)
}

/// Resolve a username to user_id. Returns None if the user doesn't exist.
pub fn resolve_user_id(conn: &Connection, username: &str) -> Result<Option<i64>> {
    let result = conn.query_row(
        "SELECT id FROM user_identities WHERE username = ?1",
        params![username.to_lowercase()],
        |row| row.get(0),
    );

    match result {
        Ok(id) => Ok(Some(id)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::create_identity;
    use crate::schema::init_schema;
    use crate::test_util::register_sqlite_vec;

    fn open_db() -> Connection {
        register_sqlite_vec();
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn, 384).unwrap();
        conn
    }

    #[test]
    fn test_save_and_load_credentials() {
        let conn = open_db();
        let user_id = create_identity(&conn, "alice", None, None).unwrap();
        let creds = r#"{"client_id":"abc","token_response":null,"granted_scopes":[]}"#;

        save_credentials(&conn, user_id, "fastmail", creds).unwrap();
        let loaded = load_credentials(&conn, user_id, "fastmail").unwrap();
        assert_eq!(loaded.as_deref(), Some(creds));
    }

    #[test]
    fn test_upsert_credentials() {
        let conn = open_db();
        let user_id = create_identity(&conn, "bob", None, None).unwrap();

        save_credentials(&conn, user_id, "gmail", r#"{"client_id":"v1"}"#).unwrap();
        save_credentials(&conn, user_id, "gmail", r#"{"client_id":"v2"}"#).unwrap();

        let loaded = load_credentials(&conn, user_id, "gmail").unwrap();
        assert_eq!(loaded.as_deref(), Some(r#"{"client_id":"v2"}"#));
    }

    #[test]
    fn test_delete_credentials() {
        let conn = open_db();
        let user_id = create_identity(&conn, "carol", None, None).unwrap();
        save_credentials(&conn, user_id, "outlook", "{}").unwrap();

        let deleted = delete_credentials(&conn, user_id, "outlook").unwrap();
        assert!(deleted);

        let loaded = load_credentials(&conn, user_id, "outlook").unwrap();
        assert!(loaded.is_none());
    }

    #[test]
    fn test_delete_nonexistent() {
        let conn = open_db();
        let user_id = create_identity(&conn, "dave", None, None).unwrap();
        let deleted = delete_credentials(&conn, user_id, "nope").unwrap();
        assert!(!deleted);
    }

    #[test]
    fn test_list_credentials() {
        let conn = open_db();
        let user_id = create_identity(&conn, "eve", None, None).unwrap();
        save_credentials(&conn, user_id, "gmail", "{}").unwrap();
        save_credentials(&conn, user_id, "fastmail", "{}").unwrap();

        let servers = list_credentials(&conn, user_id).unwrap();
        assert_eq!(servers, vec!["fastmail", "gmail"]);
    }

    #[test]
    fn test_user_isolation() {
        let conn = open_db();
        let alice_id = create_identity(&conn, "alice2", None, None).unwrap();
        let bob_id = create_identity(&conn, "bob2", None, None).unwrap();

        save_credentials(&conn, alice_id, "gmail", r#"{"user":"alice"}"#).unwrap();

        let bob_creds = load_credentials(&conn, bob_id, "gmail").unwrap();
        assert!(bob_creds.is_none());
    }

    #[test]
    fn test_resolve_user_id() {
        let conn = open_db();
        create_identity(&conn, "frank", None, None).unwrap();

        assert!(resolve_user_id(&conn, "frank").unwrap().is_some());
        assert!(resolve_user_id(&conn, "unknown").unwrap().is_none());
    }
}

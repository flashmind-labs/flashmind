//! Cross-channel user identity resolution.
//!
//! Links platform accounts (Slack, Telegram, Connect, etc.) to a canonical
//! user identity. Used for memory scoping, OAuth token lookup, and
//! cross-channel context sharing.

use rusqlite::{Connection, Result, params};

#[derive(Debug, Clone)]
pub struct UserIdentity {
    pub id: i64,
    pub username: String,
    pub display_name: Option<String>,
    pub email: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Clone)]
pub struct UserChannel {
    pub id: i64,
    pub user_id: i64,
    pub channel: String,
    pub channel_user_id: String,
    pub metadata: Option<String>,
    pub linked_by: String,
    pub confidence: f64,
    pub linked_at: String,
}

/// Resolve a channel account to a canonical username.
///
/// Returns `Some(username)` if the `(channel, channel_user_id)` pair is linked
/// to a known identity, `None` otherwise.
pub fn resolve_identity(
    conn: &Connection,
    channel: &str,
    channel_user_id: &str,
) -> Result<Option<String>> {
    let result = conn.query_row(
        "SELECT i.username
         FROM user_channels c
         JOIN user_identities i ON i.id = c.user_id
         WHERE c.channel = ?1 AND c.channel_user_id = ?2",
        params![channel, channel_user_id],
        |row| row.get(0),
    );

    match result {
        Ok(username) => Ok(Some(username)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Create a new user identity.
///
/// Username is stored lowercase. Returns the new identity's row ID.
pub fn create_identity(
    conn: &Connection,
    username: &str,
    display_name: Option<&str>,
    email: Option<&str>,
) -> Result<i64> {
    conn.execute(
        "INSERT INTO user_identities (username, display_name, email) VALUES (?1, ?2, ?3)",
        params![username.to_lowercase(), display_name, email],
    )?;
    Ok(conn.last_insert_rowid())
}

/// Link a channel account to an existing user identity.
pub fn link_channel(
    conn: &Connection,
    user_id: i64,
    channel: &str,
    channel_user_id: &str,
    linked_by: &str,
    confidence: f64,
) -> Result<()> {
    conn.execute(
        "INSERT INTO user_channels (user_id, channel, channel_user_id, linked_by, confidence)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![user_id, channel, channel_user_id, linked_by, confidence],
    )?;
    Ok(())
}

/// Link a channel account by username (convenience wrapper).
///
/// Looks up the user_id from `user_identities` by username, then inserts
/// the channel link. Returns an error if the username doesn't exist.
pub fn link_channel_by_username(
    conn: &Connection,
    username: &str,
    channel: &str,
    channel_user_id: &str,
    linked_by: &str,
    confidence: f64,
) -> Result<()> {
    let user_id: i64 = conn.query_row(
        "SELECT id FROM user_identities WHERE username = ?1",
        params![username.to_lowercase()],
        |row| row.get(0),
    )?;
    link_channel(
        conn,
        user_id,
        channel,
        channel_user_id,
        linked_by,
        confidence,
    )
}

/// Get a user identity by username.
pub fn get_identity(conn: &Connection, username: &str) -> Result<Option<UserIdentity>> {
    let result = conn.query_row(
        "SELECT id, username, display_name, email, created_at
         FROM user_identities WHERE username = ?1",
        params![username.to_lowercase()],
        |row| {
            Ok(UserIdentity {
                id: row.get(0)?,
                username: row.get(1)?,
                display_name: row.get(2)?,
                email: row.get(3)?,
                created_at: row.get(4)?,
            })
        },
    );

    match result {
        Ok(identity) => Ok(Some(identity)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Get a user identity by ID.
pub fn get_identity_by_id(conn: &Connection, id: i64) -> Result<Option<UserIdentity>> {
    let result = conn.query_row(
        "SELECT id, username, display_name, email, created_at
         FROM user_identities WHERE id = ?1",
        params![id],
        |row| {
            Ok(UserIdentity {
                id: row.get(0)?,
                username: row.get(1)?,
                display_name: row.get(2)?,
                email: row.get(3)?,
                created_at: row.get(4)?,
            })
        },
    );

    match result {
        Ok(identity) => Ok(Some(identity)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e),
    }
}

/// List all channel links for a user identity.
pub fn list_channels(conn: &Connection, user_id: i64) -> Result<Vec<UserChannel>> {
    let mut stmt = conn.prepare(
        "SELECT id, user_id, channel, channel_user_id, metadata, linked_by, confidence, linked_at
         FROM user_channels WHERE user_id = ?1 ORDER BY linked_at",
    )?;

    let channels = stmt
        .query_map(params![user_id], |row| {
            Ok(UserChannel {
                id: row.get(0)?,
                user_id: row.get(1)?,
                channel: row.get(2)?,
                channel_user_id: row.get(3)?,
                metadata: row.get(4)?,
                linked_by: row.get(5)?,
                confidence: row.get(6)?,
                linked_at: row.get(7)?,
            })
        })?
        .collect::<Result<Vec<_>>>()?;

    Ok(channels)
}

/// List all user identities.
pub fn list_identities(conn: &Connection) -> Result<Vec<UserIdentity>> {
    let mut stmt = conn.prepare(
        "SELECT id, username, display_name, email, created_at
         FROM user_identities ORDER BY username",
    )?;

    let identities = stmt
        .query_map([], |row| {
            Ok(UserIdentity {
                id: row.get(0)?,
                username: row.get(1)?,
                display_name: row.get(2)?,
                email: row.get(3)?,
                created_at: row.get(4)?,
            })
        })?
        .collect::<Result<Vec<_>>>()?;

    Ok(identities)
}

/// Remove a user identity and all linked channels (CASCADE).
pub fn remove_identity(conn: &Connection, username: &str) -> Result<bool> {
    let rows = conn.execute(
        "DELETE FROM user_identities WHERE username = ?1",
        params![username.to_lowercase()],
    )?;
    Ok(rows > 0)
}

/// Update display_name and/or email on an existing identity.
pub fn update_identity(
    conn: &Connection,
    username: &str,
    display_name: Option<&str>,
    email: Option<&str>,
) -> Result<bool> {
    let rows = conn.execute(
        "UPDATE user_identities SET
            display_name = COALESCE(?2, display_name),
            email = COALESCE(?3, email)
         WHERE username = ?1",
        params![username.to_lowercase(), display_name, email],
    )?;
    Ok(rows > 0)
}

/// Resolve or create an identity for a connect user.
///
/// Connect users already have a username from their API key. This ensures
/// a corresponding `user_identities` row and `user_channels` link exist.
pub fn ensure_connect_identity(conn: &Connection, username: &str) -> Result<i64> {
    let lower = username.to_lowercase();

    // Try to find existing identity
    if let Some(identity) = get_identity(conn, &lower)? {
        // Ensure channel link exists
        let linked: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM user_channels WHERE channel = 'connect' AND channel_user_id = ?1",
                params![lower],
                |row| row.get::<_, i64>(0),
            )
            .map(|c| c > 0)?;

        if !linked {
            link_channel(conn, identity.id, "connect", &lower, "self", 1.0)?;
        }
        return Ok(identity.id);
    }

    // Create new identity
    let id = create_identity(conn, &lower, None, None)?;
    link_channel(conn, id, "connect", &lower, "self", 1.0)?;
    Ok(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::init_schema;
    use crate::test_util::register_sqlite_vec;

    fn open_db() -> Connection {
        register_sqlite_vec();
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn, 384).unwrap();
        conn
    }

    #[test]
    fn test_create_and_resolve_identity() {
        let conn = open_db();
        let id =
            create_identity(&conn, "dario", Some("Dario Gripoll"), Some("d@test.com")).unwrap();
        link_channel(&conn, id, "slack", "U0ABC123", "admin", 1.0).unwrap();

        let resolved = resolve_identity(&conn, "slack", "U0ABC123").unwrap();
        assert_eq!(resolved, Some("dario".to_string()));
    }

    #[test]
    fn test_resolve_unknown_channel() {
        let conn = open_db();
        let resolved = resolve_identity(&conn, "slack", "UNKNOWN").unwrap();
        assert_eq!(resolved, None);
    }

    #[test]
    fn test_cross_channel_resolution() {
        let conn = open_db();
        let id = create_identity(&conn, "alice", None, None).unwrap();
        link_channel(&conn, id, "slack", "U111", "admin", 1.0).unwrap();
        link_channel(&conn, id, "telegram", "222", "identity_agent", 0.9).unwrap();
        link_channel(&conn, id, "connect", "alice", "self", 1.0).unwrap();

        assert_eq!(
            resolve_identity(&conn, "slack", "U111").unwrap(),
            Some("alice".to_string())
        );
        assert_eq!(
            resolve_identity(&conn, "telegram", "222").unwrap(),
            Some("alice".to_string())
        );
        assert_eq!(
            resolve_identity(&conn, "connect", "alice").unwrap(),
            Some("alice".to_string())
        );
    }

    #[test]
    fn test_link_channel_by_username() {
        let conn = open_db();
        create_identity(&conn, "bob", None, None).unwrap();
        link_channel_by_username(&conn, "Bob", "slack", "U333", "admin", 1.0).unwrap();

        let resolved = resolve_identity(&conn, "slack", "U333").unwrap();
        assert_eq!(resolved, Some("bob".to_string()));
    }

    #[test]
    fn test_username_stored_lowercase() {
        let conn = open_db();
        create_identity(&conn, "Dario", Some("Dario"), None).unwrap();
        let identity = get_identity(&conn, "DARIO").unwrap();
        assert_eq!(identity.unwrap().username, "dario");
    }

    #[test]
    fn test_list_channels() {
        let conn = open_db();
        let id = create_identity(&conn, "carol", None, None).unwrap();
        link_channel(&conn, id, "slack", "U444", "admin", 1.0).unwrap();
        link_channel(&conn, id, "telegram", "555", "identity_agent", 0.85).unwrap();

        let channels = list_channels(&conn, id).unwrap();
        assert_eq!(channels.len(), 2);
        assert_eq!(channels[0].channel, "slack");
        assert_eq!(channels[1].channel, "telegram");
    }

    #[test]
    fn test_ensure_connect_identity() {
        let conn = open_db();

        // First call creates
        let id1 = ensure_connect_identity(&conn, "dave").unwrap();
        let identity = get_identity(&conn, "dave").unwrap().unwrap();
        assert_eq!(identity.id, id1);

        // Second call returns same
        let id2 = ensure_connect_identity(&conn, "dave").unwrap();
        assert_eq!(id1, id2);

        // Channel link exists
        let resolved = resolve_identity(&conn, "connect", "dave").unwrap();
        assert_eq!(resolved, Some("dave".to_string()));
    }

    #[test]
    fn test_remove_identity_cascades() {
        let conn = open_db();
        let id = create_identity(&conn, "eve", None, None).unwrap();
        link_channel(&conn, id, "slack", "U666", "admin", 1.0).unwrap();

        let removed = remove_identity(&conn, "eve").unwrap();
        assert!(removed);

        let channels = list_channels(&conn, id).unwrap();
        assert!(channels.is_empty());
    }

    #[test]
    fn test_update_identity() {
        let conn = open_db();
        create_identity(&conn, "frank", None, None).unwrap();

        update_identity(&conn, "frank", Some("Frank Smith"), Some("frank@test.com")).unwrap();

        let identity = get_identity(&conn, "frank").unwrap().unwrap();
        assert_eq!(identity.display_name.as_deref(), Some("Frank Smith"));
        assert_eq!(identity.email.as_deref(), Some("frank@test.com"));
    }

    #[test]
    fn test_duplicate_channel_link_fails() {
        let conn = open_db();
        let id = create_identity(&conn, "grace", None, None).unwrap();
        link_channel(&conn, id, "slack", "U777", "admin", 1.0).unwrap();

        let result = link_channel(&conn, id, "slack", "U777", "admin", 1.0);
        assert!(result.is_err());
    }
}

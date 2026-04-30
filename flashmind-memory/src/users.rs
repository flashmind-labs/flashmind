//! User management and API key authentication for Flash Connect.
//!
//! Each user has a username and one or more API keys. Keys are stored as
//! SHA-256 hashes — the raw key is only returned at creation time.

use rand::RngExt;
use rusqlite::{Connection, Result, params};
use sha2::Sha256;
use std::time::{SystemTime, UNIX_EPOCH};

/// A registered connect user.
#[derive(Debug, Clone)]
pub struct User {
    pub id: i64,
    pub username: String,
    pub created_at: i64,
}

/// Metadata about an API key (no secret material — only the prefix is visible).
#[derive(Debug, Clone)]
pub struct ApiKeyInfo {
    pub id: i64,
    pub prefix: String,
    pub created_at: i64,
    pub private: bool,
}

fn now_epoch() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before epoch")
        .as_secs() as i64
}

/// Generate a new raw API key: `sk-flash-` followed by 32 random alphanumeric chars.
pub fn generate_api_key() -> String {
    const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut rng = rand::rng();
    let suffix: String = (0..32)
        .map(|_| {
            let idx = rng.random_range(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect();
    format!("sk-flash-{suffix}")
}

/// SHA-256 hash of a raw key, hex-encoded.
pub fn hash_key(raw: &str) -> String {
    use sha2::Digest;
    hex::encode(Sha256::digest(raw.as_bytes()))
}

/// First 12 characters of the raw key (safe to display).
pub fn key_prefix(raw: &str) -> String {
    raw.chars().take(12).collect()
}

/// Create a new user with an initial API key.
///
/// Returns `(User, raw_key)`. The raw key is only available here — store it
/// or display it to the user immediately.
pub fn create_user(conn: &Connection, username: &str, private: bool) -> Result<(User, String)> {
    let now = now_epoch();

    conn.execute(
        "INSERT INTO users (username, created_at) VALUES (?1, ?2)",
        params![username, now],
    )?;

    let user_id = conn.last_insert_rowid();
    let user = User {
        id: user_id,
        username: username.to_owned(),
        created_at: now,
    };

    let raw_key = generate_api_key();
    let hash = hash_key(&raw_key);
    let prefix = key_prefix(&raw_key);

    conn.execute(
        "INSERT INTO user_api_keys (user_id, key_hash, prefix, created_at, private) VALUES (?1, ?2, ?3, ?4, ?5)",
        params![user_id, hash, prefix, now, private as i64],
    )?;

    Ok((user, raw_key))
}

/// Generate an additional API key for an existing user.
///
/// Returns the raw key. Errors if the username does not exist.
pub fn generate_key_for_user(conn: &Connection, username: &str, private: bool) -> Result<String> {
    let user_id: i64 = conn.query_row(
        "SELECT id FROM users WHERE username = ?1",
        params![username],
        |row| row.get(0),
    )?;

    let now = now_epoch();
    let raw_key = generate_api_key();
    let hash = hash_key(&raw_key);
    let prefix = key_prefix(&raw_key);

    conn.execute(
        "INSERT INTO user_api_keys (user_id, key_hash, prefix, created_at, private) VALUES (?1, ?2, ?3, ?4, ?5)",
        params![user_id, hash, prefix, now, private as i64],
    )?;

    Ok(raw_key)
}

/// Authenticate a raw API key. Returns `(username, is_private)` if valid, `None` otherwise.
pub fn authenticate(conn: &Connection, raw_key: &str) -> Result<Option<(String, bool)>> {
    let hash = hash_key(raw_key);

    let result = conn.query_row(
        "SELECT u.username, k.private
         FROM user_api_keys k
         JOIN users u ON u.id = k.user_id
         WHERE k.key_hash = ?1",
        params![hash],
        |row| {
            let username: String = row.get(0)?;
            let private: i64 = row.get(1)?;
            Ok((username, private != 0))
        },
    );

    match result {
        Ok(pair) => Ok(Some(pair)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e),
    }
}

/// List all users with their associated key metadata.
pub fn list_users(conn: &Connection) -> Result<Vec<(User, Vec<ApiKeyInfo>)>> {
    let mut users: Vec<User> = {
        let mut stmt =
            conn.prepare("SELECT id, username, created_at FROM users ORDER BY created_at")?;
        stmt.query_map([], |row| {
            Ok(User {
                id: row.get(0)?,
                username: row.get(1)?,
                created_at: row.get(2)?,
            })
        })?
        .collect::<Result<Vec<_>>>()?
    };

    let mut result = Vec::with_capacity(users.len());

    for user in users.drain(..) {
        let mut stmt = conn.prepare(
            "SELECT id, prefix, created_at, private FROM user_api_keys WHERE user_id = ?1 ORDER BY created_at",
        )?;
        let keys: Vec<ApiKeyInfo> = stmt
            .query_map(params![user.id], |row| {
                Ok(ApiKeyInfo {
                    id: row.get(0)?,
                    prefix: row.get(1)?,
                    created_at: row.get(2)?,
                    private: row.get::<_, i64>(3)? != 0,
                })
            })?
            .collect::<Result<Vec<_>>>()?;

        result.push((user, keys));
    }

    Ok(result)
}

/// Delete a user and all their API keys (CASCADE).
///
/// Returns `true` if the user existed and was deleted.
pub fn remove_user(conn: &Connection, username: &str) -> Result<bool> {
    let rows = conn.execute("DELETE FROM users WHERE username = ?1", params![username])?;
    Ok(rows > 0)
}

/// Revoke an API key by its prefix.
///
/// Returns `true` if a key with that prefix was found and deleted.
pub fn revoke_key(conn: &Connection, prefix: &str) -> Result<bool> {
    let rows = conn.execute(
        "DELETE FROM user_api_keys WHERE prefix = ?1",
        params![prefix],
    )?;
    Ok(rows > 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::init_schema;
    use crate::test_util::register_sqlite_vec;
    use rusqlite::Connection;

    fn open_db() -> Connection {
        register_sqlite_vec();
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn, 384).unwrap();
        conn
    }

    #[test]
    fn test_create_user_and_authenticate() {
        let conn = open_db();
        let (user, raw_key) = create_user(&conn, "alice", false).unwrap();

        assert_eq!(user.username, "alice");
        assert!(raw_key.starts_with("sk-flash-"));
        assert_eq!(raw_key.len(), "sk-flash-".len() + 32);

        let found = authenticate(&conn, &raw_key).unwrap();
        assert_eq!(found, Some(("alice".to_owned(), false)));
    }

    #[test]
    fn test_authenticate_invalid_key() {
        let conn = open_db();
        create_user(&conn, "bob", false).unwrap();

        let found = authenticate(&conn, "sk-flash-totallyinvalidkey000000000").unwrap();
        assert_eq!(found, None);
    }

    #[test]
    fn test_generate_additional_key() {
        let conn = open_db();
        create_user(&conn, "carol", false).unwrap();

        let second_key = generate_key_for_user(&conn, "carol", false).unwrap();
        assert!(second_key.starts_with("sk-flash-"));

        let found = authenticate(&conn, &second_key).unwrap();
        assert_eq!(found, Some(("carol".to_owned(), false)));
    }

    #[test]
    fn test_list_users() {
        let conn = open_db();
        let (_, key1) = create_user(&conn, "dave", false).unwrap();
        create_user(&conn, "eve", false).unwrap();
        generate_key_for_user(&conn, "dave", false).unwrap();

        let list = list_users(&conn).unwrap();
        assert_eq!(list.len(), 2);

        let dave = list.iter().find(|(u, _)| u.username == "dave").unwrap();
        assert_eq!(dave.1.len(), 2);
        assert_eq!(dave.1[0].prefix, key_prefix(&key1));

        let eve = list.iter().find(|(u, _)| u.username == "eve").unwrap();
        assert_eq!(eve.1.len(), 1);
    }

    #[test]
    fn test_remove_user_cascades_keys() {
        let conn = open_db();
        let (_, raw_key) = create_user(&conn, "frank", false).unwrap();

        let removed = remove_user(&conn, "frank").unwrap();
        assert!(removed);

        // Key should no longer authenticate
        let found = authenticate(&conn, &raw_key).unwrap();
        assert_eq!(found, None);

        // User table is empty
        let list = list_users(&conn).unwrap();
        assert!(list.is_empty());
    }

    #[test]
    fn test_revoke_key() {
        let conn = open_db();
        let (_, raw_key) = create_user(&conn, "grace", false).unwrap();
        let prefix = key_prefix(&raw_key);

        let revoked = revoke_key(&conn, &prefix).unwrap();
        assert!(revoked);

        // Key no longer works
        let found = authenticate(&conn, &raw_key).unwrap();
        assert_eq!(found, None);

        // User still exists
        let list = list_users(&conn).unwrap();
        assert_eq!(list.len(), 1);
        assert!(list[0].1.is_empty());
    }

    #[test]
    fn test_duplicate_username_fails() {
        let conn = open_db();
        create_user(&conn, "heidi", false).unwrap();

        let result = create_user(&conn, "heidi", false);
        assert!(result.is_err());
    }

    #[test]
    fn test_private_key_authentication() {
        let conn = open_db();
        let (_, raw_key) = create_user(&conn, "ivan", true).unwrap();

        let found = authenticate(&conn, &raw_key).unwrap();
        assert_eq!(found, Some(("ivan".to_owned(), true)));
    }

    #[test]
    fn test_genkey_private() {
        let conn = open_db();
        create_user(&conn, "judy", false).unwrap();

        let private_key = generate_key_for_user(&conn, "judy", true).unwrap();
        let found = authenticate(&conn, &private_key).unwrap();
        assert_eq!(found, Some(("judy".to_owned(), true)));
    }

    #[test]
    fn test_list_private_key() {
        let conn = open_db();
        create_user(&conn, "karl", true).unwrap();

        let list = list_users(&conn).unwrap();
        let karl = list.iter().find(|(u, _)| u.username == "karl").unwrap();
        assert!(karl.1[0].private);
    }
}

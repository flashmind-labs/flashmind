//! Multi-user Composio integration.
//!
//! Shows how a platform connects its users to apps via Composio without users
//! needing a Composio account. The flow:
//!
//! 1. Create a session for your user (using any ID you choose)
//! 2. Redirect them to the OAuth URLs Composio returns
//! 3. After they authorize, poll the session to get their `connected_account_id`
//! 4. Create an agent scoped to that user's connections
//!
//! Also demonstrates toolkit/tool discovery via `ComposioClient`.
//!
//! ```sh
//! COMPOSIO_API_KEY=your-key cargo run -p flashmind --features composio --example composio
//! ```

use std::env;

use anyhow::Result;

use flashmind::tools::ToolBuilder;
use flashmind::tools::composio::{ComposioClient, ComposioConfig, SessionToolkits};

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = env::var("COMPOSIO_API_KEY").expect("set COMPOSIO_API_KEY");
    let client = ComposioClient::new(api_key.clone(), None, None);

    // -----------------------------------------------------------------
    // 1. Discovery: browse available toolkits and tools
    // -----------------------------------------------------------------

    println!("=== Available Toolkits ===\n");
    let toolkits = client.list_toolkits().await?;
    for tk in toolkits.iter().take(10) {
        println!(
            "  {} — {}",
            tk.slug,
            tk.description.as_deref().unwrap_or("(no description)")
        );
    }
    if toolkits.len() > 10 {
        println!("  ... and {} more", toolkits.len() - 10);
    }

    println!("\n=== GitHub Tools ===\n");
    let github_tools = client.list_tools(&["github".into()]).await?;
    for tool in github_tools.iter().take(5) {
        println!(
            "  {} — {}",
            tool.slug,
            tool.description.as_deref().unwrap_or("(no description)")
        );
    }
    if github_tools.len() > 5 {
        println!("  ... and {} more", github_tools.len() - 5);
    }

    // -----------------------------------------------------------------
    // 2. Connect a user: create a session and get OAuth URLs
    // -----------------------------------------------------------------

    println!("\n=== Connecting User ===\n");

    let session = client
        .create_session(
            "user_42",
            Some(SessionToolkits {
                enable: Some(vec!["github".into(), "slack".into()]),
                disable: None,
            }),
            Some("https://myapp.com/composio/callback"),
            None,
        )
        .await?;

    println!("  Session: {}", session.session_id);

    if !session.connection_urls.is_empty() {
        println!("  Send the user to these URLs to authorize:");
        for (toolkit, url) in &session.connection_urls {
            println!("    {toolkit}: {url}");
        }
    }

    if !session.connected_accounts.is_empty() {
        println!("  Already connected:");
        for (toolkit, ids) in &session.connected_accounts {
            println!("    {toolkit}: {}", ids.join(", "));
        }
    }

    // -----------------------------------------------------------------
    // 3. Poll for completion (after user authorizes in browser)
    // -----------------------------------------------------------------

    println!("\n=== Polling Session ===\n");

    let updated = client.get_session(&session.session_id).await?;
    println!("  Connected accounts: {:?}", updated.connected_accounts);

    // In production: pick the connected_account_id for the toolkit you need
    // and store it in your database for this user.

    // -----------------------------------------------------------------
    // 4. Create a per-user agent with their connections
    // -----------------------------------------------------------------

    println!("\n=== Building Per-User Agent ===\n");

    // In production, this ID comes from step 3 (stored in your DB).
    let connected_account_id = updated
        .connected_accounts
        .values()
        .flat_map(|ids| ids.first())
        .next()
        .cloned()
        .unwrap_or_else(|| "demo-account-id".into());

    let config = ComposioConfig {
        api_key,
        connected_account_id: Some(connected_account_id),
        toolkits: vec!["github".into(), "slack".into()],
        base_url: None,
    };

    let (tools, _sync) = ToolBuilder::new().composio(config).build_with_sync().await;

    println!("  {} tools registered", tools.list().len());
    for name in tools.list().iter().take(5) {
        println!("    - {name}");
    }

    Ok(())
}

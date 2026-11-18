//! `flsh mcp`: manage the MCP servers in `config.toml` and sign in to remote ones.

mod login;

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use clap::{ArgGroup, Args, Subcommand};
use flashmind_tools::mcp::session::{SessionServerConfig, SessionTransport, valid_name};
use flashmind_tools::mcp::{McpAuthRequired, McpConfigProvider, McpDiskConfig};
use toml_edit::{Array, ArrayOfTables, DocumentMut, InlineTable, Item, Table, value};
use url::Url;

use crate::config::{load_config, update_config};
use crate::paths::data_dir;

const KEY: &str = "mcp_servers";

#[derive(Args)]
pub(crate) struct McpArgs {
    #[command(subcommand)]
    command: McpCommand,
}

#[derive(Subcommand)]
enum McpCommand {
    /// Add a local MCP server, or a remote one with --url
    #[command(
        group(ArgGroup::new("transport").required(true).args(["command", "url"])),
        after_help = "Examples:\n  \
            flsh mcp add github --env GITHUB_TOKEN=... -- github-mcp --stdio\n  \
            flsh mcp add fastmail --url https://api.fastmail.com/mcp"
    )]
    Add {
        /// Server name. Its tools are named `mcp__<name>__<tool>`
        name: String,
        /// Environment variable for the server, as KEY=VALUE (repeatable)
        #[arg(long = "env", value_name = "KEY=VALUE", value_parser = parse_env, conflicts_with = "url")]
        env: Vec<(String, String)>,
        /// Streamable HTTP URL of a remote server. Sign in with `flsh mcp login`
        #[arg(long, value_parser = parse_url)]
        url: Option<Url>,
        /// Command that starts the server, followed by its arguments
        #[arg(trailing_var_arg = true, value_name = "COMMAND")]
        command: Vec<String>,
    },
    /// List configured MCP servers
    #[command(alias = "ls")]
    List,
    /// Remove an MCP server and its saved sign-in
    #[command(alias = "rm")]
    Remove {
        /// Server name
        name: String,
    },
    /// Sign in to a remote MCP server with OAuth
    Login {
        /// Server name
        name: String,
    },
    /// Forget the saved sign-in of a remote MCP server
    Logout {
        /// Server name
        name: String,
    },
}

pub(crate) async fn run(args: McpArgs) -> Result<()> {
    if let McpCommand::Remove { name } | McpCommand::Login { name } | McpCommand::Logout { name } =
        &args.command
    {
        // The name becomes the file name of the saved sign-in.
        ensure!(valid_name(name), "invalid server name '{name}'");
    }
    match args.command {
        McpCommand::Add {
            name,
            env,
            url,
            command,
        } => {
            let mut server = SessionServerConfig {
                name,
                url: url.map(String::from),
                env: env.into_iter().collect(),
                ..Default::default()
            };
            if let Some((program, rest)) = command.split_first() {
                server.command = Some(program.into());
                server.args = rest.to_vec();
            }
            update_config(|doc| add_server(doc, &server))?;
            println!("Added MCP server '{}'.", server.name);
            if server.url.is_some() {
                println!("Sign in with: flsh mcp login {}", server.name);
            }
        }
        McpCommand::List => {
            let config = load_config()?;
            if config.mcp_servers.is_empty() {
                println!(
                    "No MCP servers. Add one with: flsh mcp add <name> -- <command> [args...]"
                );
            }
            let credentials = credentials()?;
            for server in &config.mcp_servers {
                let signed_in = match server.url {
                    Some(_) => Some(signed_in(&credentials, &server.name).await?),
                    None => None,
                };
                println!("{}", describe(server, signed_in));
            }
        }
        McpCommand::Remove { name } => {
            update_config(|doc| remove_server(doc, &name))?;
            credentials()?.delete_config(&name).await?;
            println!("Removed MCP server '{name}'.");
        }
        McpCommand::Login { name } => login::run(&name).await?,
        McpCommand::Logout { name } => {
            let credentials = credentials()?;
            ensure!(
                signed_in(&credentials, &name).await?,
                "MCP server '{name}' has no saved sign-in"
            );
            credentials.delete_config(&name).await?;
            println!("Signed out of MCP server '{name}'.");
        }
    }
    Ok(())
}

/// Where sign-ins to remote servers are kept: one JSON file per server.
pub(crate) fn credentials() -> Result<Arc<dyn McpConfigProvider>> {
    Ok(McpDiskConfig::new(data_dir()?.join("mcp")))
}

async fn signed_in(credentials: &Arc<dyn McpConfigProvider>, name: &str) -> Result<bool> {
    Ok(credentials
        .load_credentials(name)
        .await?
        .is_some_and(|value| !value.is_null()))
}

/// Turn a failed sign-in at session start into the command that fixes it.
pub(crate) async fn explain(error: anyhow::Error) -> anyhow::Error {
    let Some(McpAuthRequired { server }) = error.downcast_ref::<McpAuthRequired>() else {
        return error;
    };
    let saved = match credentials() {
        Ok(credentials) => signed_in(&credentials, server).await.unwrap_or(false),
        Err(_) => false,
    };
    if saved {
        anyhow::anyhow!(
            "could not connect to MCP server '{server}' with its saved sign-in. \
             If the sign-in expired, run: flsh mcp login {server}"
        )
    } else {
        anyhow::anyhow!("MCP server '{server}' needs a sign-in. Run: flsh mcp login {server}")
    }
}

fn parse_env(arg: &str) -> Result<(String, String), String> {
    match arg.split_once('=') {
        Some((key, val)) if !key.is_empty() => Ok((key.into(), val.into())),
        _ => Err(format!("expected KEY=VALUE, got '{arg}'")),
    }
}

fn parse_url(arg: &str) -> Result<Url, String> {
    let url = Url::parse(arg).map_err(|error| format!("invalid URL '{arg}': {error}"))?;
    match url.scheme() {
        "https" | "http" => Ok(url),
        scheme => Err(format!("expected an http or https URL, got '{scheme}'")),
    }
}

/// One line per server. Environment values are left out because they often hold secrets.
fn describe(server: &SessionServerConfig, signed_in: Option<bool>) -> String {
    let mut line = format!("{}: ", server.name);
    match server.transport() {
        Ok(SessionTransport::Stdio { command }) => {
            line.push_str(&command.display().to_string());
        }
        Ok(SessionTransport::Http { url }) => line.push_str(url),
        Err(error) => return format!("{line}{error}"),
    }
    for arg in &server.args {
        line.push(' ');
        line.push_str(arg);
    }
    if !server.env.is_empty() {
        let keys: Vec<&str> = server.env.keys().map(String::as_str).collect();
        line.push_str(&format!(" (env: {})", keys.join(", ")));
    }
    match signed_in {
        Some(true) => line.push_str(" (signed in)"),
        Some(false) => line.push_str(" (not signed in)"),
        None => {}
    }
    line
}

fn servers_mut(doc: &mut DocumentMut) -> Result<&mut ArrayOfTables> {
    doc.entry(KEY)
        .or_insert(Item::ArrayOfTables(ArrayOfTables::new()))
        .as_array_of_tables_mut()
        .context(
            "mcp_servers in config.toml is not a list of [[mcp_servers]] tables; edit it by hand",
        )
}

fn server_name(table: &Table) -> Option<&str> {
    table.get("name").and_then(Item::as_str)
}

fn add_server(doc: &mut DocumentMut, server: &SessionServerConfig) -> Result<()> {
    ensure!(
        valid_name(&server.name),
        "invalid server name '{}': use ASCII letters, digits, '_' and '-'",
        server.name
    );
    server.transport()?;
    let servers = servers_mut(doc)?;
    if servers
        .iter()
        .any(|table| server_name(table) == Some(&server.name))
    {
        bail!(
            "MCP server '{}' already exists; remove it first with: flsh mcp remove {}",
            server.name,
            server.name
        );
    }
    let mut table = Table::new();
    table["name"] = value(&server.name);
    if let Some(command) = &server.command {
        let command = command
            .to_str()
            .context("server command is not valid UTF-8")?;
        table["command"] = value(command);
    }
    if let Some(url) = &server.url {
        table["url"] = value(url);
    }
    if !server.args.is_empty() {
        table["args"] = value(server.args.iter().collect::<Array>());
    }
    if !server.env.is_empty() {
        table["env"] = value(env_table(&server.env));
    }
    // Comments after the last setting would otherwise end up below the new
    // header, and uncommenting one would move that setting into this table.
    let trailing = doc.trailing().as_str().unwrap_or_default().trim_end();
    if !trailing.is_empty() {
        table.decor_mut().set_prefix(format!("{trailing}\n\n"));
        doc.set_trailing("");
    }
    servers_mut(doc)?.push(table);
    Ok(())
}

fn env_table(env: &BTreeMap<String, String>) -> InlineTable {
    env.iter().map(|(key, val)| (key.as_str(), val)).collect()
}

fn remove_server(doc: &mut DocumentMut, name: &str) -> Result<()> {
    let servers = servers_mut(doc)?;
    let index = servers
        .iter()
        .position(|table| server_name(table) == Some(name))
        .with_context(|| format!("no MCP server named '{name}'"))?;
    // Keep the comments written above the removed header.
    let comments = servers
        .get(index)
        .and_then(|table| table.decor().prefix())
        .and_then(|prefix| prefix.as_str())
        .filter(|prefix| !prefix.trim().is_empty())
        .map(str::to_owned);
    servers.remove(index);
    if let Some(comments) = comments {
        if let Some(next) = servers.get_mut(index) {
            let rest = next
                .decor()
                .prefix()
                .and_then(|prefix| prefix.as_str())
                .unwrap_or("\n");
            let prefix = format!("{comments}{rest}");
            next.decor_mut().set_prefix(prefix);
        } else {
            let trailing = doc.trailing().as_str().unwrap_or_default();
            let trailing = format!("{comments}{trailing}");
            doc.set_trailing(trailing);
        }
    }
    if servers_mut(doc)?.is_empty() {
        doc.remove(KEY);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;
    use crate::config::Config;

    #[derive(Parser)]
    struct Cli {
        #[command(subcommand)]
        command: McpCommand,
    }

    fn parse(args: &[&str]) -> McpCommand {
        Cli::try_parse_from(std::iter::once("mcp").chain(args.iter().copied()))
            .unwrap()
            .command
    }

    fn server(name: &str) -> SessionServerConfig {
        SessionServerConfig {
            name: name.into(),
            command: Some("example-server".into()),
            args: vec!["--stdio".into()],
            env: BTreeMap::from([("EXAMPLE_TOKEN".into(), "secret".into())]),
            ..Default::default()
        }
    }

    fn remote(name: &str) -> SessionServerConfig {
        SessionServerConfig {
            name: name.into(),
            url: Some("https://mcp.example.com/mcp".into()),
            ..Default::default()
        }
    }

    #[test]
    fn add_takes_server_flags_literally() {
        let McpCommand::Add {
            name, env, command, ..
        } = parse(&[
            "add", "x", "--env", "A=1=2", "npx", "-y", "pkg", "--env", "B=2",
        ])
        else {
            panic!("expected add");
        };
        assert_eq!(name, "x");
        assert_eq!(env, vec![("A".into(), "1=2".into())]);
        assert_eq!(command, ["npx", "-y", "pkg", "--env", "B=2"]);

        let McpCommand::Add { command, .. } = parse(&["add", "x", "--", "srv", "--stdio"]) else {
            panic!("expected add");
        };
        assert_eq!(command, ["srv", "--stdio"]);
    }

    #[test]
    fn add_requires_command_and_valid_env() {
        assert!(Cli::try_parse_from(["mcp", "add", "x"]).is_err());
        assert!(Cli::try_parse_from(["mcp", "add", "x", "--env", "NOEQUALS", "srv"]).is_err());
        assert!(Cli::try_parse_from(["mcp", "add", "x", "--env", "=v", "srv"]).is_err());
    }

    #[test]
    fn add_and_remove_keep_other_settings() {
        let mut doc: DocumentMut = "# my settings\nmodel = \"example/model\"\n"
            .parse()
            .unwrap();
        add_server(&mut doc, &server("one")).unwrap();
        add_server(&mut doc, &server("two")).unwrap();

        let config: Config = toml::from_str(&doc.to_string()).unwrap();
        assert_eq!(config.mcp_servers.len(), 2);
        assert_eq!(config.mcp_servers[0].name, "one");
        assert_eq!(config.mcp_servers[0].args, ["--stdio"]);
        assert_eq!(config.mcp_servers[0].env["EXAMPLE_TOKEN"], "secret");

        remove_server(&mut doc, "one").unwrap();
        let config: Config = toml::from_str(&doc.to_string()).unwrap();
        assert_eq!(config.mcp_servers.len(), 1);
        assert_eq!(config.mcp_servers[0].name, "two");

        remove_server(&mut doc, "two").unwrap();
        assert_eq!(
            doc.to_string(),
            "# my settings\nmodel = \"example/model\"\n"
        );
    }

    #[test]
    fn add_goes_below_trailing_comments_and_remove_keeps_them() {
        let original =
            "model = \"example/model\"\n\n# system_prompt = \"example\"\n# reasoning = \"off\"\n";
        let mut doc: DocumentMut = original.parse().unwrap();
        add_server(&mut doc, &server("one")).unwrap();
        add_server(&mut doc, &server("two")).unwrap();
        let text = doc.to_string();
        let header = text.find("[[mcp_servers]]").unwrap();
        assert!(text.find("# reasoning").unwrap() < header, "{text}");

        // Uncommenting a setting must keep it out of the server tables.
        let uncommented = text.replace("# reasoning = \"off\"", "reasoning = \"off\"");
        let config: Config = toml::from_str(&uncommented).unwrap();
        assert_eq!(config.mcp_servers.len(), 2);

        remove_server(&mut doc, "one").unwrap();
        let text = doc.to_string();
        assert!(text.find("# reasoning").unwrap() < text.find("[[mcp_servers]]").unwrap());
        remove_server(&mut doc, "two").unwrap();
        assert!(
            doc.to_string()
                .contains("# system_prompt = \"example\"\n# reasoning = \"off\"\n")
        );
        let config: Config = toml::from_str(&doc.to_string()).unwrap();
        assert!(config.mcp_servers.is_empty());
    }

    #[test]
    fn add_rejects_duplicates_and_bad_names() {
        let mut doc = DocumentMut::new();
        add_server(&mut doc, &server("one")).unwrap();
        assert!(add_server(&mut doc, &server("one")).is_err());
        assert!(add_server(&mut doc, &server("bad name")).is_err());
        assert!(add_server(&mut doc, &server("")).is_err());
    }

    #[test]
    fn remove_reports_missing_server() {
        let mut doc = DocumentMut::new();
        let error = remove_server(&mut doc, "nope").unwrap_err();
        assert!(error.to_string().contains("no MCP server named 'nope'"));
    }

    #[test]
    fn inline_array_is_reported_not_overwritten() {
        let mut doc: DocumentMut = "mcp_servers = []\n".parse().unwrap();
        assert!(add_server(&mut doc, &server("one")).is_err());
        assert_eq!(doc.to_string(), "mcp_servers = []\n");
    }

    #[test]
    fn describe_hides_env_values() {
        let line = describe(&server("one"), None);
        assert_eq!(line, "one: example-server --stdio (env: EXAMPLE_TOKEN)");
        let line = describe(&remote("two"), Some(false));
        assert_eq!(line, "two: https://mcp.example.com/mcp (not signed in)");
    }

    #[test]
    fn add_takes_url_or_command_but_not_both() {
        let McpCommand::Add { url, command, .. } =
            parse(&["add", "fastmail", "--url", "https://mcp.example.com/mcp"])
        else {
            panic!("expected add");
        };
        assert_eq!(url.unwrap().as_str(), "https://mcp.example.com/mcp");
        assert!(command.is_empty());

        for args in [
            &["add", "x"][..],
            &["add", "x", "--url", "https://mcp.example.com/mcp", "srv"],
            &[
                "add",
                "x",
                "--url",
                "https://mcp.example.com/mcp",
                "--env",
                "A=1",
            ],
            &["add", "x", "--url", "ftp://mcp.example.com"],
            &["add", "x", "--url", "not a url"],
        ] {
            assert!(
                Cli::try_parse_from(std::iter::once("mcp").chain(args.iter().copied())).is_err(),
                "{args:?}"
            );
        }
    }

    #[test]
    fn remote_server_round_trips_through_config() {
        let mut doc = DocumentMut::new();
        add_server(&mut doc, &remote("fastmail")).unwrap();
        assert_eq!(
            doc.to_string(),
            "[[mcp_servers]]\nname = \"fastmail\"\nurl = \"https://mcp.example.com/mcp\"\n"
        );
        let config: Config = toml::from_str(&doc.to_string()).unwrap();
        assert!(matches!(
            config.mcp_servers[0].transport().unwrap(),
            SessionTransport::Http { url } if url == "https://mcp.example.com/mcp"
        ));
    }
}

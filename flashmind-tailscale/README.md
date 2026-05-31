# flashmind-tailscale

Tailscale local API client and Funnel helpers for [Flashmind](https://github.com/flashmind-labs/flashmind).

## Usage

```rust
use flashmind_tailscale::{TailscaleClient, FunnelManager};

let client = TailscaleClient::new()?;
let status = client.status().await?;

// Expose a local port via Tailscale Funnel
let funnel = FunnelManager::new(client);
funnel.serve(8080, "my-service").await?;
```

## Features

- Unix socket communication with Tailscale daemon
- CLI fallback when socket unavailable
- Funnel route management

## License

MPL-2.0

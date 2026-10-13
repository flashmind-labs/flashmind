# flashmind-tools

Built-in tools and service integrations, composed through `ToolBuilder`.

- A new tool implements `Tool`, is exported from `lib.rs`, and is registered
  in `builder.rs`.
- Every integration sits behind its feature flag and offers a full and a
  read-only variant. Where the service allows it, enforce read-only on its
  side too (read-only transactions, command allowlists), not only by hiding
  tools.
- OAuth integrations register only their auth tool until a valid token
  exists.
- Each tool file carries its own tests.

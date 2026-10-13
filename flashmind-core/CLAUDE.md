# flashmind-core

Agent runtime: the loop that calls the model, runs tools and manages the
conversation.

- Stay provider-agnostic: depend only on `flashmind-types`.
- `agent.start()` borrows the agent and the conversation for the life of the
  returned stream, so the stream cannot be returned from a closure. Drive it
  with a state machine, as the `flashmind-tui` REPL does.

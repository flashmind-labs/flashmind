# jev

Install the `jev` command from this repository:

```sh
cargo install --git https://github.com/flashmind-labs/flashmind jev-cli
```

Put your OpenRouter key in `~/.jev-cli`:

```toml
openrouter_key = "YOUR_OPENROUTER_KEY"
```

Keep this file private (`chmod 600 ~/.jev-cli`). You can also set
`OPENROUTER_API_KEY` if the config file does not exist.

Pass a request as JSON on stdin or in a file. The model defaults to `jev-latest`.
Use `--model` to override it.

```sh
jev <<'JSON'
{
  "state": "The payment page is blank.",
  "questions": {
    "is_bug": {
      "type": "noul",
      "instructions": "Is this a software defect?"
    },
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Charges, refunds, and invoices",
        "technical": "Product defects and outages"
      }
    }
  }
}
JSON
```

The command prints the full JSON response to stdout. Errors go to stderr and
return a nonzero exit code.

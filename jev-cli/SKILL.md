---
name: jev
description: Make bounded, typed classification and routing decisions with Jev through OpenRouter.
---

# Jev decisions

Run `jev` with one JSON request on stdin. It calls OpenRouter's System One API
and returns JSON on stdout. The user installs the binary with:

```sh
cargo install --git https://github.com/flashmind-labs/flashmind jev-cli
```

The OpenRouter key belongs in `~/.jev-cli` as
`openrouter_key = "YOUR_OPENROUTER_KEY"`. The `OPENROUTER_API_KEY` environment
variable works when that file is absent.

Use Jev for narrow decisions, such as classification, filtering, scoring, and
routing. Treat the result as a recommendation. Do not use it for prose writing
or irreversible actions.

`state` contains the item to judge. `questions` maps stable IDs to typed
questions. Omit `model` to use `jev-latest`.

```sh
jev <<'JSON'
{
  "state": "Customer asks for a refund after a duplicate charge.",
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Charges and refunds",
        "technical": "Product defects"
      }
    },
    "needs_reply": {
      "type": "noul",
      "instructions": "Does this need a reply?"
    }
  }
}
JSON
```

`choice` selects a named option, `noul` gives a probability from 0 to 1, and
`score` rates an ordered list of criteria. Group related questions into one
request and parse the JSON response. Keep secrets and unnecessary personal data
out of `state`.

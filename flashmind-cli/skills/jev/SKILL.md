---
name: jev
description: Make bounded, typed classification and routing decisions with Jev through OpenRouter.
---

# Jev decisions

Run `flsh decide` with one JSON request on stdin. It calls OpenRouter's System One API
and returns JSON on stdout. The user installs the binary with:

```sh
cargo install flashmind-cli
```

Set `OPENROUTER_API_KEY` or configure the OpenRouter key in `flsh`.

Use Jev for narrow decisions, such as classification, filtering, scoring, and
routing. Treat the result as a recommendation. Do not use it for prose writing
or irreversible actions.

`state` contains the item to judge. `questions` maps stable IDs to typed
questions. Omit `model` to use `jev-latest`.

```sh
flsh decide <<'JSON'
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

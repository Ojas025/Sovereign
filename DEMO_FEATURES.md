# Added Local Features

These additions are deliberately separate from the core agent loop.

## 1. PII redaction

`/redact <text>` masks common email, phone, IPv4, Aadhaar-like and PAN-like values.

## 2. Human review queue

`/review` lists pending items.

`/review add <reason> <content>` creates a review item.

`/review approve <id> [note]` or `/review reject <id> [note]` records a decision.

## 3. Local knowledge

`/knowledge add <file>` indexes a local text/markdown/CSV/JSON/log file.

`/knowledge search <query>` performs deterministic local retrieval.

`/knowledge list` lists indexed files.

This is a lightweight local retrieval layer for the prototype; it does not claim to be a full vector RAG or OKF implementation.

## 4. Sovereignty status

`/sovereignty` checks the local runtime policy: offline environment flags, localhost model endpoint and proxy environment.

It explicitly reports that this is a policy check, not packet-capture proof.

## 5. Offline feature demo

Run:

`PYTHONPATH=src python scripts/demo_features.py`

This exercises the added features without requiring an LLM server, GPU, network access or model weights.

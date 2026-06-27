# Security Policy

Local Brain RAG is a portfolio project, but it is structured with production security habits.

## Supported Branch

Security fixes are accepted against `main`.

## Reporting

Please do not disclose sensitive findings publicly before a fix is available. Open a private GitHub security advisory when possible, or create a minimal issue that avoids exploit details.

## Baseline Practices

- Never commit real API keys, tokens, private data, or local `.env` files.
- Keep generated artifacts and local caches out of Git.
- Validate environment configuration before enabling integrations.
- Treat AI output as untrusted until it passes guardrails and human review.

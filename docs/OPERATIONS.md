# Operations

This document describes how to run Local Brain RAG with production-minded habits.

## Runtime Ownership

- Keep runtime configuration outside source control.
- Prefer environment variables for service endpoints and credentials.
- Document every required variable in `.env.example`.
- Review logs for sensitive content before sharing them.

## Release Checklist

1. Run the repository health check.
2. Run tests and build checks.
3. Review changed environment variables and migrations.
4. Confirm the README still matches the shipped behavior.
5. Tag or note the release after GitHub Actions pass.

## Observability

- Health endpoints or readiness scripts should fail loudly when required files, scripts, or configuration are missing.
- User-facing workflows should expose clear error states instead of silent failures.
- AI-enabled workflows should retain enough trace context for review without storing unnecessary private data.

## Incident Response

- Revert the smallest risky change first.
- Preserve logs and failing inputs for debugging.
- Add a regression test or eval scenario before closing the incident.

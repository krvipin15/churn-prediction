# Security Policy

## Supported Versions

This project is a deployed application, not a versioned library — only the latest code on the `main` branch (and the most recently published container images on GHCR) receives security fixes. There is no support commitment for older tags or previous `IMAGE_VERSION` releases.

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it privately rather than opening a public issue.

**Preferred:** Use [GitHub's private vulnerability reporting](https://github.com/krvipin15/churn-prediction/security/advisories/new) ("Report a vulnerability" under the Security tab). This keeps the report and any discussion private until a fix is available.

**Alternative:** Email [krvipin15@tutamail.com](mailto:krvipin15@tutamail.com) with a description of the issue, steps to reproduce, and its potential impact.

Please do not disclose the vulnerability publicly (issues, pull requests, social media) until it has been triaged and a fix or mitigation is released.

### What to include

- A clear description of the vulnerability and its impact.
- Steps to reproduce, or a proof of concept, where possible.
- The affected component (e.g. the FastAPI service, the Streamlit dashboard, the training pipeline, a container image, or a dependency).

### Response process

This is a solo-maintained project without a formal SLA, but reports will be handled as follows on a best-effort basis:

1. **Acknowledgement** — within a few days of the report.
2. **Triage** — confirm the issue, assess severity and affected scope.
3. **Fix** — a patch is developed and, where applicable, coordinated disclosure is agreed with the reporter.
4. **Disclosure** — a fix is released and, for significant issues, a GitHub Security Advisory is published.

## Scope

**In scope:**
- The FastAPI inference service and its routes (`predict`, `explain`, `health`)
- The Streamlit dashboard
- The training/prediction pipelines and data validation logic
- `Containerfile.api`, `Containerfile.dashboard`, and `compose.yaml`
- CI/CD workflows under `.github/workflows/`

**Out of scope / handled differently:**
- Vulnerabilities in third-party dependencies with no available patch — these are tracked via Dependabot alerts and assessed individually rather than through this reporting process; still feel free to flag one you believe is actually exploitable in this project's context.
- Findings requiring physical access to a deployment, or that rely on a misconfigured deployment not matching the defaults in this repository.

## Automated Security Tooling

This repository runs continuous automated scanning; you don't need to report findings these already surface, but you're welcome to flag ones that appear to have been missed:

- **CodeQL** (SAST) — on every push/PR to `main`/`develop`, and weekly on a schedule
- **OSV-Scanner** — dependency vulnerability scanning
- **Dependabot** — dependency and package-ecosystem alerts
- **pre-commit hooks** — `detect-secrets` (secret scanning) and `hadolint` (Containerfile linting) run locally and in CI

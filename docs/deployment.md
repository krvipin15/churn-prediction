# Deployment

## Containerization

The API and dashboard each have a dedicated `Containerfile` and are orchestrated together via `compose.yaml`. Required model and data artifacts (`models/preprocessor.joblib`, `models/model.ubj`, `models/card.json`, `data/raw/test.csv`) are copied into each image at build time, so neither container needs DVC/DagsHub credentials or network access at runtime.

```bash
just container-up     # Build and start the application containers in detached mode
just container-down   # Stop the application containers

# Without just command
podman-compose up --build -d --remove-orphans --force-recreate
```

Managing the stack manually:

```bash
podman-compose ps            # View running container status
podman-compose logs -f       # Stream container logs
podman-compose down          # Stop running stack
```

The API serves on `http://localhost:8000` (`/docs` for interactive API docs) and the dashboard on `http://localhost:8501`.

!!! note "Shared volumes"
    The `api` and `dashboard` services read/write some of the same host directories directly (`data/`, `reports/`) rather than only communicating over HTTP. If either service starts writing a new artifact type, both sides of `compose.yaml` need the matching volume mount.

Containerfiles are validated with `hadolint`:

```bash
hadolint Containerfile.api Containerfile.dashboard
```

## Automated image publishing (CI/CD)

Pushing a version tag (`v*.*.*`) — or a manual workflow dispatch — triggers `.github/workflows/build.yml`, which for each service (`api`, `dashboard`) in parallel:

1. Pulls only the DVC artifacts that service needs (the model + preprocessor for the API, the demo CSV for the dashboard).
2. Builds the image locally in the runner's Docker daemon.
3. Scans it with **Trivy**, failing the job on any CRITICAL/HIGH severity CVE.
4. If the scan passes, tags it with the semver version and short commit SHA and pushes it to GHCR.

```bash
git tag v0.2.0
git push origin v0.2.0
```

There's no manual `podman build`/`podman push` step — publishing an image now only happens through this pipeline.

## Deploying to Render

Both `Containerfile.api` and `Containerfile.dashboard` end in a shell-form `CMD` that reads Render's injected `$PORT` environment variable (falling back to `8000`/`8501` locally), so they run on Render's "Web Service" (Docker) deploy type with no changes needed:

1. Create a new Web Service, connect the GitHub repo, and set **Environment** to `Docker`.
2. Set **Dockerfile Path** to `Containerfile.api` (or `Containerfile.dashboard` for a second service).
3. Leave the port field to Render's auto-detection — the container already binds to `$PORT`.

!!! warning "Environment variable validation"
    `Settings` currently requires `SENTRY_DSN`, `KAGGLE_USERNAME`/`KAGGLE_KEY`, and `DAGSHUB_ACCESS_ID` to be set whenever `ENVIRONMENT` is `staging` or `production` — even though, with artifacts baked into the image at build time, the running API/dashboard containers no longer actually use Kaggle or DagsHub credentials at all. Until that validator is scoped more narrowly (e.g. only required for the training pipeline), you'll need to either set those variables anyway on Render or set `ENVIRONMENT=development` there, even in a real deployment.

Since the dashboard talks to the API over HTTP, set the dashboard's `API_BASE_URL` environment variable on Render to the API service's public Render URL.

# Deployment

## Containerization

The API and dashboard each have a dedicated `Containerfile` and are orchestrated together via `compose.yaml`.

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

## Publishing images to GHCR

Images are currently published manually. Automating this via CI/CD (build + push on version tag) is planned.

```bash
just ghcr-login              # Log in to GHCR using GHCR_USER / GHCR_PAT
just ghcr-build              # Build API & Dashboard images with version and latest tags
just ghcr-push               # Push version and latest images to GHCR

# Without just command
echo "${GHCR_PAT}" | podman login ghcr.io -u $GHCR_USER --password-stdin

podman build -f Containerfile.api \
  -t ghcr.io/${GHCR_USER}/churn-prediction-api:v0.1.0 \
  -t ghcr.io/${GHCR_USER}/churn-prediction-api:latest .
podman build -f Containerfile.dashboard \
  -t ghcr.io/${GHCR_USER}/churn-prediction-dashboard:v0.1.0 \
  -t ghcr.io/${GHCR_USER}/churn-prediction-dashboard:latest .

podman push ghcr.io/${GHCR_USER}/churn-prediction-api:v0.1.0
podman push ghcr.io/${GHCR_USER}/churn-prediction-api:latest
podman push ghcr.io/${GHCR_USER}/churn-prediction-dashboard:v0.1.0
podman push ghcr.io/${GHCR_USER}/churn-prediction-dashboard:latest
```

## Isolated pod deployment (Podman Pods)

An alternative to `compose.yaml` — runs both services inside a single Podman pod sharing a port and network namespace, pulling published images from GHCR instead of building locally.

```bash
just pod-up                  # Create pod, launch API and Dashboard containers
just pod-down                # Stop running pod containers
just clean-pod               # Force-remove the pod

# Target specific steps or custom pod names
just pod-create pod=churn-pod
just pod-run-api pod=churn-pod owner=$GHCR_USER tag=latest
just pod-run-dashboard pod=churn-pod owner=$GHCR_USER tag=latest

# Without just command
podman pod create --name churn-pod --userns=keep-id -p 8000:8000 -p 8501:8501

podman run -d \
  --pod churn-pod \
  --name churn-api \
  --restart unless-stopped \
  --env-file .env \
  -e FASTAPI_HOST=0.0.0.0 \
  -e FASTAPI_PORT=8000 \
  -v ./data:/app/data:z \
  -v ./models:/app/models:ro,z \
  -v ./reports:/app/reports:z \
  -v ./logs:/app/logs:z \
  ghcr.io/${GHCR_USER}/churn-prediction-api:latest

podman run -d \
  --pod churn-pod \
  --name churn-dashboard \
  --restart unless-stopped \
  --env-file .env \
  -e FASTAPI_HOST=127.0.0.1 \
  -e FASTAPI_PORT=8000 \
  -e API_BASE_URL=http://127.0.0.1:8000 \
  -v ./data:/app/data:ro,z \
  -v ./models:/app/models:ro,z \
  -v ./reports:/app/reports:ro,z \
  ghcr.io/${GHCR_USER}/churn-prediction-dashboard:latest
```

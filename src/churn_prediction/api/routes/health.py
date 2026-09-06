"""Health Check API Module.

This module provides a FastAPI router with endpoints to monitor the operational
status of the API. It verifies the availability of critical application state
dependencies, specifically the encoder and the machine learning model,
to ensure the service is ready to handle requests.
"""

from datetime import UTC, datetime

import psutil
from fastapi import APIRouter, Request, status
from pydantic import BaseModel, Field

# Initialize the API router for health checks
router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Response model for the health check endpoint."""

    status: str = Field(..., description="Overall health status of the API (HEALTHY or DEGRADED)")
    encoder_loaded: bool = Field(..., description="Indicates if the encoder is loaded")
    model_loaded: bool = Field(..., description="Indicates if the machine learning model is loaded")
    memory_usage_percent: float = Field(..., description="Current memory usage percentage")
    timestamp: str = Field(..., description="Timestamp of the health check")


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="System Health Check",
    description=(
        "Performs a diagnostic check of the API's operational state. "
        "This endpoint verifies that critical ML dependencies (encoder and booster model) "
        "are successfully loaded into memory and reports current system resource utilization. "
        "It is primarily used by orchestrators (e.g., Kubernetes) for liveness and readiness probes."
    ),
    responses={
        status.HTTP_200_OK: {
            "description": "Health check completed successfully. Status may be 'HEALTHY' or 'DEGRADED'.",
            "model": HealthResponse,
        },
    },
)
async def check_health(request: Request) -> HealthResponse:
    """Report API and machine-learning dependency health.

    Verifies that the preprocessing encoder and XGBoost model are available
    in application state and reports the current process memory usage and
    health-check timestamp.

    Parameters
    ----------
    request : fastapi.Request
        Incoming FastAPI request used to inspect application state.

    Returns
    -------
    HealthResponse
        Health information containing API status, encoder availability,
        model availability, memory utilization, and timestamp.
    """
    # Check if the encoder and model are loaded in the application state
    encoder_loaded = getattr(request.app.state, "encoder", None) is not None
    model_loaded = getattr(request.app.state, "booster", None) is not None
    is_healthy = encoder_loaded and model_loaded

    return HealthResponse(
        status="HEALTHY" if is_healthy else "DEGRADED",
        encoder_loaded=encoder_loaded,
        model_loaded=model_loaded,
        memory_usage_percent=psutil.virtual_memory().percent,
        timestamp=datetime.now(UTC).isoformat(),
    )

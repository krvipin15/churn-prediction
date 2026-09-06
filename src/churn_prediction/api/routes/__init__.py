"""
API Routes Module.

This module serves as the central entry point for all API endpoints in the
churn prediction service. It aggregates routers from various functional
domains—such as inference, explainability, and system health—to provide a
unified routing structure for the FastAPI application.

Routers
----------
explain :
    Contains endpoints for model explainability and feature importance analysis.
health :
    Contains endpoints for system diagnostics and dependency verification.
predict :
    Contains endpoints for batch prediction uploads and result downloads.
"""

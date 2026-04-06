"""
Server entry point for OpenEnv multi-mode deployment.
Launches the ClinicalTrialMatchEnv FastAPI server.
"""

import uvicorn


def main():
    """Start the ClinicalTrialMatchEnv server."""
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=7860,
    )


if __name__ == "__main__":
    main()

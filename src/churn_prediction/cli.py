"""Command-line interface for the churn prediction application.

This module provides a command-line interface (CLI) to run
the Streamlit application for churn prediction. It sets up
the necessary environment and invokes the Streamlit CLI to
launch the application.
"""

import sys
from pathlib import Path

from streamlit.web import cli as stcli


def main() -> None:
    """Launch the Streamlit dashboard application.

    Resolves the dashboard entrypoint, configures the Streamlit command-line
    arguments, and delegates application execution to Streamlit.

    Raises
    ------
    FileNotFoundError
        If the Streamlit dashboard entrypoint cannot be found.
    SystemExit
        Raised by the Streamlit command-line runner when the application exits.
    """
    client_main_path = (Path(__file__).parent / "client" / "main.py").resolve()

    # Check if the Streamlit application entrypoint exists
    if not Path(client_main_path).exists():
        msg = f"Streamlit application entrypoint not found at: {client_main_path}"
        raise FileNotFoundError(msg)

    # Set the command-line arguments to run the Streamlit application
    sys.argv = ["streamlit", "run", str(client_main_path)]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()

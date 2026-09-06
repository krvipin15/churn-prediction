"""Test the CLI entrypoint for the churn_prediction package."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from churn_prediction.cli import main


def test_main_success():
    """Test that main() correctly resolves the path and calls the streamlit CLI when the entrypoint exists."""
    # We patch the Path class entirely to control the behavior of any Path object created
    with patch("churn_prediction.cli.Path") as MockPath:
        mock_instance = MagicMock()
        MockPath.return_value = mock_instance

        # Handle the chaining: Path -> .parent -> / "client" -> / "main.py"
        mock_instance.parent = mock_instance
        mock_instance.__truediv__.return_value = mock_instance

        # Define the final behaviors
        mock_instance.exists.return_value = True
        mock_instance.resolve.return_value = "/fake/path/main.py"

        with patch("sys.exit") as mock_exit, patch("streamlit.web.cli.main") as mock_st_main:
            mock_st_main.return_value = 0

            main()

            assert sys.argv == ["streamlit", "run", "/fake/path/main.py"]
            mock_st_main.assert_called_once()
            mock_exit.assert_called_once_with(0)


def test_main_file_not_found():
    """Test that main() raises FileNotFoundError when the Streamlit entrypoint is missing."""
    with patch("churn_prediction.cli.Path") as MockPath:
        mock_instance = MagicMock()
        MockPath.return_value = mock_instance

        # Chain the divisions to return the same mock
        mock_instance.parent = mock_instance
        mock_instance.__truediv__.return_value = mock_instance

        # TRIGGER the error branch in cli.py
        mock_instance.exists.return_value = False

        with pytest.raises(FileNotFoundError) as excinfo:
            main()

        assert "Streamlit application entrypoint not found" in str(excinfo.value)

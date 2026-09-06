"""Test the Kaggle dataset ingestion module."""

import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from pytest import MonkeyPatch, fixture

from churn_prediction.data.ingestion import (
    _convert_csv_to_parquet_atomic,
    _has_valid_files,
    _is_valid_parquet,
    _remove_unwanted_paths,
    get_dataset,
)


@fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a temporary sample training CSV."""
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "target": [0, 1],
        }
    )

    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


@fixture
def sample_parquet(tmp_path: Path) -> Path:
    """Create a temporary valid Parquet dataset."""
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "target": [0, 1],
        }
    )

    parquet_path = tmp_path / "data.parquet"
    df.to_parquet(
        parquet_path,
        index=False,
    )

    return parquet_path


def test_is_valid_parquet(tmp_path: Path, sample_parquet: Path):
    """Validate valid, missing, empty, and corrupt Parquet files."""
    # Valid Parquet file.
    assert _is_valid_parquet(sample_parquet) is True

    # Missing Parquet file.
    missing_file = tmp_path / "ghost.parquet"
    assert _is_valid_parquet(missing_file) is False

    # Empty Parquet file.
    empty_file = tmp_path / "empty.parquet"
    empty_file.write_text("")
    assert _is_valid_parquet(empty_file) is False

    # Corrupt Parquet file.
    corrupt_file = tmp_path / "corrupt.parquet"
    corrupt_file.write_text("not a parquet content")
    assert _is_valid_parquet(corrupt_file) is False


def test_has_valid_files(tmp_path: Path, sample_parquet: Path):
    """Validate the presence and integrity of expected dataset files."""
    # Arrange: Create a valid dataset with expected files.
    sample_parquet.rename(tmp_path / "train.parquet")
    (tmp_path / "test.csv").write_text("id,target\n1,0\n")

    # Act & Assert: Check for valid files.
    expected_files = frozenset(
        {
            "train.parquet",
            "test.csv",
        }
    )

    assert (
        _has_valid_files(
            tmp_path,
            expected_files,
        )
        is True
    )

    assert (
        _has_valid_files(
            tmp_path,
            frozenset({"missing.csv"}),
        )
        is False
    )

    assert (
        _has_valid_files(
            tmp_path / "non_existent_dir",
            expected_files,
        )
        is False
    )


def test_has_valid_files_rejects_empty_required_csv(tmp_path: Path):
    """Reject a dataset when a required CSV file is empty."""
    # Arrange: Create a valid Parquet file and an empty CSV file.
    parquet_path = tmp_path / "train.parquet"

    pd.DataFrame(
        {
            "target": [0, 1],
        }
    ).to_parquet(
        parquet_path,
        index=False,
    )

    empty_csv = tmp_path / "test.csv"
    empty_csv.write_text("")

    # Act & Assert: Check for valid files.
    expected_files = frozenset(
        {
            "train.parquet",
            "test.csv",
        }
    )

    assert (
        _has_valid_files(
            tmp_path,
            expected_files,
        )
        is False
    )


def test_convert_csv_to_parquet_atomic(tmp_path: Path, sample_csv: Path):
    """Convert a CSV file to Parquet using atomic replacement."""
    # Arrange: Define the target Parquet file path.
    target_parquet = tmp_path / "target.parquet"

    # Act: Perform the CSV-to-Parquet conversion.
    _convert_csv_to_parquet_atomic(
        sample_csv,
        target_parquet,
    )

    # Assert: Verify that the Parquet file was created and is valid.
    assert target_parquet.exists()

    df = pd.read_parquet(target_parquet)

    assert not df.empty
    assert "id" not in df.columns
    assert list(df.columns) == ["target"]
    assert df["target"].tolist() == [0, 1]


def test_convert_csv_to_parquet_atomic_without_id(tmp_path: Path):
    """Convert a CSV that does not contain an ID column."""
    # Arrange: Create a sample CSV without an ID column.
    csv_path = tmp_path / "train.csv"

    pd.DataFrame(
        {
            "feature": [10, 20],
            "target": [0, 1],
        }
    ).to_csv(
        csv_path,
        index=False,
    )

    target_parquet = tmp_path / "target.parquet"

    # Act: Perform the CSV-to-Parquet conversion.
    _convert_csv_to_parquet_atomic(
        csv_path,
        target_parquet,
    )

    df = pd.read_parquet(target_parquet)

    # Assert: Verify that the Parquet file was created and is valid.
    assert list(df.columns) == ["feature", "target"]


def test_convert_csv_to_parquet_atomic_failure(tmp_path: Path):
    """Raise ValueError when CSV-to-Parquet conversion fails."""
    # Arrange: Define a non-existent CSV file path and a target Parquet file path.
    missing_csv = tmp_path / "non_existent.csv"
    target_parquet = tmp_path / "target.parquet"

    # Act & Assert: Expect a ValueError when attempting to convert a missing CSV file.
    with pytest.raises(
        ValueError,
        match="Conversion failed",
    ):
        _convert_csv_to_parquet_atomic(
            missing_csv,
            target_parquet,
        )

    assert not target_parquet.exists()


def test_remove_unwanted_paths(tmp_path: Path):
    """Remove unwanted artifacts while protecting paths outside the root."""
    # Arrange: Create a temporary directory with files and directories to be removed.
    (tmp_path / "train.csv").write_text("data")
    (tmp_path / "sample_submission.csv").write_text("data")
    (tmp_path / "wanted.txt").write_text("data")

    unwanted_dir = tmp_path / "unwanted_dir"
    unwanted_dir.mkdir()

    unwanted = frozenset(
        {
            "train.csv",
            "sample_submission.csv",
            "unwanted_dir",
        }
    )

    # Act: Remove unwanted paths.
    _remove_unwanted_paths(
        tmp_path,
        unwanted,
    )

    # Assert: Verify that unwanted paths were removed and wanted paths remain.
    assert not (tmp_path / "train.csv").exists()
    assert not (tmp_path / "sample_submission.csv").exists()
    assert not (tmp_path / "unwanted_dir").exists()
    assert (tmp_path / "wanted.txt").exists()

    # Security check: attempt to remove a path outside the dataset root.
    outside_file = tmp_path.parent / "outside.txt"
    outside_file.write_text("data")

    try:
        _remove_unwanted_paths(
            tmp_path,
            frozenset({"../outside.txt"}),
        )

        assert outside_file.exists()

    finally:
        outside_file.unlink(missing_ok=True)


def test_remove_unwanted_paths_missing_paths(tmp_path: Path):
    """Ignore unwanted paths that do not exist."""
    _remove_unwanted_paths(
        tmp_path,
        frozenset(
            {
                "missing.csv",
                "missing_dir",
            }
        ),
    )


def test_get_dataset_idempotency(tmp_path: Path, sample_parquet: Path):
    """Skip downloading when a valid local dataset already exists."""
    # Arrange: Create a valid Parquet file and a CSV file in the temporary directory.
    sample_parquet.rename(tmp_path / "train.parquet")

    (tmp_path / "test.csv").write_text("id,target\n1,0\n")

    expected_files = frozenset(
        {
            "train.parquet",
            "test.csv",
        }
    )

    # Act & Assert: Call get_dataset and verify that the download function is not called.
    with patch("churn_prediction.data.ingestion.kagglehub.competition_download") as mock_download:
        get_dataset(
            tmp_path,
            expected_files=expected_files,
        )

        mock_download.assert_not_called()


def test_get_dataset_missing_credentials(tmp_path: Path, monkeypatch: MonkeyPatch):
    """Raise ValueError when Kaggle credentials are unavailable."""
    # Arrange: Remove the Kaggle credentials from the environment.
    monkeypatch.delenv(
        "KAGGLE_USERNAME",
        raising=False,
    )
    monkeypatch.delenv(
        "KAGGLE_KEY",
        raising=False,
    )

    monkeypatch.setattr(
        "churn_prediction.data.ingestion.load_dotenv",
        lambda *args, **kwargs: None,  # noqa: ARG005
    )

    # Act & Assert: Expect a ValueError when attempting to download without credentials.
    with pytest.raises(
        ValueError,
        match="Missing Kaggle API credentials",
    ):
        get_dataset(
            tmp_path,
            force_download=True,
        )


def test_get_dataset_full_flow(tmp_path: Path, monkeypatch: MonkeyPatch):
    """Download, process, convert, and clean a mocked Kaggle dataset."""
    # Arrange: Set up fake Kaggle credentials and a mocked cache directory with dataset files.
    monkeypatch.setenv(
        "KAGGLE_USERNAME",
        "user",
    )
    monkeypatch.setenv(
        "KAGGLE_KEY",
        "key",
    )

    cache_dir = tmp_path / "kaggle_cache"
    cache_dir.mkdir()

    (cache_dir / "train.csv").write_text("id,target\n1,0\n2,1\n")
    (cache_dir / "test.csv").write_text("id,target\n3,0\n")
    (cache_dir / "sample_submission.csv").write_text("id,target\n3,1\n")
    (cache_dir / "other.txt").write_text("some info")

    # Act: Call get_dataset with force_download=True to trigger the full flow.
    with patch(
        "churn_prediction.data.ingestion.kagglehub.competition_download",
        return_value=str(cache_dir),
    ):
        get_dataset(
            tmp_path,
            force_download=True,
        )

    # Assert: Verify that the expected files were created and are valid.
    assert (tmp_path / "train.parquet").exists()
    assert (tmp_path / "test.csv").exists()
    assert (tmp_path / "other.txt").exists()
    assert (tmp_path / ".gitkeep").exists()

    # Verify Parquet is readable.
    train_df = pd.read_parquet(tmp_path / "train.parquet")

    assert not train_df.empty
    assert "target" in train_df.columns

    # Verify unwanted artifacts were not copied into output.
    assert not (tmp_path / "sample_submission.csv").exists()
    assert not (tmp_path / "train.csv").exists()


def test_get_dataset_download_failure(tmp_path: Path, monkeypatch: MonkeyPatch):
    """Raise RuntimeError when KaggleHub download fails."""
    # Arrange: Set up fake Kaggle credentials and mock the download function to raise an exception.
    monkeypatch.setenv(
        "KAGGLE_USERNAME",
        "user",
    )
    monkeypatch.setenv(
        "KAGGLE_KEY",
        "key",
    )

    # Act & Assert: Expect a RuntimeError when the download function raises an exception.
    with (
        patch(
            "churn_prediction.data.ingestion.kagglehub.competition_download",
            side_effect=Exception("Network Error"),
        ),
        pytest.raises(
            RuntimeError,
            match="Failed to download",
        ),
    ):
        get_dataset(
            tmp_path,
            force_download=True,
        )


def test_has_valid_files_rejects_corrupt_required_parquet(tmp_path: Path):
    """Reject a dataset when a required Parquet file is corrupt."""
    # Arrange: Create a corrupt Parquet file and a valid CSV file in the temporary directory.
    corrupt_parquet = tmp_path / "train.parquet"
    corrupt_parquet.write_text("not parquet")

    (tmp_path / "test.csv").write_text("id,target\n1,0\n")

    expected_files = frozenset(
        {
            "train.parquet",
            "test.csv",
        }
    )

    # Act & Assert: Check for valid files and expect the function to return False due to the corrupt Parquet file.
    assert _has_valid_files(tmp_path, expected_files) is False


def test_convert_csv_to_parquet_atomic_empty_dataframe(tmp_path: Path, monkeypatch: MonkeyPatch):
    """Handle a CSV that produces a DataFrame with no columns."""
    # Arrange: Create an empty CSV file and define the target Parquet file path.
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("")

    target_parquet = tmp_path / "target.parquet"

    monkeypatch.setattr(
        "churn_prediction.data.ingestion.pd.read_csv",
        lambda _: pd.DataFrame(),
    )

    # Act: Perform the CSV-to-Parquet conversion and expect it to handle the empty DataFrame gracefully.
    _convert_csv_to_parquet_atomic(csv_path, target_parquet)

    # Assert: Verify that the Parquet file was created and is empty.
    assert target_parquet.exists()
    assert pd.read_parquet(target_parquet).empty


def test_convert_csv_to_parquet_atomic_cleans_temp_file_when_no_temp_created(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
):
    """Wrap temporary-file creation failures in ValueError."""
    # Arrange: Create a sample CSV file and define the target Parquet file path.
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("id,target\n1,0\n")

    target_parquet = tmp_path / "target.parquet"

    def raise_os_error(*args, **kwargs):
        raise OSError("cannot create temporary file")

    monkeypatch.setattr(
        "churn_prediction.data.ingestion.tempfile.NamedTemporaryFile",
        raise_os_error,
    )

    # Act & Assert: Expect a ValueError when temporary file creation fails.
    with pytest.raises(ValueError, match="Conversion failed"):
        _convert_csv_to_parquet_atomic(csv_path, target_parquet)

    assert not target_parquet.exists()


def test_remove_unwanted_paths_handles_unrecognized_existing_path(tmp_path: Path):
    """Leave an existing path untouched when it is neither file nor directory."""
    # Arrange: Create a named pipe (FIFO) in the temporary directory.
    fifo = tmp_path / "special"
    os.mkfifo(fifo)

    # Act & Assert: Call _remove_unwanted_paths and verify that the FIFO still exists afterward.
    try:
        _remove_unwanted_paths(tmp_path, frozenset({"special"}))
        assert fifo.exists()
    finally:
        fifo.unlink(missing_ok=True)


def test_remove_unwanted_paths_logs_unlink_error(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
):
    """Handle an OSError raised while removing an unwanted file."""
    # Arrange: Create a file that will cause an OSError when attempted to be removed.
    unwanted = tmp_path / "bad.txt"
    unwanted.write_text("data")

    original_unlink = Path.unlink

    def raise_for_target(self, *, missing_ok=False):
        if self == unwanted:
            raise OSError("permission denied")
        return original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", raise_for_target)

    # Act: Call _remove_unwanted_paths and verify that the unwanted file still exists afterward.
    _remove_unwanted_paths(tmp_path, frozenset({"bad.txt"}))

    # Assert: The unwanted file should still exist due to the OSError during unlinking.
    assert unwanted.exists()


def test_remove_unwanted_paths_logs_rmtree_error(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
):
    """Handle an OSError raised while removing an unwanted directory."""
    # Arrange: Create a directory that will cause an OSError when attempted to be removed.
    unwanted_dir = tmp_path / "bad_dir"
    unwanted_dir.mkdir()

    def raise_os_error(_):
        raise OSError("permission denied")

    monkeypatch.setattr(
        "churn_prediction.data.ingestion.shutil.rmtree",
        raise_os_error,
    )

    # Act: Call _remove_unwanted_paths and verify that the unwanted directory still exists afterward.
    _remove_unwanted_paths(tmp_path, frozenset({"bad_dir"}))

    # Assert: The unwanted directory should still exist due to the OSError during rmtree.
    assert unwanted_dir.exists()


def test_get_dataset_missing_local_files_enters_download_flow(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
):
    """Continue to download when local expected artifacts are invalid."""
    # Arrange: Set up fake Kaggle credentials and a mocked cache directory with a missing file.
    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    monkeypatch.setenv("KAGGLE_KEY", "key")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "test.csv").write_text("id,target\n1,0\n")

    expected_files = frozenset(
        {
            "missing.parquet",
            "test.csv",
        }
    )

    # Act: Call get_dataset and verify that the download function is called due to missing expected files.
    with patch(
        "churn_prediction.data.ingestion.kagglehub.competition_download",
        return_value=str(cache_dir),
    ):
        get_dataset(
            tmp_path,
            expected_files=expected_files,
            force_download=False,
        )

    # Assert: Verify that the expected files were created and are valid.
    assert (tmp_path / "test.csv").exists()


def test_get_dataset_copies_directory_and_replaces_existing_directory(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
):
    """Copy cached directories and replace an existing destination directory."""
    # Arrange: Set up fake Kaggle credentials and a mocked cache directory with a source directory.
    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    monkeypatch.setenv("KAGGLE_KEY", "key")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    source_dir = cache_dir / "metadata"
    source_dir.mkdir()
    (source_dir / "source.txt").write_text("new")

    destination_dir = tmp_path / "metadata"
    destination_dir.mkdir()
    (destination_dir / "old.txt").write_text("old")

    # Act: Call get_dataset and verify that the source directory is copied and replaces the existing destination directory.
    with patch(
        "churn_prediction.data.ingestion.kagglehub.competition_download",
        return_value=str(cache_dir),
    ):
        get_dataset(tmp_path, force_download=True)

    # Assert: Verify that the source directory was copied and replaced the existing destination directory.
    assert (destination_dir / "source.txt").read_text() == "new"
    assert not (destination_dir / "old.txt").exists()


def test_get_dataset_processing_failure(tmp_path: Path, monkeypatch: MonkeyPatch):
    """Wrap cache-processing failures in RuntimeError."""
    # Arrange: Set up fake Kaggle credentials and a mocked cache directory that will cause an OSError during processing.
    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    monkeypatch.setenv("KAGGLE_KEY", "key")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Act & Assert: Expect a RuntimeError when processing the cache directory raises an OSError.
    with (
        patch(
            "churn_prediction.data.ingestion.kagglehub.competition_download",
            return_value=str(cache_dir),
        ),
        patch(
            "churn_prediction.data.ingestion.Path.iterdir",
            side_effect=OSError("cannot read cache"),
        ),
        pytest.raises(RuntimeError, match="Failed to download Kaggle competition"),
    ):
        get_dataset(tmp_path, force_download=True)


def test_get_dataset_preserves_existing_gitkeep(tmp_path: Path, monkeypatch: MonkeyPatch):
    """Do not recreate .gitkeep when it already exists."""
    # Arrange: Set up fake Kaggle credentials and create an existing .gitkeep file in the temporary directory.
    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    monkeypatch.setenv("KAGGLE_KEY", "key")

    gitkeep = tmp_path / ".gitkeep"
    gitkeep.write_text("")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Act: Call get_dataset and verify that the existing .gitkeep file is preserved.
    with patch(
        "churn_prediction.data.ingestion.kagglehub.competition_download",
        return_value=str(cache_dir),
    ):
        get_dataset(tmp_path, force_download=True)

    # Assert: Verify that the existing .gitkeep file still exists after processing.
    assert gitkeep.exists()


def test_get_dataset_handles_gitkeep_creation_error(tmp_path: Path, monkeypatch: MonkeyPatch):
    """Continue when .gitkeep cannot be created."""
    # Arrange: Set up fake Kaggle credentials and a mocked cache directory that will raise an OSError when attempting to create .gitkeep.
    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    monkeypatch.setenv("KAGGLE_KEY", "key")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    original_touch = Path.touch

    def raise_for_gitkeep(self, mode=0o666, *, exist_ok=False):
        if self.name == ".gitkeep":
            raise OSError("permission denied")
        return original_touch(self, mode=mode, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "touch", raise_for_gitkeep)

    # Act: Call get_dataset and verify that it continues processing even when .gitkeep creation fails.
    with patch(
        "churn_prediction.data.ingestion.kagglehub.competition_download",
        return_value=str(cache_dir),
    ):
        get_dataset(tmp_path, force_download=True)

    # Assert: Verify that .gitkeep does not exist due to the OSError during creation.
    assert not (tmp_path / ".gitkeep").exists()

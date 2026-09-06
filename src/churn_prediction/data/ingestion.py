"""Kaggle Dataset Acquisition Module.

Provides utilities for downloading and managing customer churn datasets
from Kaggle. It handles API authentication, idempotent downloads, and
cleanup of unnecessary artifacts to ensure a clean raw data directory.
"""

import os
import shutil
import tempfile
from pathlib import Path

import kagglehub
import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv
from pyarrow import ArrowInvalid

from churn_prediction.config.logger import get_logger
from churn_prediction.config.settings import get_settings

# Initialize settings
settings = get_settings()

# Module-level immutable constants
DEFAULT_EXPECTED_FILES = frozenset({"train.parquet", "test.csv"})
UNWANTED_DATA = frozenset({"sample_submission.csv", "train.csv", ".complete"})


def _is_valid_parquet(file_path: Path) -> bool:
    """Check whether a Parquet file is present and structurally valid.

    Verifies that the file exists, is non-empty, and can be opened as a
    valid Parquet dataset with readable metadata.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the Parquet file to validate.

    Returns
    -------
    bool
        ``True`` when the file exists, is non-empty, and has valid Parquet
        metadata; otherwise ``False``.
    """
    logger = get_logger()

    if not file_path.exists():
        logger.debug("Parquet validation failed: File does not exist at %s", file_path)
        return False

    if file_path.stat().st_size == 0:
        logger.debug("Parquet validation failed: File is empty at %s", file_path)
        return False

    try:
        _ = pq.read_metadata(file_path)
        return True
    except (ArrowInvalid, OSError) as err:
        logger.debug(
            "Parquet validation failed: Invalid metadata for %s. Error: %s", file_path, err
        )
        return False


def _has_valid_files(
    target_dir: Path,
    expected_files: frozenset[str],
) -> bool:
    """Validate the expected dataset files in a directory.

    Parameters
    ----------
    target_dir : pathlib.Path
        Directory containing the downloaded dataset artifacts.
    expected_files : frozenset of str
        Filenames that must be present and valid.

    Returns
    -------
    bool
        ``True`` when every expected file exists and passes its integrity
        checks; otherwise ``False``.
    """
    logger = get_logger()

    if not target_dir.is_dir():
        logger.warning("Target directory does not exist or is not a directory: %s", target_dir)
        return False

    logger.debug("Validating existence of expected files in %s: %s", target_dir, expected_files)
    for required_file in expected_files:
        file_path = target_dir / required_file

        if required_file.endswith(".parquet"):
            if not _is_valid_parquet(file_path):
                logger.warning(
                    "Required Parquet file '%s' is missing or corrupt in %s",
                    required_file,
                    target_dir,
                )
                return False
        elif not file_path.exists() or file_path.stat().st_size == 0:
            logger.warning(
                "Required file '%s' is missing or empty in %s", required_file, target_dir
            )
            return False

    return True


def _convert_csv_to_parquet_atomic(csv_file: Path, target_parquet: Path) -> None:
    """Convert a CSV dataset to Parquet using an atomic file replacement.

    The CSV is loaded and written to a temporary Parquet file first. The
    completed temporary file is then moved into its final destination,
    preventing partially written Parquet artifacts from being exposed.

    Parameters
    ----------
    csv_file : pathlib.Path
        Path to the source CSV file.
    target_parquet : pathlib.Path
        Destination path for the resulting Parquet file.

    Raises
    ------
    ValueError
        If the source CSV cannot be parsed or does not contain the expected
        dataset structure.
    OSError
        If the temporary or destination file cannot be created or replaced.
    """
    logger = get_logger()
    temp_dir = target_parquet.parent

    try:
        logger.debug("Starting atomic conversion: %s -> %s", csv_file.name, target_parquet.name)
        # Write to temporary file first
        with tempfile.NamedTemporaryFile(dir=temp_dir, suffix=".tmp", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        # Load the CSV file
        df = pd.read_csv(csv_file)
        logger.debug("Read CSV %s successfully. Shape: %s", csv_file.name, df.shape)

        # Remove `id` column if it exists
        try:
            if "id" in df.columns[0].lower():
                df = df.drop(df.columns[0], axis=1)
        except IndexError:
            logger.warning("DataFrame has no columns to check for an id")

        # Write to Parquet
        df.to_parquet(tmp_path, index=False, engine="pyarrow", compression="snappy")

        # Atomic replacement guarantees no partial files exist on disk
        tmp_path.replace(target_parquet)
        logger.info("Successfully converted %s to %s", csv_file.name, target_parquet.name)

    except Exception as err:
        if "tmp_path" in locals() and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        logger.error("Failed to convert %s to Parquet: %s", csv_file.name, err)
        raise ValueError(f"Conversion failed for {csv_file}: {err}") from err


def _remove_unwanted_paths(
    dataset_path: Path,
    unwanted_data: frozenset[str],
) -> None:
    """Remove unwanted files and directories from a dataset directory.

    Only paths whose names are explicitly included in ``unwanted_data`` are
    considered for removal.

    Parameters
    ----------
    dataset_path : pathlib.Path
        Root directory containing the downloaded dataset.
    unwanted_data : frozenset of str
        Names of files, symbolic links, or directories that should be removed.

    Raises
    ------
    OSError
        If an unwanted path cannot be removed because of permissions or
        another operating-system error.
    """
    logger = get_logger()
    resolved_root = dataset_path.resolve()
    logger.debug("Cleaning up unwanted artifacts in %s", resolved_root)

    for name in unwanted_data:
        target = (dataset_path / name).resolve()
        logger.debug("Checking path for cleanup: %s", target)

        if not target.is_relative_to(resolved_root):
            logger.warning("Skipping cleanup for path outside target root: %s", target)
            continue

        if not target.exists() and not target.is_symlink():
            logger.debug("Unwanted artifact %s not found; skipping.", name)
            continue

        try:
            if target.is_file() or target.is_symlink():
                target.unlink(missing_ok=True)
                logger.info("Removed unwanted file artifact: %s", target.name)
            elif target.is_dir():
                shutil.rmtree(target)
                logger.info("Removed unwanted directory artifact: %s", target.name)
        except OSError as err:
            logger.warning("Failed to remove path %s: %s", target, err)


def get_dataset(
    output_dir: Path | str,
    expected_files: frozenset[str] | None = None,
    competition_name: str = "playground-series-s6e3",
    *,
    force_download: bool = False,
) -> None:
    """Download and prepare the Kaggle competition dataset.

    The function operates idempotently: when all expected dataset artifacts
    already exist locally and pass integrity checks, no download is performed.
    Otherwise, the dataset is downloaded, converted into the required format,
    and cleaned of unnecessary files.

    Parameters
    ----------
    output_dir : pathlib.Path or str
        Destination directory for the dataset artifacts.
    expected_files : frozenset of str or None, optional
        Set of filenames required for a valid local dataset. If ``None``,
        the module's default expected file set is used.
    competition_name : str, default="playground-series-s6e3"
        Kaggle competition identifier used to retrieve the dataset.
    force_download : bool, default=False
        Whether to force dataset retrieval and processing even when the
        existing local files pass validation.

    Raises
    ------
    ValueError
        If required Kaggle credentials are unavailable.
    RuntimeError
        If downloading, extracting, converting, or preparing the dataset
        fails.
    OSError
        If required directories or dataset files cannot be created or modified.
    """
    logger = get_logger()
    load_dotenv(override=False)

    output_dir = Path(output_dir).expanduser().resolve()
    targets = expected_files if expected_files is not None else DEFAULT_EXPECTED_FILES

    # 0. Check if dataset already exists locally and is non-corrupt
    if not force_download:
        if _has_valid_files(output_dir, targets):
            logger.info("Valid dataset already exists at %s. Skipping download.", output_dir)
            return
        logger.debug(
            "Local dataset at %s is missing or invalid. Proceeding with download.", output_dir
        )
    else:
        logger.debug("force_download=True: bypassing local validation for %s", output_dir)

    # 1. Validate and configure Kaggle API credentials
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if not username or not key:
        logger.error("Kaggle credentials missing. KAGGLE_USERNAME or KAGGLE_KEY not set.")
        raise ValueError(
            "Missing Kaggle API credentials. Set KAGGLE_USERNAME and KAGGLE_KEY in your environment or .env file."
        )

    # Set Kaggle API credentials if not already set
    os.environ["KAGGLE_API_TOKEN"] = key
    logger.info("Configured Kaggle API environment credentials for user: %s", username)

    # 2. Download competition files via KaggleHub
    try:
        logger.info("Fetching Kaggle competition dataset '%s' via KaggleHub...", competition_name)
        cache_path_str = kagglehub.competition_download(
            competition_name,
            force_download=force_download,
        )
        cache_path = Path(cache_path_str).expanduser().resolve()
    except Exception as exc:
        logger.error("KaggleHub competition download failed for '%s'.", competition_name)
        raise RuntimeError(f"Failed to download Kaggle competition '{competition_name}'.") from exc

    # 3. Efficient Processing directly from Cache into output_dir
    try:
        logger.debug("Processing items from cache %s to output %s", cache_path, output_dir)

        for item in cache_path.iterdir():
            if item.suffix.lower() == ".csv":
                if item.name == "sample_submission.csv":
                    logger.debug("Skipping conversion for %s (ignored file)", item.name)
                    continue

                # ONLY convert train.csv; preserve test.csv
                if item.name == "test.csv":
                    dest_item = output_dir / "test.csv"
                    shutil.copy2(item, dest_item)
                    logger.info("Preserved CSV file %s -> %s", item.name, output_dir)
                else:
                    parquet_dest = output_dir / f"{item.stem}.parquet"
                    _convert_csv_to_parquet_atomic(item, parquet_dest)

            elif item.is_file():
                dest_item = output_dir / item.name
                shutil.copy2(item, dest_item)
                logger.debug("Copied file %s -> %s", item.name, output_dir)
            elif item.is_dir():
                dest_item = output_dir / item.name
                if dest_item.exists():
                    logger.debug("Removing existing directory %s before copy", dest_item)
                    shutil.rmtree(dest_item)
                shutil.copytree(item, dest_item)

    except Exception as exc:
        logger.exception("KaggleHub competition download failed for '%s'.", competition_name)
        raise RuntimeError(
            f"Failed to download Kaggle competition '{competition_name}'. Check credentials or network connectivity."
        ) from exc

    # 4. Maintain Git structure
    gitkeep = output_dir / ".gitkeep"
    if not gitkeep.exists():
        try:
            gitkeep.touch(exist_ok=True)
            logger.debug("Created .gitkeep in %s", output_dir)
        except OSError as err:
            logger.warning("Could not create .gitkeep file at %s: %s", output_dir, err)

    # 5. Clean up unwanted artifacts in local output_dir (leaving cache pristine)
    _remove_unwanted_paths(output_dir, UNWANTED_DATA)
    logger.info("Dataset successfully prepared and cleaned at %s", output_dir)

"""Dataset Validation Module.

Implements the validation logic for churn prediction datasets using
Pandera schemas. It provides a workflow to load tabular data, verify its
integrity against structural constraints, and persist detailed diagnostic
reports in JSON format.
"""

from pathlib import Path
from typing import Any

import orjson
import pandas as pd
import pandera.errors as pa_errors
import pandera.pandas as pa

from churn_prediction.config.logger import get_logger


def _extract_failures(
    err: pa_errors.SchemaError | pa_errors.SchemaErrors,
) -> list[dict[str, Any]]:
    """Convert Pandera validation errors into standardized records.

    Extracts the relevant failure information from a Pandera schema
    exception and converts it into JSON-serializable dictionaries suitable
    for validation reports.

    Parameters
    ----------
    err : pandera.errors.SchemaError or pandera.errors.SchemaErrors
        Pandera validation exception containing one or more schema failures.

    Returns
    -------
    list of dict
        Validation failure records containing standardized information about
        the failed checks, columns, indexes, and failure cases.
    """
    logger = get_logger()
    failures: list[dict[str, Any]] = []

    if isinstance(err, pa_errors.SchemaErrors):
        logger.debug("Extracting multiple validation failures from SchemaErrors")
        failure_cases = err.failure_cases
        if failure_cases is not None and not failure_cases.empty:
            records = failure_cases.to_dict(orient="records")
            for record in records:
                col = record.get("column")
                check = record.get("check")
                failures.append(
                    {
                        "column": str(col) if pd.notna(col) else None,
                        "check": str(check) if pd.notna(check) else None,
                        "failure_case": str(record.get("failure_case"))
                        if pd.notna(record.get("failure_case"))
                        else None,
                        "index": str(record.get("index"))
                        if pd.notna(record.get("index"))
                        else None,
                        "error": f"Check '{check}' failed for column '{col}'",
                    }
                )
        return failures

    # Handle single SchemaError
    logger.debug("Extracting single validation failure from SchemaError")
    schema_obj = getattr(err, "schema", None)
    col_name = getattr(schema_obj, "name", None) if schema_obj is not None else None
    check_name = (
        str(getattr(err, "check", None)) if getattr(err, "check", None) is not None else None
    )

    failures.append(
        {
            "column": str(col_name) if col_name is not None else None,
            "check": check_name,
            "failure_case": str(err.failure_cases)
            if hasattr(err, "failure_cases") and err.failure_cases is not None
            else None,
            "index": None,
            "error": str(err),
        }
    )

    return failures


def _write_report(
    report_path: Path,
    report_data: dict[str, Any],
) -> None:
    """Write a validation report to disk as formatted JSON.

    Parameters
    ----------
    report_path : pathlib.Path
        Destination path for the validation report.
    report_data : dict
        JSON-serializable validation report contents.

    Raises
    ------
    OSError
        If the report directory cannot be created or the report cannot be
        written to disk.
    """
    logger = get_logger()

    try:
        logger.debug("Serializing validation report to JSON for path: %s", report_path)
        report_bytes = orjson.dumps(report_data, option=orjson.OPT_INDENT_2, default=str)
        report_path.write_bytes(report_bytes)
        logger.info("Validation report successfully saved to %s", report_path)
    except Exception as err:
        logger.error("Failed to write validation report to %s: %s", report_path, err)
        raise OSError(f"Could not persist validation report to {report_path}: {err}") from err


def validate_dataset(
    dataset_path: str | Path,
    report_path: str | Path,
    schema: pa.DataFrameSchema,
    *,
    lazy: bool = True,
) -> None:
    """Validate a tabular dataset against a Pandera schema.

    Loads the dataset according to its file format, validates its structure
    and values against the supplied Pandera schema, and writes a diagnostic
    JSON report describing the validation result.

    Parameters
    ----------
    dataset_path : str or pathlib.Path
        Path to the dataset being validated. Supported formats depend on the
        dataset loading implementation.
    report_path : str or pathlib.Path
        Destination path for the generated JSON validation report.
    schema : pandera.DataFrameSchema
        Pandera schema defining the expected columns, data types, nullability,
        uniqueness constraints, and value restrictions.
    lazy : bool, default=True
        Whether Pandera should collect all validation failures before raising
        an exception. When ``True``, the resulting report contains all
        detected validation failures.

    Raises
    ------
    FileNotFoundError
        If ``dataset_path`` does not exist.
    ValueError
        If the dataset cannot be loaded or violates the supplied schema.
    OSError
        If the validation report cannot be written to disk.
    """
    logger = get_logger()
    dataset_path = Path(dataset_path).expanduser().resolve()
    report_path = Path(report_path).expanduser().resolve()

    # 0. Validate input path existence
    if not dataset_path.exists():
        logger.error("Dataset path does not exist: %s", dataset_path)
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    if not dataset_path.is_file():
        logger.error("Dataset path is not a valid file: %s", dataset_path)
        raise ValueError(f"Target path is a directory or special file: {dataset_path}")

    # 1. Load the dataset into a Pandas DataFrame
    try:
        logger.debug("Loading dataset from %s into Pandas DataFrame", dataset_path)
        df = pd.read_parquet(dataset_path)
    except Exception as err:
        logger.exception("Failed to load dataset file %s: %s", dataset_path, err)
        raise ValueError(f"Failed to load dataset from {dataset_path}: {err}") from err

    logger.info(
        "Successfully loaded dataset from %s (rows=%d, cols=%d, memory=%.2f MB)",
        dataset_path,
        len(df),
        len(df.columns),
        df.memory_usage(deep=True).sum() / (1024 * 1024),
    )

    # 2. Prepare Diagnostic Structure
    report: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "schema_name": schema.name,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns_present": list(df.columns),
        "status": "PASSED",
        "failure_count": 0,
        "failures": [],
    }

    # 3. Perform Validation
    try:
        logger.debug("Executing schema validation (lazy=%s) for %s", lazy, schema.name)
        schema.validate(df, lazy=lazy)
        logger.info(
            "Dataframe validation PASSED for %s against schema context: %s",
            dataset_path.name,
            schema.name,
        )
    except (pa_errors.SchemaError, pa_errors.SchemaErrors) as err:
        report["status"] = "FAILED"
        failures = _extract_failures(err)
        report["failures"] = failures
        report["failure_count"] = len(failures)

        # Extract specific failing columns for high-level logging summary
        failed_columns: set[str] = {
            str(f["column"]) for f in failures if f.get("column") is not None
        }

        logger.error(
            "Dataframe validation FAILED for %s with %d violation(s) across %d column(s) %s",
            dataset_path.name,
            report["failure_count"],
            len(failed_columns),
            sorted(failed_columns) if failed_columns else [],
        )

        # 4. Persist Diagnostic Report
        _write_report(report_path, report)

    # 5. Raise Exception if Validation Failed
    if report["status"] == "FAILED":
        raise ValueError(
            f"Dataset validation failed with {report['failure_count']} violation(s). "
            f"Detailed diagnostic report persisted to: {report_path}"
        )

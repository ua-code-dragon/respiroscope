from __future__ import annotations
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd
from tqdm.auto import tqdm

RecordProcessor = Callable[[dict[str, Any], dict[str, Any]], list[dict[str, Any]]]


class TRSDataset:
    """Loader for Temporal-Respiratory-Support style CSV records."""

    DEFAULT_DATA_DIR = Path(
        "data/a-temporal-dataset-for-respiratory-support-in-critically-ill-patients-1.1.0"
    )

    REQUIRED_COLUMNS = ("patient_id", "time", "SpO2", "FiO2")
    DEFAULT_MIN_LINES = 48

    _PATIENT_ID_ALIASES = {"patient_id", "subject_id", "id"}
    _TIME_ALIASES = {"hr", "time", "t"}
    _SPO2_ALIASES = {"spo2", "spo_2", "sao2", "oxygen_saturation"}
    _FIO2_ALIASES = {"fio2", "fio_2", "set_fio2", "fraction_inspired_oxygen"}
    _LOS_ALIASES = {"los"}

    def __init__(
        self,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        cache_dir: str | Path = ".",
        record_processor: Optional[RecordProcessor] = None,
        min_lines: int = DEFAULT_MIN_LINES,
        enable_progress: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.record_processor = record_processor or self.default_record_processor
        self.min_lines = min_lines
        self.enable_progress = enable_progress
        self.logger = logger or logging.getLogger(f"{__name__}.TRSDataset")
        self._use_default_toss_validation = record_processor is None

    def load_all(
        self,
        pattern: str = "*.csv",
        *,
        max_workers: Optional[int] = None,
        use_cache: bool = True,
        force_reload: bool = False,
        index_filename: str = "dataset_index.csv",
        cache_filename: str = "dataset_cache.h5",
        cache_key: str = "trs_dataset",
        max_records: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load all CSV files into one normalized DataFrame with cache support."""
        files = sorted(self.data_dir.rglob(pattern))
        if not files:
            raise FileNotFoundError(f"No CSV files found under: {self.data_dir}")
        self.logger.info("Discovered %d CSV files under %s", len(files), self.data_dir)

        index_path = self.cache_dir / index_filename
        cache_path = self.cache_dir / cache_filename

        current_index = self._build_index(files)

        if use_cache and not force_reload and index_path.exists() and cache_path.exists():
            saved_index = self._read_index(index_path)
            if self._index_matches(current_index, saved_index):
                try:
                    self.logger.info("Cache hit: loading dataset from %s", cache_path)
                    return pd.read_hdf(cache_path, key=cache_key)
                except ImportError as exc:
                    raise ImportError(
                        "HDF5 cache requires the 'tables' package. Install it or call load_all(..., use_cache=False)."
                    ) from exc

        frames: list[pd.DataFrame] = []
        n_processed = 0
        show_record_progress = bool(self.enable_progress and max_workers == 1)
        if max_workers == 1:
            iterator = tqdm(
                files,
                total=len(files),
                desc="Loading CSV files",
                unit="file",
                disable=not self.enable_progress,
            )
            for file_path in iterator:
                frames.append(self.load_file(file_path, show_progress=show_record_progress))
                n_processed += 1
                if max_records is not None and n_processed >= max_records:
                    self.logger.info("Reached max_records limit (%d), stopping load.", max_records)
                    break
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(self.load_file, file_path): file_path for file_path in files}
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Loading/preprocessing CSV files",
                    unit="file",
                    disable=not self.enable_progress,
                ):
                    file_path = futures[future]
                    try:
                        frames.append(future.result())
                        n_processed += 1
                        if max_records is not None and n_processed >= max_records:
                            self.logger.info("Reached max_records limit (%d), stopping load.", max_records)
                            break
                    except Exception:
                        self.logger.exception("Failed while processing file: %s", file_path)
                        raise

        df = pd.concat(frames, ignore_index=True, sort=False)
        self.logger.info("Loaded %d records after preprocessing", len(df))

        # Persist cache artifacts for fast reloads.
        current_index.to_csv(index_path, index=False)
        if use_cache:
            try:
                df.to_hdf(cache_path, key=cache_key, mode="w")
                self.logger.info("Updated cache at %s", cache_path)
            except ImportError as exc:
                raise ImportError(
                    "HDF5 cache requires the 'tables' package. Install it or call load_all(..., use_cache=False)."
                ) from exc

        return df

    def load_file(self, file_path: str | Path, *, show_progress: bool = False) -> pd.DataFrame:
        """Load and normalize a single CSV file."""
        path = Path(file_path)
        df = pd.read_csv(path)

        if df.empty:
            self.logger.warning("Skipping empty file: %s", path)
            return pd.DataFrame(columns=list(self.REQUIRED_COLUMNS))

        original_columns = list(df.columns)
        column_lookup = {str(c).strip().lower(): c for c in original_columns}

        patient_col = self._find_column(column_lookup, self._PATIENT_ID_ALIASES)
        time_col = self._find_column(column_lookup, self._TIME_ALIASES)
        spo2_col = self._find_column(column_lookup, self._SPO2_ALIASES)
        fio2_col = self._find_column(column_lookup, self._FIO2_ALIASES)

        if spo2_col is None:
            raise ValueError(f"Missing SpO2 column in file: {path}")
        if fio2_col is None:
            raise ValueError(f"Missing FiO2 column in file: {path}")
        los_col = self._find_column(column_lookup, self._LOS_ALIASES)

        if self._use_default_toss_validation:
            ok, reason, rows_limit = self._toss_validate_file(df, los_col, fio2_col, spo2_col)
            if not ok:
                self.logger.warning("Skipping file %s: %s", path, reason)
                return pd.DataFrame(columns=list(self.REQUIRED_COLUMNS))
            if rows_limit < len(df):
                df = df.iloc[:rows_limit].copy()

        if patient_col is None:
            df["patient_id"] = path.stem
        else:
            df["patient_id"] = df[patient_col]

        if time_col is None:
            df["time"] = pd.RangeIndex(start=0, stop=len(df), step=1)
        else:
            # parsed_time = pd.to_datetime(df[time_col], errors="coerce")
            # if parsed_time.notna().any():
            #    df["time"] = parsed_time
            # else:
            #    df["time"] = pd.to_numeric(df[time_col], errors="coerce")
            df["time"] = pd.to_numeric(df[time_col], errors="coerce")

        processed = self._apply_record_processor(
            df=df,
            path=path,
            spo2_col=spo2_col,
            fio2_col=fio2_col,
            show_progress=show_progress,
        )
        if processed.empty:
            self.logger.warning("No valid records left after preprocessing: %s", path)
            return pd.DataFrame(columns=list(self.REQUIRED_COLUMNS))

        processed["SpO2"] = pd.to_numeric(processed["SpO2"], errors="coerce")
        processed["FiO2"] = self._normalize_fio2(pd.to_numeric(processed["FiO2"], errors="coerce"))
        processed = processed.dropna(subset=["SpO2", "FiO2"])
        if len(processed) < 4:
            self.logger.warning("Skipping file %s: less than 4 valid rows after preprocessing", path)
            return pd.DataFrame(columns=list(self.REQUIRED_COLUMNS))

        required = list(self.REQUIRED_COLUMNS)
        optional = [c for c in original_columns if c not in {patient_col, time_col, spo2_col, fio2_col}]
        ordered_cols = required + optional

        return processed.reindex(columns=ordered_cols)

    def _apply_record_processor(
        self,
        *,
        df: pd.DataFrame,
        path: Path,
        spo2_col: str,
        fio2_col: str,
        show_progress: bool,
    ) -> pd.DataFrame:
        state: dict[str, Any] = {
            "path": path,
            "last_fio2": None,
            "pending_records": [],
        }
        output_rows: list[dict[str, Any]] = []

        iterator = df.iterrows()
        use_tqdm = bool(
            show_progress and self.enable_progress and threading.current_thread() is threading.main_thread()
        )
        if use_tqdm:
            iterator = tqdm(
                iterator,
                total=len(df),
                desc=f"Preprocessing {path.name}",
                unit="row",
                leave=False,
            )

        for row_index, row in iterator:
            record = row.to_dict()
            record["FiO2"] = row.get(fio2_col)
            record["SpO2"] = row.get(spo2_col)
            record["row_index"] = row_index
            emitted = self.record_processor(record, state)
            if emitted:
                output_rows.extend(emitted)

        pending = state.get("pending_records")
        if pending:
            # Matching fill.py behavior: unresolved initial rows stay dropped if FiO2 never appears.
            self.logger.debug("Dropping %d unresolved pending records from %s", len(pending), path)

        return pd.DataFrame(output_rows)

    def _toss_validate_file(
        self, df: pd.DataFrame, los_col: Optional[str], fio2_col: str, spo2_col: str
    ) -> tuple[bool, str, int]:
        """Adapted from toss.py _data_lines: LOS-based trimming and minimal length check."""
        if df.empty:
            return False, "Empty file", 0

        first = df.iloc[0]
        required = [los_col, fio2_col, spo2_col]
        if any(col is None for col in required):
            return False, "Missing required toss columns (los/set_fio2/spo2)", 0

        los_value = self._to_float(first.get(los_col))
        if los_value is None:
            return False, f"Invalid LOS value: {first.get(los_col)}", 0

        rows_limit = math.ceil(los_value * 24.0)
        if rows_limit < self.min_lines:
            return False, f"Short file ({los_value=} {rows_limit=})", rows_limit
        return True, "", rows_limit

    @staticmethod
    def default_record_processor(record: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
        """Default per-record preprocessing adapted from fill.py rules."""
        fio2 = TRSDataset._to_float(record.get("FiO2"))
        spo2 = TRSDataset._to_float(record.get("SpO2"))

        if fio2 is not None and (fio2 < 20 or fio2 > 100.0):
            return []
        if spo2 is None or spo2 < 80 or spo2 > 100.0:
            return []

        pending: list[dict[str, Any]] = state.setdefault("pending_records", [])
        emitted: list[dict[str, Any]] = []

        if fio2 is None:
            if state.get("last_fio2") is None:
                pending_record = dict(record)
                pending_record["SpO2"] = spo2
                pending_record["FiO2"] = None
                pending.append(pending_record)
                return []
            fio2 = state["last_fio2"]
        else:
            state["last_fio2"] = fio2
            if pending:
                for pending_record in pending:
                    pending_record["FiO2"] = fio2
                emitted.extend(pending)
                pending.clear()

        cleaned = dict(record)
        cleaned["FiO2"] = fio2
        cleaned["SpO2"] = spo2
        emitted.append(cleaned)
        return emitted

    def _build_index(self, files: list[Path]) -> pd.DataFrame:
        rows = []
        for path in files:
            stat = path.stat()
            rows.append(
                {
                    "patient_id": path.stem,
                    "relative_path": path.relative_to(self.data_dir).as_posix(),
                    "file_size": int(stat.st_size),
                    "modified_time_ns": int(stat.st_mtime_ns),
                }
            )

        return pd.DataFrame(rows).sort_values("relative_path").reset_index(drop=True)

    @staticmethod
    def _read_index(index_path: Path) -> pd.DataFrame:
        df = pd.read_csv(index_path)
        expected = ["patient_id", "relative_path", "file_size", "modified_time_ns"]
        missing = [col for col in expected if col not in df.columns]
        if missing:
            return pd.DataFrame(columns=expected)

        df = df[expected].copy()
        df["patient_id"] = df["patient_id"].astype(str)
        df["relative_path"] = df["relative_path"].astype(str)
        df["file_size"] = pd.to_numeric(df["file_size"], errors="coerce").fillna(-1).astype("int64")
        df["modified_time_ns"] = (
            pd.to_numeric(df["modified_time_ns"], errors="coerce").fillna(-1).astype("int64")
        )
        return df.sort_values("relative_path").reset_index(drop=True)

    @staticmethod
    def _index_matches(current: pd.DataFrame, saved: pd.DataFrame) -> bool:
        if list(current.columns) != list(saved.columns):
            return False
        if len(current) != len(saved):
            return False
        return current.equals(saved)

    @staticmethod
    def _find_column(column_lookup: Dict[str, str], aliases: set[str]) -> Optional[str]:
        for alias in aliases:
            if alias in column_lookup:
                return column_lookup[alias]
        return None

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if pd.isna(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_fio2(series: pd.Series) -> pd.Series:
        """Normalize FiO2 to fraction range [0, 1] when values look like percentages."""
        normalized = series.copy()
        valid = normalized.dropna()
        if valid.empty:
            return normalized

        # If typical values are above 1, data is likely in percent scale (21..100).
        if valid.quantile(0.5) > 1.5:
            return normalized / 100.0

        # Mixed scale fallback: convert only values clearly above fraction range.
        normalized = normalized.where((normalized <= 1.5) | normalized.isna(), normalized / 100.0)
        return normalized


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    loader = TRSDataset()
    start_time = time.perf_counter()
    # df = loader.load_all(max_workers=8, use_cache=False)
    # df = loader.load_all(max_workers=8, use_cache=True)
    df = loader.load_all(max_workers=1, use_cache=False, max_records=100)
    elapsed = time.perf_counter() - start_time
    print(f"Loaded dataset with {len(df)} records in {elapsed:.2f} seconds.\n")
    print(df.head())
    print(df.columns.tolist())

from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


class TRSDataset:
    """Loader for Temporal-Respiratory-Support style CSV records."""

    REQUIRED_COLUMNS = ("patient_id", "time", "SpO2", "FiO2")

    _PATIENT_ID_ALIASES = {"patient_id", "patientid", "subject_id", "id"}
    _TIME_ALIASES = {"time", "timestamp", "datetime", "dt", "charttime", "t"}
    _SPO2_ALIASES = {"spo2", "spo_2", "sao2", "oxygen_saturation"}
    _FIO2_ALIASES = {"fio2", "fio_2", "fraction_inspired_oxygen"}

    def __init__(self, data_dir: str | Path = "work/stat", cache_dir: str | Path = ".") -> None:
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)

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
    ) -> pd.DataFrame:
        """Load all CSV files into one normalized DataFrame with cache support."""
        files = sorted(self.data_dir.rglob(pattern))
        if not files:
            raise FileNotFoundError(f"No CSV files found under: {self.data_dir}")

        index_path = self.cache_dir / index_filename
        cache_path = self.cache_dir / cache_filename

        current_index = self._build_index(files)

        if use_cache and not force_reload and index_path.exists() and cache_path.exists():
            saved_index = self._read_index(index_path)
            if self._index_matches(current_index, saved_index):
                try:
                    return pd.read_hdf(cache_path, key=cache_key)
                except ImportError as exc:
                    raise ImportError(
                        "HDF5 cache requires the 'tables' package. Install it or call load_all(..., use_cache=False)."
                    ) from exc

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            frames = list(pool.map(self.load_file, files))

        df = pd.concat(frames, ignore_index=True, sort=False)

        # Persist cache artifacts for fast reloads.
        current_index.to_csv(index_path, index=False)
        if use_cache:
            try:
                df.to_hdf(cache_path, key=cache_key, mode="w")
            except ImportError as exc:
                raise ImportError(
                    "HDF5 cache requires the 'tables' package. Install it or call load_all(..., use_cache=False)."
                ) from exc

        return df

    def load_file(self, file_path: str | Path) -> pd.DataFrame:
        """Load and normalize a single CSV file."""
        path = Path(file_path)
        df = pd.read_csv(path)

        if df.empty:
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

        if patient_col is None:
            df["patient_id"] = path.stem
        else:
            df["patient_id"] = df[patient_col]

        if time_col is None:
            df["time"] = pd.RangeIndex(start=0, stop=len(df), step=1)
        else:
            parsed_time = pd.to_datetime(df[time_col], errors="coerce")
            if parsed_time.notna().any():
                df["time"] = parsed_time
            else:
                df["time"] = pd.to_numeric(df[time_col], errors="coerce")

        df["SpO2"] = pd.to_numeric(df[spo2_col], errors="coerce")
        df["FiO2"] = self._normalize_fio2(pd.to_numeric(df[fio2_col], errors="coerce"))

        required = list(self.REQUIRED_COLUMNS)
        optional = [c for c in original_columns if c not in {patient_col, time_col, spo2_col, fio2_col}]
        ordered_cols = required + optional

        return df.reindex(columns=ordered_cols)

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
    loader = TRSDataset("work/stat")
    start_time = time.perf_counter()
    # df = loader.load_all(max_workers=8, use_cache=False)
    df = loader.load_all(max_workers=8, use_cache=True)
    elapsed = time.perf_counter() - start_time
    print(f"Loaded dataset with {len(df)} records in {elapsed:.2f} seconds.\n")
    print(df.head())
    print(df.columns.tolist())

"""Safe loading, profiling, and filtering of a user-supplied fire dataset."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from functools import cached_property
import hashlib
import json
import os
import pandas as pd
from app.config.settings import Config
from app.utils.column_detector import detect_columns, column_kind


def _get_config(key: str, default=None):
    try:
        from flask import current_app, has_app_context
        if has_app_context():
            return current_app.config.get(key, default)
    except Exception:
        pass
    return getattr(Config, key, default)


class DatasetUnavailable(RuntimeError):
    pass


@dataclass
class DatasetService:
    configured_path: str = ""
    # Internal LRU-style cache: maps filter_hash -> filtered DataFrame
    _filter_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _CACHE_SIZE: int = field(default=16, init=False, repr=False, compare=False)

    def _find_file(self) -> Path:
        candidates: list[Path] = []
        if self.configured_path:
            candidates.append(Path(self.configured_path))
        else:
            data_dir = Path(_get_config("DATA_DIRECTORY"))
            if data_dir.exists():
                candidates.extend(sorted(data_dir.glob("*.csv")))
                candidates.extend(sorted(data_dir.glob("*.xlsx")))
                candidates.extend(sorted(data_dir.glob("*.xls")))
            # Fallback to bundled sample dataset (ideal for fresh git clones, CI, and Cloud deployment)
            sample_dir = Path(Config.PROJECT_ROOT) / "data" / "sample"
            if sample_dir.exists():
                candidates.extend(sorted(sample_dir.glob("*.csv")))
                candidates.extend(sorted(sample_dir.glob("*.xlsx")))
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in (".csv", ".xlsx", ".xls"):
                return candidate
        raise DatasetUnavailable("No CSV or Excel dataset found. Add a CSV or Excel file to data/raw or configure DATASET_PATH.")

    @cached_property
    def path(self) -> Path:
        return self._find_file()

    @cached_property
    def raw(self) -> pd.DataFrame:
        try:
            if self.path.suffix.lower() in (".xlsx", ".xls"):
                frame = pd.read_excel(self.path)
            else:
                frame = pd.read_csv(self.path, low_memory=False)
        except (OSError, UnicodeDecodeError, pd.errors.ParserError, Exception) as error:
            raise DatasetUnavailable("The dataset could not be read as a valid CSV or Excel file.") from error
        if frame.empty:
            raise DatasetUnavailable("The dataset contains no records.")
        return frame

    @cached_property
    def columns(self) -> dict[str, str | None]:
        return detect_columns(self.raw)

    @cached_property
    def frame(self) -> pd.DataFrame:
        """Processed base DataFrame — parsed dates/numerics. Never mutate this."""
        f = self.raw.copy()
        date_col = self.columns.get("date")
        if date_col:
            f[date_col] = pd.to_datetime(f[date_col], errors="coerce")
        for key in ("latitude", "longitude", "frp", "brightness"):
            col = self.columns.get(key)
            if col:
                f[col] = pd.to_numeric(f[col], errors="coerce")
        return f

    @cached_property
    def _search_col(self) -> pd.Series | None:
        """Pre-built concatenated string column for fast text search."""
        searchable = [
            col for col in self.raw.columns
            if column_kind(self.frame[col], col, self.columns) in {"categorical", "text"}
        ]
        if not searchable:
            return None
        
        series = self.frame[searchable[0]].astype(str)
        for col in searchable[1:]:
            series = series.str.cat(self.frame[col].astype(str), sep=" ")
            
        return series.str.lower()

    @cached_property
    def _duplicate_count(self) -> int:
        """Computed once — duplicated() on large DataFrames is expensive."""
        if len(self.raw) > 200000:
            return 0  # Skip for large datasets as it hangs the server
        return int(self.raw.duplicated().sum())

    def cleaning_stats(self) -> dict:
        date_col = self.columns.get("date")
        invalid_dates = int(self.frame[date_col].isna().sum()) if date_col else 0
        return {
            "records_loaded": int(len(self.raw)),
            "date_values_invalid": invalid_dates,
            "duplicates_preserved": self._duplicate_count,
            "rows_dropped": 0,
        }

    @cached_property
    def _cached_schema(self) -> dict:
        """Schema is immutable for a given loaded dataset — compute once."""
        rows = []
        for column in self.raw.columns:
            series = self.frame[column]
            rows.append({
                "name": column,
                "type": str(series.dtype),
                "kind": column_kind(series, column, self.columns),
                "missing": int(series.isna().sum()),
                "missing_pct": round(float(series.isna().mean() * 100), 2),
                "unique": int(series.nunique(dropna=True)),
            })
        return {
            "dataset": self.path.name,
            "records": len(self.raw),
            "columns": len(self.raw.columns),
            "detected": self.columns,
            "fields": rows,
            "cleaning": self.cleaning_stats(),
        }

    def schema(self) -> dict:
        return self._cached_schema

    @staticmethod
    def _make_filter_key(params: dict) -> str:
        """Stable hash of filter params for cache keying."""
        # Sort to ensure key is order-independent
        canonical = json.dumps(params, sort_keys=True, default=str)
        return hashlib.md5(canonical.encode()).hexdigest()  # noqa: S324 – not crypto

    def filter(self, params: dict) -> pd.DataFrame:
        key = self._make_filter_key(params)
        if key in self._filter_cache:
            return self._filter_cache[key]

        # Build a boolean mask; avoid copying the full frame until we slice
        mask = pd.Series(True, index=self.frame.index)
        has_filters = False

        date_col = self.columns.get("date")
        if date_col:
            start, end = params.get("date_start"), params.get("date_end")
            if start:
                ts = pd.to_datetime(start, errors="coerce")
                if ts is not pd.NaT:
                    mask &= self.frame[date_col] >= ts
                    has_filters = True
            if end:
                ts = pd.to_datetime(end, errors="coerce")
                if ts is not pd.NaT:
                    mask &= self.frame[date_col] <= ts
                    has_filters = True

        for semantic in ("source", "confidence", "daynight", "region"):
            column, selected = self.columns.get(semantic), params.get(semantic)
            if column and selected:
                values = set(map(str, selected if isinstance(selected, list) else [selected]))
                mask &= self.frame[column].astype(str).isin(values)
                has_filters = True

        for semantic in ("frp", "brightness"):
            column = self.columns.get(semantic)
            if column:
                if params.get(f"{semantic}_min") not in (None, ""):
                    mask &= self.frame[column] >= float(params[f"{semantic}_min"])
                    has_filters = True
                if params.get(f"{semantic}_max") not in (None, ""):
                    mask &= self.frame[column] <= float(params[f"{semantic}_max"])
                    has_filters = True

        query = str(params.get("search", "")).strip().lower()
        if query and self._search_col is not None:
            mask &= self._search_col.str.contains(query, regex=False, na=False)
            has_filters = True

        result = self.frame[mask] if has_filters else self.frame

        # Evict oldest entry if cache is full
        if len(self._filter_cache) >= self._CACHE_SIZE:
            oldest = next(iter(self._filter_cache))
            del self._filter_cache[oldest]
        self._filter_cache[key] = result
        return result


_standalone_dataset_service: DatasetService | None = None


def get_dataset(path: str | None = None) -> DatasetService:
    global _standalone_dataset_service
    try:
        from flask import current_app, has_app_context
        if has_app_context():
            service = current_app.extensions.get("dataset_service")
            if service is None or (path and service.configured_path != path):
                target_path = path if path is not None else current_app.config.get("DATASET_PATH", "")
                service = DatasetService(target_path)
                current_app.extensions["dataset_service"] = service
            return service
    except Exception:
        pass

    target_path = path if path is not None else str(_get_config("DATASET_PATH", ""))
    if _standalone_dataset_service is None or _standalone_dataset_service.configured_path != target_path:
        _standalone_dataset_service = DatasetService(target_path)
    return _standalone_dataset_service


def set_active_dataset(path: str) -> DatasetService:
    """Set active dataset for both Flask and standalone/Streamlit environments."""
    global _standalone_dataset_service
    try:
        from flask import current_app, has_app_context
        if has_app_context():
            current_app.config["DATASET_PATH"] = path
            current_app.extensions.pop("dataset_service", None)
    except Exception:
        pass
    _standalone_dataset_service = DatasetService(path)
    return _standalone_dataset_service


def list_available_datasets() -> list[dict]:
    """Find all CSV/Excel files across data_dir and fallback locations."""
    data_dir = Path(_get_config("DATA_DIRECTORY"))
    patterns = ["*.csv", "*.xlsx", "*.xls"]

    current_active = ""
    try:
        ds = get_dataset()
        current_active = str(ds.path.resolve())
    except Exception:
        current_active = ""

    seen_paths = set()
    found: list[dict] = []

    # 1. Custom configured file if set
    cfg_path = _get_config("DATASET_PATH")
    if cfg_path and Path(cfg_path).is_file():
        p = Path(cfg_path).resolve()
        seen_paths.add(p)
        found.append({
            "name": p.name,
            "path": str(p),
            "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
            "active": str(p) == current_active,
            "can_delete": data_dir.resolve() in p.parents,
        })

    # 2. Files in data/raw
    if data_dir.exists():
        for pat in patterns:
            for p in sorted(data_dir.glob(pat)):
                rp = p.resolve()
                if rp not in seen_paths and p.is_file():
                    seen_paths.add(rp)
                    found.append({
                        "name": p.name,
                        "path": str(p),
                        "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                        "active": str(rp) == current_active,
                        "can_delete": True,
                    })

    # 3. Bundled sample dataset (for git clones, Streamlit Cloud, and CI runners)
    sample_dir = Path(Config.PROJECT_ROOT) / "data" / "sample"
    if sample_dir.exists():
        for pat in patterns:
            for p in sorted(sample_dir.glob(pat)):
                rp = p.resolve()
                if rp not in seen_paths and p.is_file():
                    seen_paths.add(rp)
                    found.append({
                        "name": p.name,
                        "path": str(p),
                        "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                        "active": str(rp) == current_active,
                        "can_delete": False,
                    })

    if found and not any(d["active"] for d in found) and not current_active:
        found[0]["active"] = True

    return found


"""Reusable analytics that operate on the detected dataset semantics."""
from __future__ import annotations
import hashlib
import json
import pandas as pd
import numpy as np
from app.config.settings import Config
from app.services.dataset_service import DatasetService
from app.utils.helpers import records


def _get_config(key: str, default=None):
    try:
        from flask import current_app, has_app_context
        if has_app_context():
            return current_app.config.get(key, default)
    except Exception:
        pass
    return getattr(Config, key, default)


class AnalyticsService:
    def __init__(self, dataset: DatasetService):
        self.dataset = dataset
        self._insights_cache: dict[str, list[dict]] = {}

    def _source(self, frame: pd.DataFrame):
        return self.dataset.columns.get("source")

    # ------------------------------------------------------------------
    # Summary / KPIs
    # ------------------------------------------------------------------

    def summary(self, frame: pd.DataFrame) -> dict:
        detected = self.dataset.columns
        output: dict = {"total_detections": int(len(frame)), "kpis": []}
        output["kpis"].append({
            "label": "Fire detections",
            "value": f"{len(frame):,}",
            "icon": "fa-fire-flame-curved",
            "detail": "Filtered records",
        })
        for semantic, label, unit, icon, func in [
            ("frp", "Average FRP", " MW", "fa-bolt", "mean"),
            ("frp", "Maximum FRP", " MW", "fa-arrow-trend-up", "max"),
        ]:
            col = detected.get(semantic)
            if col and frame[col].notna().any():
                value = getattr(frame[col].dropna(), func)()
                output["kpis"].append({
                    "label": label,
                    "value": f"{value:,.2f}{unit}",
                    "icon": icon,
                    "detail": "Available records",
                })
        conf = detected.get("confidence")
        if conf:
            high = self.high_confidence(frame[conf])
            output["kpis"].append({
                "label": "High confidence",
                "value": f"{int(high.sum()):,}",
                "icon": "fa-shield-halved",
                "detail": f"{high.mean() * 100:.1f}% of records",
            })
        lat, lon = detected.get("latitude"), detected.get("longitude")
        if lat and lon:
            output["kpis"].append({
                "label": "Unique locations",
                "value": f"{frame[[lat, lon]].dropna().drop_duplicates().shape[0]:,}",
                "icon": "fa-location-dot",
                "detail": "Distinct coordinate pairs",
            })
        source = self._source(frame)
        if source:
            output["kpis"].append({
                "label": "Active sources",
                "value": str(frame[source].nunique()),
                "icon": "fa-satellite",
                "detail": "Available source values",
            })
        return output

    @staticmethod
    def high_confidence(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().mean() > 0.6:
            return numeric >= 80
        return series.astype(str).str.lower().isin({"h", "high", "high confidence"})

    # ------------------------------------------------------------------
    # Time-series
    # ------------------------------------------------------------------

    def time_series(self, frame: pd.DataFrame, period: str = "day") -> dict:
        date = self.dataset.columns.get("date")
        if not date:
            return {"available": False, "message": "No date field was detected."}
        valid = frame.dropna(subset=[date])
        if valid.empty:
            return {"available": False, "message": "No valid dates are available after filtering."}
        freq = {"day": "D", "week": "W-MON", "month": "MS"}.get(period, "D")
        series = valid.set_index(date).resample(freq).size()
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekday = (
            valid[date].dt.dayofweek
            .value_counts()
            .reindex(range(7), fill_value=0)
            .rename(lambda x: days[x])
        )
        
        month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        months = (
            valid[date].dt.month
            .value_counts()
            .reindex(range(1, 13), fill_value=0)
            .rename(lambda x: month_names[x - 1])
        )
        return {
            "available": True,
            "series": [{"date": idx.strftime("%Y-%m-%d"), "count": int(v)} for idx, v in series.items()],
            "weekday": [{"label": i, "value": int(v)} for i, v in weekday.items()],
            "months": [{"label": i, "value": int(v)} for i, v in months.items()],
            "peak": {
                "date": series.idxmax().strftime("%b %d, %Y"),
                "count": int(series.max()),
                "average": round(float(series.mean()), 1),
            },
        }

    # ------------------------------------------------------------------
    # Categorical
    # ------------------------------------------------------------------

    def categorical(self, frame: pd.DataFrame, semantic: str, limit: int = 12) -> dict:
        column = self.dataset.columns.get(semantic)
        if not column:
            return {"available": False, "message": f"No {semantic} field was detected."}
        counts = frame[column].fillna("Missing").astype(str).value_counts().head(limit)
        return {
            "available": True,
            "column": column,
            "items": [{"label": str(k), "value": int(v)} for k, v in counts.items()],
        }

    # ------------------------------------------------------------------
    # Satellites & Sensor Telemetry
    # ------------------------------------------------------------------

    def satellites(self, frame: pd.DataFrame) -> dict:
        source_col = self._source(frame)
        instrument_col = next((c for c in frame.columns if c.lower() in ("instrument", "sensor")), None)
        version_col = next((c for c in frame.columns if c.lower() in ("version", "collection")), None)
        time_col = self.dataset.columns.get("time") or next((c for c in frame.columns if c.lower() in ("acq_time", "time")), None)
        daynight_col = self.dataset.columns.get("daynight")
        frp_col = self.dataset.columns.get("frp")
        bright_col = self.dataset.columns.get("brightness")
        conf_col = self.dataset.columns.get("confidence")
        scan_col = next((c for c in frame.columns if c.lower() == "scan"), None)
        track_col = next((c for c in frame.columns if c.lower() == "track"), None)

        if not source_col and not instrument_col:
            return {"available": False, "message": "No satellite, sensor, or source column detected in this dataset."}

        primary_col = source_col or instrument_col
        total_detections = len(frame)
        if total_detections == 0:
            return {"available": False, "message": "No records found after filtering."}

        raw_counts = frame[primary_col].fillna("Unknown").astype(str).value_counts()

        name_map = {
            "N": "Suomi NPP",
            "SUOMI-NPP": "Suomi NPP",
            "NPP": "Suomi NPP",
            "1": "NOAA-20",
            "J1": "NOAA-20",
            "JPSS-1": "NOAA-20",
            "2": "NOAA-21",
            "J2": "NOAA-21",
            "JPSS-2": "NOAA-21",
            "A": "Aqua (MODIS)",
            "AQUA": "Aqua (MODIS)",
            "T": "Terra (MODIS)",
            "TERRA": "Terra (MODIS)",
            "G16": "GOES-16",
            "GOES-16": "GOES-16",
            "G17": "GOES-17",
            "GOES-17": "GOES-17",
            "G18": "GOES-18",
            "GOES-18": "GOES-18",
        }

        platforms = []
        for code, count in raw_counts.items():
            norm_code = code.strip().upper()
            decoded_name = name_map.get(norm_code, code)
            sub = frame[frame[primary_col].astype(str) == code]

            p_info = {
                "code": code,
                "name": decoded_name,
                "count": int(count),
                "percentage": round(float(count / total_detections * 100), 1),
            }
            if instrument_col and instrument_col in sub.columns:
                p_info["instrument"] = str(sub[instrument_col].dropna().iloc[0]) if not sub[instrument_col].dropna().empty else "Optical/Thermal"
            else:
                p_info["instrument"] = "VIIRS (375m)" if "NPP" in decoded_name or "NOAA" in decoded_name else "Satellite Sensor"

            if frp_col and frp_col in sub.columns:
                frp_s = pd.to_numeric(sub[frp_col], errors="coerce").dropna()
                if not frp_s.empty:
                    p_info["avg_frp"] = round(float(frp_s.mean()), 1)
                    p_info["max_frp"] = round(float(frp_s.max()), 1)

            if bright_col and bright_col in sub.columns:
                b_s = pd.to_numeric(sub[bright_col], errors="coerce").dropna()
                if not b_s.empty:
                    p_info["avg_brightness"] = round(float(b_s.mean()), 1)

            platforms.append(p_info)

        items = [
            {"label": p["name"] if p["name"] != p["code"] else p["code"], "value": p["count"]}
            for p in platforms
        ]

        # 1. Hourly Overpass Distribution (00 to 23 UTC)
        hourly_distribution = []
        if time_col and time_col in frame.columns:
            time_numeric = pd.to_numeric(frame[time_col], errors="coerce").dropna()
            if not time_numeric.empty:
                hours = (time_numeric // 100).astype(int).clip(0, 23)
                hour_counts = hours.value_counts().reindex(range(24), fill_value=0)
                for h in range(24):
                    hourly_distribution.append({
                        "hour": h,
                        "label": f"{h:02d}:00",
                        "count": int(hour_counts[h]),
                    })

        # 2. Day vs Night breakdown
        daynight_stats = {"available": False}
        if daynight_col and daynight_col in frame.columns:
            dn_series = frame[daynight_col].astype(str).str.strip().str.upper()
            day_mask = dn_series.isin(["D", "DAY"])
            night_mask = dn_series.isin(["N", "NIGHT"])
            day_count = int(day_mask.sum())
            night_count = int(night_mask.sum())
            total_dn = max(day_count + night_count, 1)

            day_avg_frp = round(float(pd.to_numeric(frame.loc[day_mask, frp_col], errors="coerce").mean()), 1) if frp_col and day_count else None
            night_avg_frp = round(float(pd.to_numeric(frame.loc[night_mask, frp_col], errors="coerce").mean()), 1) if frp_col and night_count else None
            day_avg_b = round(float(pd.to_numeric(frame.loc[day_mask, bright_col], errors="coerce").mean()), 1) if bright_col and day_count else None
            night_avg_b = round(float(pd.to_numeric(frame.loc[night_mask, bright_col], errors="coerce").mean()), 1) if bright_col and night_count else None

            daynight_stats = {
                "available": True,
                "day_count": day_count,
                "night_count": night_count,
                "day_pct": round(float(day_count / total_dn * 100), 1),
                "night_pct": round(float(night_count / total_dn * 100), 1),
                "day_avg_frp": day_avg_frp,
                "night_avg_frp": night_avg_frp,
                "day_avg_brightness": day_avg_b,
                "night_avg_brightness": night_avg_b,
            }

        # 3. Confidence breakdown
        conf_stats = {"available": False}
        if conf_col and conf_col in frame.columns:
            conf_raw = frame[conf_col].dropna()
            conf_num = pd.to_numeric(conf_raw, errors="coerce")
            if conf_num.notna().mean() > 0.6:
                h_c = int((conf_num >= 80).sum())
                n_c = int(((conf_num >= 30) & (conf_num < 80)).sum())
                l_c = int((conf_num < 30).sum())
            else:
                conf_s = conf_raw.astype(str).str.lower().str.strip()
                h_c = int(conf_s.isin(["h", "high"]).sum())
                n_c = int(conf_s.isin(["n", "nominal"]).sum())
                l_c = int(conf_s.isin(["l", "low"]).sum())
            tot_conf = max(h_c + n_c + l_c, 1)
            conf_stats = {
                "available": True,
                "high": h_c,
                "nominal": n_c,
                "low": l_c,
                "high_pct": round(float(h_c / tot_conf * 100), 1),
                "nominal_pct": round(float(n_c / tot_conf * 100), 1),
                "low_pct": round(float(l_c / tot_conf * 100), 1),
            }

        # 4. Sensor viewing geometry (scan & track)
        geometry = {"available": False}
        if scan_col and track_col and scan_col in frame.columns and track_col in frame.columns:
            scan_s = pd.to_numeric(frame[scan_col], errors="coerce").dropna()
            track_s = pd.to_numeric(frame[track_col], errors="coerce").dropna()
            if not scan_s.empty and not track_s.empty:
                fp = scan_s * track_s
                geometry = {
                    "available": True,
                    "mean_scan": round(float(scan_s.mean()), 2),
                    "mean_track": round(float(track_s.mean()), 2),
                    "mean_footprint": round(float(fp.mean()), 2),
                    "min_footprint": round(float(fp.min()), 2),
                    "max_footprint": round(float(fp.max()), 2),
                    "nadir_pct": round(float((scan_s <= 0.5).mean() * 100), 1),
                    "edge_pct": round(float((scan_s >= 0.7).mean() * 100), 1),
                }

        # 5. KPIs
        primary_platform = platforms[0]["name"] if platforms else "Unknown"
        primary_sensor = platforms[0].get("instrument", "Optical/Thermal") if platforms else "Sensor"
        if version_col and version_col in frame.columns:
            v_val = str(frame[version_col].dropna().iloc[0]) if not frame[version_col].dropna().empty else "1.0"
        else:
            v_val = "NRT Collection 2"

        kpis = [
            {
                "label": "Primary Platform",
                "value": primary_platform,
                "icon": "fa-satellite",
                "detail": f"{primary_sensor}",
            },
            {
                "label": "Source Detections",
                "value": f"{total_detections:,}",
                "icon": "fa-bullseye",
                "detail": f"{len(platforms)} active platform{'s' if len(platforms) != 1 else ''}",
            },
        ]
        if daynight_stats["available"]:
            kpis.append({
                "label": "Day / Night Ratio",
                "value": f"{daynight_stats['day_pct']}% Day",
                "icon": "fa-circle-half-stroke",
                "detail": f"{daynight_stats['night_pct']}% nocturnal detections",
            })
        if geometry["available"]:
            kpis.append({
                "label": "Mean Pixel Footprint",
                "value": f"{geometry['mean_footprint']} km²",
                "icon": "fa-vector-square",
                "detail": f"{geometry['mean_scan']} km × {geometry['mean_track']} km",
            })
        if frp_col and frp_col in frame.columns:
            frp_mean = pd.to_numeric(frame[frp_col], errors="coerce").dropna().mean()
            if pd.notna(frp_mean):
                kpis.append({
                    "label": "Mean Thermal FRP",
                    "value": f"{frp_mean:.2f} MW",
                    "icon": "fa-bolt",
                    "detail": "Observed radiative power",
                })

        # 6. Dynamic Insights
        insights = []
        if len(platforms) == 1:
            insights.append(f"100% of observations collected via {primary_platform} carrying the {primary_sensor} radiometer payload.")
        else:
            insights.append(f"Multi-platform constellation active: {primary_platform} contributes {platforms[0]['percentage']}% of total detected thermal events.")

        if daynight_stats["available"]:
            if daynight_stats["day_pct"] > daynight_stats["night_pct"]:
                insights.append(f"Daytime observation dominance ({daynight_stats['day_pct']}%), capturing peak solar heating and active burning phases.")
            else:
                insights.append(f"Nighttime observation dominance ({daynight_stats['night_pct']}%), leveraging optimal infrared contrast against cooled terrain.")

        if geometry["available"]:
            insights.append(f"Sensor pixel resolution averages {geometry['mean_footprint']} km² with {geometry['nadir_pct']}% of detections situated near Nadir (optimal optical clarity).")

        return {
            "available": True,
            "column": primary_col,
            "items": items,
            "platforms": platforms,
            "kpis": kpis,
            "hourly_overpass": hourly_distribution,
            "daynight": daynight_stats,
            "confidence": conf_stats,
            "geometry": geometry,
            "insights": insights,
            "pipeline_version": v_val,
        }

    # ------------------------------------------------------------------
    # Confidence & Quality Assurance Analytics
    # ------------------------------------------------------------------

    def confidence_analysis(self, frame: pd.DataFrame) -> dict:
        col = self.dataset.columns.get("confidence")
        if not col or col not in frame.columns:
            return {"available": False, "message": "No confidence field detected in this dataset."}

        raw_series = frame[col].dropna()
        if raw_series.empty:
            return {"available": False, "message": "No confidence values available after filtering."}

        total_detections = len(raw_series)
        frp_col = self.dataset.columns.get("frp")
        bright_col = self.dataset.columns.get("brightness")
        bright_t31_col = next((c for c in frame.columns if "31" in c.lower()), None)
        dn_col = self.dataset.columns.get("daynight")

        numeric_series = pd.to_numeric(raw_series, errors="coerce")
        is_numeric = numeric_series.notna().mean() > 0.6

        class_definitions = [
            {
                "code": "h",
                "label": "High Confidence",
                "short": "High",
                "desc": "Unambiguous, intense thermal signature with optimal optical clarity",
                "color": "#10b981",
                "class_name": "conf-high",
            },
            {
                "code": "n",
                "label": "Nominal Confidence",
                "short": "Nominal",
                "desc": "Standard verified fire pixel under clear atmospheric conditions",
                "color": "#f59e0b",
                "class_name": "conf-nom",
            },
            {
                "code": "l",
                "label": "Low Confidence",
                "short": "Low",
                "desc": "Weak thermal emission, solar glint, or cloud-edge candidate",
                "color": "#64748b",
                "class_name": "conf-low",
            },
        ]

        if is_numeric:
            masks = {
                "h": numeric_series >= 80,
                "n": (numeric_series >= 30) & (numeric_series < 80),
                "l": numeric_series < 30,
            }
        else:
            str_series = raw_series.astype(str).str.strip().str.lower()
            masks = {
                "h": str_series.isin(["h", "high"]),
                "n": str_series.isin(["n", "nominal"]),
                "l": str_series.isin(["l", "low"]),
            }

        high_count = int(masks["h"].sum())
        nominal_count = int(masks["n"].sum())
        low_count = int(masks["l"].sum())
        actionable_count = high_count + nominal_count
        actionable_pct = round(float(actionable_count / total_detections * 100), 1)

        classes = []
        items = []

        for cdef in class_definitions:
            code = cdef["code"]
            mask = masks[code]
            count = int(mask.sum())
            pct = round(float(count / total_detections * 100), 1)
            sub = frame.loc[raw_series[mask].index]

            c_info = {
                "code": code,
                "label": cdef["label"],
                "short": cdef["short"],
                "desc": cdef["desc"],
                "color": cdef["color"],
                "class_name": cdef["class_name"],
                "count": count,
                "percentage": pct,
            }

            if frp_col and frp_col in sub.columns and count > 0:
                frp_s = pd.to_numeric(sub[frp_col], errors="coerce").dropna()
                if not frp_s.empty:
                    c_info["avg_frp"] = round(float(frp_s.mean()), 1)
                    c_info["max_frp"] = round(float(frp_s.max()), 1)

            if bright_col and bright_col in sub.columns and count > 0:
                b_s = pd.to_numeric(sub[bright_col], errors="coerce").dropna()
                if not b_s.empty:
                    c_info["avg_brightness"] = round(float(b_s.mean()), 1)

            if bright_t31_col and bright_t31_col in sub.columns and count > 0:
                t31_s = pd.to_numeric(sub[bright_t31_col], errors="coerce").dropna()
                if not t31_s.empty:
                    c_info["avg_bright_t31"] = round(float(t31_s.mean()), 1)

            if dn_col and dn_col in sub.columns and count > 0:
                dn_s = sub[dn_col].astype(str).str.strip().str.upper()
                d_c = int(dn_s.isin(["D", "DAY"]).sum())
                n_c = int(dn_s.isin(["N", "NIGHT"]).sum())
                tot_dn = max(d_c + n_c, 1)
                c_info["day_count"] = d_c
                c_info["night_count"] = n_c
                c_info["day_pct"] = round(float(d_c / tot_dn * 100), 1)
                c_info["night_pct"] = round(float(n_c / tot_dn * 100), 1)

            classes.append(c_info)
            items.append({"label": cdef["label"], "value": count})

        high_class = next((c for c in classes if c["code"] == "h"), {})
        nom_class = next((c for c in classes if c["code"] == "n"), {})
        low_class = next((c for c in classes if c["code"] == "l"), {})

        kpis = [
            {
                "label": "Actionable Reliability Rate",
                "value": f"{actionable_pct}%",
                "icon": "fa-shield-halved",
                "detail": "Verified High + Nominal quality",
            },
            {
                "label": "High-Confidence Firefronts",
                "value": f"{high_count:,}",
                "icon": "fa-circle-check",
                "detail": f"{high_class.get('percentage', 0)}% of dataset",
            },
            {
                "label": "Nominal Standard Quality",
                "value": f"{nominal_count:,}",
                "icon": "fa-circle-dot",
                "detail": f"{nom_class.get('percentage', 0)}% of dataset",
            },
            {
                "label": "Screened Candidates (Low)",
                "value": f"{low_count:,}",
                "icon": "fa-filter-circle-xmark",
                "detail": f"{low_class.get('percentage', 0)}% under solar/cloud noise",
            },
        ]
        if high_class.get("avg_frp"):
            kpis.append({
                "label": "High-Confidence Mean FRP",
                "value": f"{high_class['avg_frp']} MW",
                "icon": "fa-bolt-lightning",
                "detail": f"vs {nom_class.get('avg_frp', 'N/A')} MW nominal",
            })

        insights = []
        insights.append(f"{actionable_pct}% of detections meet verified quality standards (High + Nominal) suitable for operational response.")
        if low_class.get("day_pct") and low_class["day_pct"] >= 95:
            insights.append(f"Low-confidence detections are almost entirely diurnal ({low_class['day_pct']}% daytime), driven by solar glint and heated soil.")
        if high_class.get("avg_frp") and nom_class.get("avg_frp") and nom_class["avg_frp"] > 0:
            ratio = round(high_class["avg_frp"] / nom_class["avg_frp"], 1)
            insights.append(f"High-confidence fires release on average {ratio}× higher radiative power ({high_class['avg_frp']} MW) than nominal detections.")
        if high_class.get("avg_brightness") and high_class["avg_brightness"] >= 365:
            insights.append(f"High-confidence events consistently hit the sensor's maximum thermal saturation ceiling (~{high_class['avg_brightness']} K).")

        return {
            "available": True,
            "column": col,
            "items": items,
            "classes": classes,
            "kpis": kpis,
            "actionable_rate": actionable_pct,
            "insights": insights,
        }

    # ------------------------------------------------------------------
    # Intensity / FRP
    # ------------------------------------------------------------------

    def intensity(self, frame: pd.DataFrame) -> dict:
        col = self.dataset.columns.get("frp")
        if not col:
            return {"available": False, "message": "FRP was not detected in this dataset."}
        raw_vals = frame[col].dropna()
        values = pd.to_numeric(raw_vals, errors="coerce").dropna()
        if values.empty:
            return {"available": False, "message": "No valid FRP values are available after filtering."}

        total_detections = len(values)
        total_energy_gw = round(float(values.sum() / 1000), 1)
        mean_val = round(float(values.mean()), 2)
        median_val = round(float(values.median()), 2)
        max_val = round(float(values.max()), 2)
        p90_val = round(float(values.quantile(0.9)), 2)
        p99_val = round(float(values.quantile(0.99)), 2)
        extreme_count = int((values >= 100.0).sum())
        high_count = int((values >= p90_val).sum())

        # 1. Backwards compatible linear histogram
        n_bins = min(30, max(8, int(np.sqrt(min(len(values), 10000)))))
        histogram, bins = np.histogram(values, bins=n_bins)
        linear_hist = [
            {"x": round(float((bins[i] + bins[i + 1]) / 2), 2), "y": int(histogram[i])}
            for i in range(len(histogram))
        ]

        # 2. Log-scaled & well-spaced distribution bins
        log_bins = [0, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 100000]
        bin_labels = ["< 2 MW", "2–5 MW", "5–10 MW", "10–25 MW", "25–50 MW", "50–100 MW", "100–250 MW", "250–500 MW", "500–1K MW", "1K+ MW"]
        binned = pd.cut(values, bins=log_bins, labels=bin_labels, right=False)
        bin_counts = binned.value_counts(sort=False)
        log_distribution = [
            {"label": str(label), "count": int(bin_counts.get(label, 0)), "percentage": round(float(bin_counts.get(label, 0) / total_detections * 100), 1)}
            for label in bin_labels
        ]

        # 3. Wildfire Severity Classification Tiers
        tiers_def = [
            ("Low (< 10 MW)", values < 10, "Surface / smoldering fire", "tier-low"),
            ("Moderate (10–50 MW)", (values >= 10) & (values < 50), "Active flaming front", "tier-mod"),
            ("High (50–200 MW)", (values >= 50) & (values < 200), "Heavy timber torching", "tier-high"),
            ("Extreme (> 200 MW)", values >= 200, "Violent crown mega-fire", "tier-ext"),
        ]
        severity_tiers = []
        total_sum_frp = max(float(values.sum()), 0.001)
        for label, mask, desc, cls_name in tiers_def:
            t_count = int(mask.sum())
            t_frp_sum = float(values[mask].sum()) if t_count else 0.0
            severity_tiers.append({
                "label": label,
                "desc": desc,
                "class": cls_name,
                "count": t_count,
                "percentage": round(float(t_count / total_detections * 100), 1),
                "energy_share": round(float(t_frp_sum / total_sum_frp * 100), 1),
                "avg_frp": round(float(values[mask].mean()), 1) if t_count else 0.0,
            })

        # 4. Thermal Bands (brightness I4 375m & bright_t31 11um)
        bright_col = self.dataset.columns.get("brightness")
        bright_t31_col = next((c for c in frame.columns if "31" in c.lower()), None)
        thermal_bands = {"available": False}
        if bright_col and bright_col in frame.columns:
            b_vals = pd.to_numeric(frame[bright_col], errors="coerce").dropna()
            if not b_vals.empty:
                b_hist, b_edges = np.histogram(b_vals, bins=15)
                b_dist = [{"x": round(float((b_edges[i] + b_edges[i+1])/2), 1), "count": int(b_hist[i])} for i in range(len(b_hist))]

                t31_dist = []
                mean_t31 = None
                mean_delta_t = None
                if bright_t31_col and bright_t31_col in frame.columns:
                    t31_vals = pd.to_numeric(frame[bright_t31_col], errors="coerce").dropna()
                    if not t31_vals.empty:
                        t31_hist, t31_edges = np.histogram(t31_vals, bins=15)
                        t31_dist = [{"x": round(float((t31_edges[i] + t31_edges[i+1])/2), 1), "count": int(t31_hist[i])} for i in range(len(t31_hist))]
                        mean_t31 = round(float(t31_vals.mean()), 1)
                        common_idx = b_vals.index.intersection(t31_vals.index)
                        if not common_idx.empty:
                            mean_delta_t = round(float((b_vals.loc[common_idx] - t31_vals.loc[common_idx]).mean()), 1)

                thermal_bands = {
                    "available": True,
                    "mean_brightness": round(float(b_vals.mean()), 1),
                    "max_brightness": round(float(b_vals.max()), 1),
                    "mean_t31": mean_t31,
                    "mean_delta_t": mean_delta_t,
                    "brightness_distribution": b_dist,
                    "t31_distribution": t31_dist,
                }

        # 5. Top 5 Extreme Mega-Fires
        date_col = self.dataset.columns.get("date")
        lat_col = self.dataset.columns.get("latitude")
        lon_col = self.dataset.columns.get("longitude")
        time_col = self.dataset.columns.get("time") or next((c for c in frame.columns if "time" in c.lower()), None)
        conf_col = self.dataset.columns.get("confidence")
        dn_col = self.dataset.columns.get("daynight")

        top_indices = values.sort_values(ascending=False).head(5).index
        mega_fires = []
        for rank, idx in enumerate(top_indices, start=1):
            row = frame.loc[idx]
            m_lat = float(row[lat_col]) if lat_col and lat_col in row and pd.notna(row[lat_col]) else None
            m_lon = float(row[lon_col]) if lon_col and lon_col in row and pd.notna(row[lon_col]) else None
            m_date = str(row[date_col]) if date_col and date_col in row and pd.notna(row[date_col]) else "N/A"
            m_time = str(row[time_col]) if time_col and time_col in row and pd.notna(row[time_col]) else ""
            if m_time and len(m_time) <= 4:
                m_time = f"{int(m_time):04d} UTC" if m_time.isdigit() else m_time

            item = {
                "rank": rank,
                "frp": round(float(values.loc[idx]), 1),
                "date": m_date,
                "time": m_time,
                "lat": round(m_lat, 4) if m_lat is not None else None,
                "lon": round(m_lon, 4) if m_lon is not None else None,
                "location": f"{abs(m_lat):.2f}°{'N' if m_lat>=0 else 'S'}, {abs(m_lon):.2f}°{'E' if m_lon>=0 else 'W'}" if m_lat is not None and m_lon is not None else "N/A",
                "brightness": round(float(row[bright_col]), 1) if bright_col and bright_col in row and pd.notna(row[bright_col]) else None,
                "confidence": str(row[conf_col]) if conf_col and conf_col in row and pd.notna(row[conf_col]) else None,
                "daynight": "Day" if dn_col and dn_col in row and str(row[dn_col]).strip().upper() == "D" else "Night" if dn_col and dn_col in row and str(row[dn_col]).strip().upper() == "N" else None,
            }
            mega_fires.append(item)

        # 6. Top KPIs
        kpis = [
            {
                "label": "Peak Radiative Power",
                "value": f"{max_val:,.1f} MW",
                "icon": "fa-bolt-lightning",
                "detail": "Record catastrophic mega-fire",
            },
            {
                "label": "Median Fire Intensity",
                "value": f"{median_val:,.2f} MW",
                "icon": "fa-fire-flame-curved",
                "detail": "Typical flame front output",
            },
            {
                "label": "Mean Radiative Output",
                "value": f"{mean_val:,.2f} MW",
                "icon": "fa-fire",
                "detail": "Average radiative power",
            },
            {
                "label": "Cumulative Radiative Energy",
                "value": f"{total_energy_gw:,.1f} GW",
                "icon": "fa-atom",
                "detail": "Total thermal energy released",
            },
            {
                "label": "High-Energy Fire Fronts",
                "value": f"{extreme_count:,}",
                "icon": "fa-triangle-exclamation",
                "detail": "Detections exceeding 100 MW",
            },
        ]

        # 7. Insights
        insights = []
        top_tier = severity_tiers[3] if len(severity_tiers) > 3 else None
        low_tier = severity_tiers[0] if severity_tiers else None
        if low_tier:
            insights.append(f"{low_tier['percentage']}% of detections are low-intensity burns (< 10 MW), releasing only {low_tier['energy_share']}% of cumulative radiative heat.")
        if top_tier and top_tier['count'] > 0:
            insights.append(f"Mega-fires (> 200 MW) constitute just {top_tier['percentage']}% of detections but account for a disproportionate {top_tier['energy_share']}% of total atmospheric thermal release.")
        if mega_fires:
            insights.append(f"Highest single thermal burst was {mega_fires[0]['frp']:,} MW recorded at {mega_fires[0]['location']}.")

        return {
            "available": True,
            "column": col,
            "stats": {
                "mean": mean_val,
                "median": median_val,
                "maximum": max_val,
                "high_count": high_count,
                "extreme_count": extreme_count,
                "total_energy_gw": total_energy_gw,
                "p90": p90_val,
                "p99": p99_val,
            },
            "kpis": kpis,
            "histogram": linear_hist,
            "log_distribution": log_distribution,
            "severity_tiers": severity_tiers,
            "thermal_bands": thermal_bands,
            "mega_fires": mega_fires,
            "insights": insights,
        }

    # ------------------------------------------------------------------
    # Map
    # ------------------------------------------------------------------

    def map_data(self, frame: pd.DataFrame) -> dict:
        lat, lon = self.dataset.columns.get("latitude"), self.dataset.columns.get("longitude")
        if not lat or not lon:
            return {"available": False, "message": "Geographic coordinates were not detected."}
        fields = [
            f for f in [
                lat, lon,
                self.dataset.columns.get("date"),
                self.dataset.columns.get("frp"),
                self.dataset.columns.get("confidence"),
                self._source(frame),
                self.dataset.columns.get("daynight"),
            ] if f
        ]
        valid = frame.dropna(subset=[lat, lon])
        sample_limit = int(_get_config("MAP_POINT_LIMIT", 5000))
        if len(valid) > sample_limit:
            valid = valid.sample(sample_limit, random_state=42)
        return {
            "available": True,
            "points": records(valid[fields]),
            "latitude": lat,
            "longitude": lon,
            "frp": self.dataset.columns.get("frp"),
            "sampled": len(frame) > sample_limit,
        }

    # ------------------------------------------------------------------
    # Geography
    # ------------------------------------------------------------------

    def geography(self, frame: pd.DataFrame) -> dict:
        lat, lon = self.dataset.columns.get("latitude"), self.dataset.columns.get("longitude")
        region = self.dataset.columns.get("region")

        if not lat or not lon:
            if region:
                return self.categorical(frame, "region", 12)
            return {"available": False, "message": "Geographic coordinates (latitude & longitude) were not detected in this dataset."}

        # Numeric coordinate validation
        valid = frame.dropna(subset=[lat, lon]).copy()
        if valid.empty:
            return {"available": False, "message": "No valid geographic coordinates available after filtering."}

        lat_numeric = pd.to_numeric(valid[lat], errors="coerce")
        lon_numeric = pd.to_numeric(valid[lon], errors="coerce")
        valid = valid.loc[lat_numeric.notna() & lon_numeric.notna()].copy()
        valid[lat] = lat_numeric.loc[valid.index]
        valid[lon] = lon_numeric.loc[valid.index]

        if valid.empty:
            return {"available": False, "message": "Coordinate values could not be converted to numbers."}

        total_detections = len(valid)
        lat_vals = valid[lat]
        lon_vals = valid[lon]

        lat_min, lat_max = float(lat_vals.min()), float(lat_vals.max())
        lon_min, lon_max = float(lon_vals.min()), float(lon_vals.max())
        center_lat, center_lon = float(lat_vals.mean()), float(lon_vals.mean())
        lat_span = max(lat_max - lat_min, 0.01)
        lon_span = max(lon_max - lon_min, 0.01)

        # Named region categorization (if available)
        has_named_regions = False
        named_regions = []
        if region and region in valid.columns:
            reg_counts = valid[region].fillna("Unknown").astype(str).value_counts().head(10)
            if not reg_counts.empty:
                has_named_regions = True
                named_regions = [{"label": str(k), "value": int(v)} for k, v in reg_counts.items()]

        # Dynamic Grid Hotspot Detection (0.05° to 1.0° resolution based on spatial span)
        if lat_span > 25 or lon_span > 25:
            grid_step = 1.0
        elif lat_span > 6 or lon_span > 6:
            grid_step = 0.5
        elif lat_span > 1.5 or lon_span > 1.5:
            grid_step = 0.2
        else:
            grid_step = 0.05

        grid_lat = (valid[lat] / grid_step).round() * grid_step
        grid_lon = (valid[lon] / grid_step).round() * grid_step

        frp_col = self.dataset.columns.get("frp")
        brightness_col = self.dataset.columns.get("brightness")

        cluster_counts = pd.Series(list(zip(grid_lat, grid_lon))).value_counts().head(8)
        max_cluster_count = int(cluster_counts.iloc[0]) if not cluster_counts.empty else 1

        clusters = []
        for (c_lat, c_lon), count in cluster_counts.items():
            mask = (grid_lat == c_lat) & (grid_lon == c_lon)
            sub = valid.loc[mask]

            c_lat_f = float(c_lat)
            c_lon_f = float(c_lon)
            lat_str = f"{abs(c_lat_f):.2f}°{'N' if c_lat_f >= 0 else 'S'}"
            lon_str = f"{abs(c_lon_f):.2f}°{'E' if c_lon_f >= 0 else 'W'}"
            label = f"{lat_str}, {lon_str}"

            item = {
                "label": label,
                "lat": round(c_lat_f, 4),
                "lon": round(c_lon_f, 4),
                "count": int(count),
                "percentage": round(float((count / total_detections) * 100), 1),
                "relative_ratio": round(float(count / max_cluster_count), 3),
            }

            if frp_col and frp_col in sub.columns:
                frp_s = pd.to_numeric(sub[frp_col], errors="coerce").dropna()
                if not frp_s.empty:
                    item["avg_frp"] = round(float(frp_s.mean()), 1)
                    item["max_frp"] = round(float(frp_s.max()), 1)

            if brightness_col and brightness_col in sub.columns:
                b_s = pd.to_numeric(sub[brightness_col], errors="coerce").dropna()
                if not b_s.empty:
                    item["avg_brightness"] = round(float(b_s.mean()), 1)

            clusters.append(item)

        # Coordinate distribution histograms (for Plotly distribution curves)
        n_bins = min(20, max(8, int(np.sqrt(min(len(valid), 10000)))))
        lat_hist, lat_edges = np.histogram(lat_vals, bins=n_bins)
        lon_hist, lon_edges = np.histogram(lon_vals, bins=n_bins)

        lat_dist = [
            {"x": round(float((lat_edges[i] + lat_edges[i+1]) / 2), 2), "count": int(lat_hist[i])}
            for i in range(len(lat_hist))
        ]
        lon_dist = [
            {"x": round(float((lon_edges[i] + lon_edges[i+1]) / 2), 2), "count": int(lon_hist[i])}
            for i in range(len(lon_hist))
        ]

        # Hemispheric and Quadrants stats
        north_count = int((lat_vals >= 0).sum())
        south_count = int((lat_vals < 0).sum())
        east_count = int((lon_vals >= 0).sum())
        west_count = int((lon_vals < 0).sum())

        # Top hotspot cluster
        top_cluster = clusters[0] if clusters else None
        top_clusters_share = round(sum(c["percentage"] for c in clusters[:3]), 1) if clusters else 0

        # Sample points for mini map preview (up to 150 points)
        sample_limit = 150
        pts_sample = valid if len(valid) <= sample_limit else valid.sample(sample_limit, random_state=42)
        sample_points = []
        for _, row in pts_sample.iterrows():
            pt = {"lat": round(float(row[lat]), 4), "lon": round(float(row[lon]), 4)}
            if frp_col and frp_col in row and pd.notna(row[frp_col]):
                try:
                    pt["frp"] = round(float(row[frp_col]), 1)
                except (ValueError, TypeError):
                    pass
            sample_points.append(pt)

        # KPIs
        center_lat_str = f"{abs(center_lat):.2f}°{'N' if center_lat >= 0 else 'S'}"
        center_lon_str = f"{abs(center_lon):.2f}°{'E' if center_lon >= 0 else 'W'}"
        kpis = [
            {
                "label": "Spatial Centroid",
                "value": f"{center_lat_str}, {center_lon_str}",
                "icon": "fa-bullseye",
                "detail": f"Center of {total_detections:,} fires",
            },
            {
                "label": "Peak Fire Hotspot",
                "value": top_cluster["label"] if top_cluster else "N/A",
                "icon": "fa-fire-flame-curved",
                "detail": f"{top_cluster['count']:,} fires ({top_cluster['percentage']}%)" if top_cluster else "No cluster detected",
            },
            {
                "label": "Latitude Span",
                "value": f"{lat_span:.2f}°",
                "icon": "fa-arrows-up-down",
                "detail": f"{lat_min:.1f}° to {lat_max:.1f}°",
            },
            {
                "label": "Longitude Span",
                "value": f"{lon_span:.2f}°",
                "icon": "fa-arrows-left-right",
                "detail": f"{lon_min:.1f}° to {lon_max:.1f}°",
            },
            {
                "label": "Top Hotspots Share",
                "value": f"{top_clusters_share}%",
                "icon": "fa-chart-pie",
                "detail": "Concentration in top 3 clusters",
            },
        ]

        # Dynamic Insights
        insights = []
        if top_cluster:
            insights.append(f"Highest thermal cluster concentrated at {top_cluster['label']} containing {top_cluster['count']:,} fire events ({top_cluster['percentage']}% of filtered dataset).")
        if top_clusters_share >= 40:
            insights.append(f"Strong geographic concentration: {top_clusters_share}% of detections occur inside just 3 primary hotspot zones.")
        if north_count and not south_count:
            insights.append("100% of fire activity is located in the Northern Hemisphere.")
        elif south_count and not north_count:
            insights.append("100% of fire activity is located in the Southern Hemisphere.")
        elif north_count and south_count:
            hemi_name = "Northern" if north_count >= south_count else "Southern"
            hemi_pct = round(max(north_count, south_count) / total_detections * 100, 1)
            insights.append(f"{hemi_pct}% of fire detections occurred in the {hemi_name} Hemisphere.")
        if east_count and not west_count:
            insights.append("All fire detections are distributed across the Eastern Hemisphere.")
        elif west_count and not east_count:
            insights.append("All fire detections are distributed across the Western Hemisphere.")

        # Default items for any consumers expecting .items
        items = named_regions if has_named_regions else [{"label": c["label"], "value": c["count"]} for c in clusters]

        return {
            "available": True,
            "has_named_regions": has_named_regions,
            "items": items,
            "clusters": clusters,
            "kpis": kpis,
            "bounds": {
                "lat_min": lat_min, "lat_max": lat_max,
                "lon_min": lon_min, "lon_max": lon_max,
                "center_lat": center_lat, "center_lon": center_lon,
            },
            "lat_distribution": lat_dist,
            "lon_distribution": lon_dist,
            "quadrants": {
                "north": north_count,
                "south": south_count,
                "east": east_count,
                "west": west_count,
                "north_pct": round(north_count / total_detections * 100, 1),
                "south_pct": round(south_count / total_detections * 100, 1),
                "east_pct": round(east_count / total_detections * 100, 1),
                "west_pct": round(west_count / total_detections * 100, 1),
            },
            "sample_points": sample_points,
            "insights": insights,
        }

    # ------------------------------------------------------------------
    # Data quality — reads only raw/constant stats; cached on instance
    # ------------------------------------------------------------------

    def data_quality(self, frame: pd.DataFrame) -> dict:
        raw = self.dataset.raw
        return {
            "records": int(len(raw)),
            "columns": int(len(raw.columns)),
            "duplicates": self.dataset._duplicate_count,
            "memory_mb": round(float(raw.memory_usage(deep=True).sum() / 1048576), 2),
            "missing_total": int(raw.isna().sum().sum()),
            "filtered_records": int(len(frame)),
            "cleaning": self.dataset.cleaning_stats(),
        }

    # ------------------------------------------------------------------
    # Insights — cached per filter hash
    # ------------------------------------------------------------------

    @staticmethod
    def _frame_hash(frame: pd.DataFrame) -> str:
        return hashlib.md5(str(frame.index.tolist()).encode()).hexdigest()  # noqa: S324

    def insights(self, frame: pd.DataFrame) -> list[dict]:
        cache_key = self._frame_hash(frame)
        if cache_key in self._insights_cache:
            return self._insights_cache[cache_key]

        result: list[dict] = []
        temporal = self.time_series(frame, "month")
        if temporal.get("available"):
            result.append({
                "icon": "fa-calendar-days",
                "title": "Peak activity",
                "text": (
                    f"The dataset indicates the highest activity occurred around "
                    f"{temporal['peak']['date']}, with {temporal['peak']['count']:,} detections."
                ),
            })
        source = self.categorical(frame, "source", 1)
        if source.get("available") and source["items"]:
            result.append({
                "icon": "fa-satellite",
                "title": "Leading source",
                "text": (
                    f"Among available records, {source['items'][0]['label']} contributed "
                    f"the most detections ({source['items'][0]['value']:,})."
                ),
            })
        intensity = self.intensity(frame)
        if intensity.get("available"):
            result.append({
                "icon": "fa-bolt",
                "title": "Fire radiative power",
                "text": (
                    f"Average FRP is {intensity['stats']['mean']:,.2f} MW; "
                    f"the highest recorded value is {intensity['stats']['maximum']:,.2f} MW."
                ),
            })
        confidence = self.categorical(frame, "confidence", 1)
        if confidence.get("available") and confidence["items"]:
            result.append({
                "icon": "fa-shield-halved",
                "title": "Confidence pattern",
                "text": (
                    f"The most common confidence value is {confidence['items'][0]['label']} "
                    f"({confidence['items'][0]['value']:,} detections)."
                ),
            })

        self._insights_cache[cache_key] = result
        return result

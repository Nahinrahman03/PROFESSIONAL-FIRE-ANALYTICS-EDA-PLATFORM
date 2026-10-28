# FireScope — Professional Fire Analytics & Geospatial Intelligence Platform

**FireScope** is a high-performance, responsive **Streamlit** command-center platform for exploring active-fire detection data and satellite thermal anomaly telemetry. It converts compatible CSV or Excel files (such as NASA FIRMS VIIRS/MODIS exports) into an interactive geospatial intelligence environment featuring real-time data quality profiling, multi-layer heatmaps, diurnal burning dynamics, machine learning forecasting, and automated executive intelligence briefings.

> *Note: This is an independent analytics project. It is not affiliated with, endorsed by, or branded as NASA. Detections originate from FIRMS-derived satellite observation exports.*

---

## 🌟 Key Features & Capabilities

1. **Executive Command Center**:
   - Real-time Telemetry Ribbon displaying dataset coverage, spatial bounding coordinates, and system health.
   - Dual-axis temporal timeline: fire detection counts (area fill) overlaid with Mean Fire Radiative Power (MW).
   - Multi-dimensional Sunburst hierarchy breakdown: `Satellite Platform` → `Solar Elevation (Day/Night)` → `Confidence Tier`.
   - Glassmorphic KPI cards with dynamic category styling and active alert notification banners.

2. **Geospatial Map Explorer**:
   - **Thermal Density Heatmap**: High-density thermal intensity rendering with custom blur radius and color scales (`Inferno`, `Plasma`, `Turbo`, `Hot`).
   - **Pinpoint Hotspot Markers**: Sub-kilometer coordinate plotting with FRP-scaled sizing and hover telemetry.
   - **K-Means Spatial Cluster Centers**: Automated hotspot detection highlighting centroids of persistent cluster formations.
   - Multiple basemap styles: Carto Darkmatter, OpenStreetMap, and Carto Positron.

3. **Temporal Dynamics & Diurnal Cycles**:
   - Grouped activity timelines (Daily, Weekly, Monthly) with 7-day moving averages.
   - **24-Hour Diurnal Fire Cycle (UTC)**: Hour-by-hour satellite overpass breakdown to pinpoint peak burning windows.
   - **7×24 Temporal Fire Activity Heatmap Matrix**: Day-of-week vs. UTC hour matrix identifying diurnal ignition peaks.

4. **Fire Intensity & Radiative Power (FRP)**:
   - FRP distribution histograms with logarithmic and linear toggles.
   - **Marginal Scatter Plot**: Brightness temperature vs. FRP with marginal top histogram, right boxplot, and OLS regression trendline.
   - **Day vs. Night FRP Violin Distribution**: Comparative analysis of daytime flame fronts versus nocturnal smoldering.
   - **Pareto Energy Concentration Curve**: Visualizes cumulative energy concentration (e.g., top 10% of detections releasing >75% of total radiative energy).
   - Top Catastrophic Mega-Fire incident table.

5. **Confidence & Sensor Telemetry**:
   - **Actionable Reliability Index Gauge**: Gauge meter with color-coded operational reliability zones.
   - Cross-sensor comparison (VIIRS 375m vs. MODIS 1km, Suomi NPP, NOAA-20, NOAA-21, Aqua, Terra).
   - Platform performance and radiometer channel specification benchmarks.

6. **Geographic Insights**:
   - Latitudinal and longitudinal spatial frequency histograms with kernel density projections.
   - Coordinate bounding box metrics and spatial density calculations.
   - Spatial cluster centroid rankings.

7. **Data Overview & Quality Audit**:
   - Automated dataset profiling: row/column counts, memory footprint, duplicate detection, and missingness audit.
   - **Cross-Channel Correlation Matrix**: Pearson correlation heatmap across all numeric telemetry channels (`brightness`, `bright_t31`, `scan`, `track`, `frp`).

8. **Export & GIS Integration**:
   - One-click CSV filtered dataset download.
   - **GeoJSON GIS Spatial Layer (`.geojson`)**: Formatted point features for instant loading into QGIS, ArcGIS, Mapbox, or Google Earth.
   - **Executive Fire Intelligence Briefing (`.md`)**: Situation assessment briefing with active threat status and top fire rankings.

9. **Machine Learning Engine**:
   - **Random Forest Regressor**: Predict fire radiative power (FRP) with MAE, RMSE, R² scores, and top feature importance rankings.
   - **Random Forest Classifier**: Classify detection confidence tiers with accuracy scoring and feature importances.
   - **K-Means Spatial Clustering**: Detect hotspot clusters with silhouette evaluation and centroid coordinates.
   - **Model Persistence**: Export and download trained models as `.joblib`/`.pkl` binaries.
   - **Real-Time Firefront Intensity & Danger Simulator**: Interactive What-If simulation calculating estimated FRP (MW), Fire Danger Index (FDI), and tactical firefighting containment advisories.

10. **Alert Rules**:
    - Configure custom threshold conditions (FRP, confidence, satellite platform, solar elevation).
    - Real-time rule evaluation engine with immediate alert status cards.

---

## 🏛️ Architecture

```text
app/
├── config/              Runtime and environment configuration (settings.py)
├── ml/                  Feature engineering and scikit-learn model trainers
├── services/            Dataset caching, analytical aggregations, and alert engine
├── utils/               Dynamic schema detector and formatting helpers
└── streamlit_app.py     Streamlit command-center application
data/
├── raw/                 Local CSV/Excel datasets (.gitkeep tracked, large files ignored)
├── models/              Trained model directory (.gitkeep tracked)
└── processed/           Processed data cache (.gitkeep tracked)
tests/                   Comprehensive pytest suite
streamlit_app.py         Root Streamlit entrypoint
run.py                   Cross-platform launcher with UTF-8 support
run.bat / run.ps1        Windows one-click launchers
requirements.txt         Python dependency declarations
Dockerfile               Production Docker container specification
docker-compose.yml       Container orchestration configuration
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or newer (Python 3.11 – 3.14 compatible)
- Git

### Installation

```bash
# 1. Clone the repository
git clone <your-repository-url>
cd "PROFESSIONAL FIRE ANALYTICS EDA PLATFORM"

# 2. Set up virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# macOS / Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Environment configuration (optional)
cp .env.example .env
```

### Launching the Dashboard

You can start FireScope with any of the following commands:

```bash
# Option A: Unified Python launcher (auto-activates venv & handles UTF-8)
python run.py

# Option B: Direct Streamlit CLI
streamlit run app/streamlit_app.py

# Option C: Windows one-click script
.\run.ps1
# or
run.bat
```

Open your browser at **`http://localhost:8501`**.

---

## 🧪 Testing

Run the automated test suite with `pytest`:

```bash
pytest
```

---

## 📄 License

This project is open-source and available under the terms of the [MIT License](LICENSE).

/**
 * FireScope — Dashboard UI
 *
 * Key performance decisions:
 *  - Overview uses /api/overview (1 request instead of 7).
 *  - Client-side TTL cache: repeated view switches within 30s are instant.
 *  - Loading skeleton replaces blank content during fetch.
 *  - Plotly charts are rendered inside requestAnimationFrame to avoid jank.
 */

/* ─── State ──────────────────────────────────────────────────────────────── */

const state = {
  view: 'overview',
  filters: {},
  options: {},
  schema: null,
  map: null,
  miniMap: null,
  layer: null,
};

/* ─── Client-side TTL cache (30 s) ──────────────────────────────────────── */

const _cache = new Map(); // key -> { data, ts }
const CACHE_TTL = 30_000; // milliseconds

function cacheGet(key) {
  const entry = _cache.get(key);
  if (!entry) return null;
  if (Date.now() - entry.ts > CACHE_TTL) { _cache.delete(key); return null; }
  return entry.data;
}

function cacheSet(key, data) {
  _cache.set(key, { data, ts: Date.now() });
}

function cacheClear() { _cache.clear(); }

function cacheKey(path, body) {
  return path + '|' + JSON.stringify(body ?? {});
}

/* ─── Utilities ──────────────────────────────────────────────────────────── */

const $ = s => document.querySelector(s);
const esc = v => String(v ?? '').replace(/[&<>"']/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[c]));

/** Fetch JSON from /api/<path>, caching results for CACHE_TTL ms. */
async function apiCached(path, body) {
  const key = cacheKey(path, body);
  const cached = cacheGet(key);
  if (cached) return cached;

  const options = body
    ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
    : {};
  const r = await fetch('/api/' + path, options);
  const type = r.headers.get('content-type') || '';
  if (!r.ok) throw new Error((type.includes('json') ? (await r.json()).error : await r.text()) || 'Request failed');
  if (!type.includes('json')) return r; // binary response (download) — no cache
  const data = await r.json();
  cacheSet(key, data);
  return data;
}

function filters(extra = {}) { return { ...state.filters, ...extra }; }

/* ─── DOM helpers ────────────────────────────────────────────────────────── */

function card(title, body, cls = '', action = '') {
  return `<article class="card animate-in ${cls}"><div class="card-header"><h2>${esc(title)}</h2>${action ? action : ''}</div>${body}</article>`;
}

function empty(message) {
  return `<div class="empty">${esc(message)}</div>`;
}

function skeleton(rows = 3) {
  return `<div class="skeleton-wrap">${Array.from({ length: rows }, () =>
    '<div class="skeleton-row"><div class="skeleton-bar wide"></div><div class="skeleton-bar medium"></div></div>'
  ).join('')}</div>`;
}

function loadingContent() {
  return `
    <section class="grid kpis">
      ${Array.from({ length: 5 }, () => `
        <article class="card kpi skeleton-card">
          <span class="kpi-icon skeleton-icon"></span>
          <div><div class="skeleton-bar medium"></div><div class="skeleton-bar wide"></div></div>
        </article>`).join('')}
    </section>
    <section class="grid overview-layout">${skeleton(4)}</section>`;
}

function chartCard(title, id, cls = '') {
  return card(title, `<div class="chart" id="${id}"></div>`, cls);
}

function kpis(items) {
  return `<section class="grid kpis">${items.map((x, i) => `
    <article class="card kpi animate-in delay-${(i % 4) + 1}">
      <span class="kpi-icon"><i class="fa-solid ${x.icon}"></i></span>
      <div class="kpi-content">
        <div class="kpi-label">${esc(x.label)}</div>
        <div class="kpi-value">${esc(x.value)}</div>
        <div class="kpi-detail">
          <i class="fa-solid fa-circle-check" style="color:var(--cyan);font-size:10px"></i>
          <span>${esc(x.detail || 'Verified metric')}</span>
        </div>
      </div>
    </article>`).join('')}</section>`;
}

/* ─── Chart helpers ──────────────────────────────────────────────────────── */

const PLOT_DEFAULTS = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor:  'rgba(0,0,0,0)',
  font: { color: '#64748b', family: 'Inter, sans-serif' },
  margin: { l: 48, r: 16, t: 20, b: 80 },
  xaxis: { gridcolor: '#f1f5f9', zerolinecolor: '#f1f5f9', automargin: true },
  yaxis: { gridcolor: '#f1f5f9', zerolinecolor: '#f1f5f9', automargin: true },
};

function plot(id, data, layout = {}) {
  const mergedLayout = {
    ...PLOT_DEFAULTS,
    ...layout,
    xaxis: { ...PLOT_DEFAULTS.xaxis, ...(layout.xaxis || {}) },
    yaxis: { ...PLOT_DEFAULTS.yaxis, ...(layout.yaxis || {}) },
    margin: { ...PLOT_DEFAULTS.margin, ...(layout.margin || {}) }
  };
  requestAnimationFrame(() =>
    Plotly.newPlot(id, data, mergedLayout, { responsive: true, displayModeBar: false })
  );
}

function bar(id, items, color) {
  plot(id, [{ x: items.map(i => i.label), y: items.map(i => i.value), type: 'bar', marker: { color } }],
    { xaxis: { tickangle: -30, gridcolor: '#f1f5f9' }, yaxis: { gridcolor: '#f1f5f9' } });
}

function simpleDonut(id, result, colors) {
  if (!result.available) { $('#' + id).innerHTML = empty(result.message); return; }
  plot(id, [{
    values: result.items.map(x => x.value),
    labels: result.items.map(x => x.label),
    type: 'pie', hole: 0.65,
    marker: { colors: colors || ['#ef4444', '#f59e0b', '#fcd34d', '#3b82f6', '#8b5cf6', '#10b981', '#94a3b8'] },
    textinfo: 'none', hoverinfo: 'label+percent',
  }], { showlegend: true, legend: { orientation: 'v', x: 1, y: 0.5 }, margin: { l: 0, r: 0, t: 10, b: 10 } });
}

/* ─── Map ────────────────────────────────────────────────────────────────── */

function renderMap(data) {
  if (state.map) state.map.remove();
  state.map = L.map('fire-map', { zoomControl: true }).setView([20, 0], 2);
  const dark  = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    { attribution: '© OpenStreetMap © CARTO' }).addTo(state.map);
  const light = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    { attribution: '© OpenStreetMap' });
  L.control.layers({ Dark: dark, Light: light }).addTo(state.map);
  const group = L.markerClusterGroup({ disableClusteringAtZoom: 7, maxClusterRadius: 45 });
  const bounds = [];
  for (const p of data.points) {
    const lat = Number(p[data.latitude]), lon = Number(p[data.longitude]);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;
    const power = data.frp ? Number(p[data.frp]) : 0;
    const color = power > 100 ? '#ef4444' : power > 30 ? '#ff8429' : '#ffbd22';
    const popup = Object.entries(p).map(([k, v]) => `<b>${esc(k)}</b>: ${esc(v)} `).join('<br>');
    group.addLayer(L.circleMarker([lat, lon], { radius: 5, fillColor: color, color, weight: 0, fillOpacity: 0.8 }).bindPopup(popup));
    bounds.push([lat, lon]);
  }
  group.addTo(state.map);
  if (bounds.length) state.map.fitBounds(bounds, { padding: [20, 20], maxZoom: 5 });
}

/* ─── Views ──────────────────────────────────────────────────────────────── */

async function overview() {
  // Single combined API call instead of 7 parallel calls
  const d = await apiCached('overview', filters());
  const { summary, time_series: time, frp, confidence, satellites: sources, insights, map } = d;

  let html = kpis(summary.kpis);
  html += `<section class="grid overview-layout">
    ${card('Global Fire Activity', map.available ? '<div id="fire-map" class="map"></div>' : empty(map.message), 'map-card')}
  </section>`;
  html += `<section class="grid secondary-grid">
    ${chartCard('Fire Detections Over Time', 'time-chart')}
    ${chartCard('Fire Radiative Power (FRP) Distribution', 'frp-chart')}
  </section>`;
  html += `<section class="grid secondary-grid">
    ${chartCard('Detections by Confidence', 'confidence-chart')}
    ${chartCard('Detections by Satellite Source', 'sources-chart')}
  </section>`;
  
  // Real evaluated alerts from dataset & alert rules
  let alertsHtml = '';
  if (d.alerts && d.alerts.length > 0) {
    alertsHtml = d.alerts.map(a => `
      <div class="insight">
        <i class="fa-solid fa-triangle-exclamation" style="color:var(--red);font-size:12px;margin-top:2px"></i>
        <div>
          <strong>${esc(a.name)}</strong>
          <span>${(a.matched_count || 0).toLocaleString()} fire detections matched alert conditions</span>
        </div>
      </div>`).join('');
  } else {
    const totalDetections = summary?.total_detections ?? 0;
    const maxFrp = summary?.kpis?.find(k => k.label.includes('Maximum FRP'));
    alertsHtml = `
      <div class="insight">
        <i class="fa-solid fa-circle-check" style="color:var(--green);font-size:12px;margin-top:2px"></i>
        <div>
          <strong>Real-Time Monitoring Operational</strong>
          <span>${totalDetections.toLocaleString()} detections actively scanned. No custom alert conditions triggered.</span>
        </div>
      </div>
      ${maxFrp ? `
      <div class="insight">
        <i class="fa-solid fa-bolt" style="color:var(--orange);font-size:12px;margin-top:2px"></i>
        <div>
          <strong>Peak FRP Observed: ${esc(maxFrp.value)}</strong>
          <span>Highest thermal radiative power detection in active dataset</span>
        </div>
      </div>` : ''}
    `;
  }

  html += `<section class="grid secondary-grid">
    ${card('Recent Active Alerts', `<div class="insights">${alertsHtml}</div>`, '', '<a href="#" id="view-alerts-link" style="font-size:12px;color:var(--cyan);font-weight:600;text-decoration:none">Manage Rules <i class="fa-solid fa-arrow-right"></i></a>')}
    ${card('Key Insights', `<div class="insights">${
      insights.items.map(i => `
        <div class="insight">
          <i class="fa-solid fa-circle" style="color:var(--orange);font-size:6px"></i>
          <div><p>${esc(i.text)}</p></div>
        </div>`).join('') || empty('No insights available.')
    }</div>`)}
  </section>`;

  $('#content').innerHTML = html;

  const viewAlertsLink = $('#view-alerts-link');
  if (viewAlertsLink) {
    viewAlertsLink.addEventListener('click', e => {
      e.preventDefault();
      const navItem = document.querySelector('[data-view="alerts"]');
      if (navItem) navItem.click();
    });
  }

  if (time.available) {
    plot('time-chart', [{
      x: time.series.map(x => x.date),
      y: time.series.map(x => x.count),
      type: 'scatter', mode: 'lines',
      line: { color: '#ef4444', width: 2 },
      fill: 'tozeroy', fillcolor: 'rgba(239,68,68,0.1)',
    }]);
  }
  if (frp.available) {
    plot('frp-chart', [{
      x: frp.histogram.map(x => x.x),
      y: frp.histogram.map(x => x.y),
      type: 'bar', marker: { color: '#f59e0b' },
    }], { xaxis: { title: 'FRP (MW)' }, yaxis: { title: 'Count' } });
  }
  simpleDonut('confidence-chart', confidence, ['#ef4444', '#f59e0b', '#fcd34d']);
  simpleDonut('sources-chart', sources, ['#3b82f6', '#8b5cf6', '#10b981', '#14b8a6', '#94a3b8']);
  if (map.available) renderMap(map);
}

async function temporal() {
  const d = await apiCached('time-series', filters());
  $('#content').innerHTML = `
    <div class="page-section-header">
      <div>
        <p class="page-eyebrow"><i class="fa-regular fa-clock"></i> Temporal Analysis</p>
        <h2 class="page-heading">Fire Activity Over Time</h2>
        <p class="page-subtext">Track how fire detection frequency changes across time periods, months, and days of the week.</p>
      </div>
    </div>
    <section class="grid page-grid">
      ${chartCard('Detection activity over time', 'temporal-line', 'full')}
      ${chartCard('Detections by month', 'month-chart')}
      ${chartCard('Detections by day of week', 'weekday-chart')}
      ${d.available ? card('Peak activity', `
        <div class="stat-row"><span>Highest activity date</span><strong>${esc(d.peak.date)}</strong></div>
        <div class="stat-row"><span>Detections at peak</span><strong>${d.peak.count.toLocaleString()}</strong></div>
        <div class="stat-row"><span>Average per period</span><strong>${d.peak.average}</strong></div>
      `) : ''}
    </section>`;
  if (!d.available) { $('#temporal-line').innerHTML = empty(d.message); return; }
  plot('temporal-line', [{
    x: d.series.map(x => x.date), y: d.series.map(x => x.count),
    type: 'scatter', mode: 'lines+markers',
    line: { color: '#ff8429', width: 2 }, marker: { size: 5, color: '#ff8429' },
    fill: 'tozeroy', fillcolor: 'rgba(255,132,41,0.08)',
  }]);
  bar('month-chart', d.months, '#9b74e8');
  bar('weekday-chart', d.weekday, '#22c7dc');
}

async function intensity() {
  const d = await apiCached('frp', filters());

  if (!d.available) {
    $('#content').innerHTML = `
      <div class="page-section-header">
        <div>
          <p class="page-eyebrow"><i class="fa-solid fa-bolt"></i> Fire Intensity</p>
          <h2 class="page-heading">Radiative Power Distribution</h2>
          <p class="page-subtext">Explore how fire radiative power (FRP) is distributed across all detected fire events.</p>
        </div>
      </div>
      <section class="grid page-grid">
        ${card('Fire Intensity', empty(d.message || 'No FRP telemetry available.'))}
      </section>`;
    return;
  }

  // 1. Severity Tiers HTML
  let severityHtml = '';
  if (d.severity_tiers && d.severity_tiers.length) {
    severityHtml = `
      <div class="severity-list">
        ${d.severity_tiers.map(t => `
          <div class="severity-item ${t.class}">
            <div class="severity-top">
              <div class="severity-title-group">
                <span class="severity-pill">${esc(t.label.split(' ')[0])}</span>
                <span class="severity-name">${esc(t.label)}</span>
              </div>
              <div class="severity-count">${t.count.toLocaleString()} <span>(${t.percentage}%)</span></div>
            </div>
            <div class="severity-bars">
              <div class="severity-bar-track">
                <div class="severity-bar-fill" style="width:${Math.max(2, t.percentage)}%"></div>
              </div>
            </div>
            <div class="severity-meta">
              <span>${esc(t.desc)}</span>
              <span>Energy Share: <strong>${t.energy_share}%</strong> · Avg: <strong>${t.avg_frp} MW</strong></span>
            </div>
          </div>
        `).join('')}
      </div>`;
  } else {
    severityHtml = empty('Severity classification not available.');
  }

  // 2. Top Extreme Mega-Fires HTML
  let megaFiresHtml = '';
  if (d.mega_fires && d.mega_fires.length) {
    megaFiresHtml = `
      <div class="mega-fires-list">
        ${d.mega_fires.map(m => `
          <div class="mega-fire-row">
            <div class="mega-fire-left">
              <div class="mega-fire-rank">#${m.rank}</div>
              <div>
                <div class="mega-fire-location">${esc(m.location)}</div>
                <div class="mega-fire-date">${esc(m.date)} ${m.time ? `· ${esc(m.time)}` : ''} ${m.daynight ? `· ${esc(m.daynight)}` : ''}</div>
              </div>
            </div>
            <div class="mega-fire-right">
              <div class="mega-fire-frp"><i class="fa-solid fa-bolt-lightning"></i> ${m.frp.toLocaleString()} MW</div>
              <div class="mega-fire-sub">${m.brightness ? `Brightness: ${m.brightness} K` : 'Peak Firefront'}</div>
            </div>
          </div>
        `).join('')}
      </div>`;
  } else {
    megaFiresHtml = empty('No extreme fire outbreaks recorded.');
  }

  // 3. Dynamic Energy Insights HTML
  const insightsHtml = (d.insights && d.insights.length)
    ? `<div class="insights">
        ${d.insights.map(txt => `
          <div class="insight">
            <i class="fa-solid fa-bolt" style="color:var(--orange);font-size:12px;margin-top:2px"></i>
            <div><p>${esc(txt)}</p></div>
          </div>
        `).join('')}
      </div>`
    : empty('No radiative insights available.');

  // 4. Statistical Summary Breakdown
  const statsHtml = d.stats
    ? `<div class="stat-row"><span>Record Maximum FRP</span><strong>${d.stats.maximum.toLocaleString()} MW</strong></div>
       <div class="stat-row"><span>99th Percentile FRP</span><strong>${(d.stats.p99 || 0).toLocaleString()} MW</strong></div>
       <div class="stat-row"><span>90th Percentile FRP</span><strong>${(d.stats.p90 || 0).toLocaleString()} MW</strong></div>
       <div class="stat-row"><span>Mean Radiative Output</span><strong>${d.stats.mean.toLocaleString()} MW</strong></div>
       <div class="stat-row"><span>Median Typical Flame</span><strong>${d.stats.median.toLocaleString()} MW</strong></div>
       <div class="stat-row"><span>High-Energy Fires (≥100 MW)</span><strong style="color:var(--red)">${(d.stats.extreme_count || 0).toLocaleString()} fires</strong></div>
       <div class="stat-row"><span>Cumulative Thermal Power</span><strong style="color:var(--orange)">${(d.stats.total_energy_gw || 0).toLocaleString()} GW</strong></div>`
    : empty('Stats not computed.');

  let html = `
    <div class="page-section-header">
      <div>
        <p class="page-eyebrow"><i class="fa-solid fa-bolt"></i> Fire Intensity & Radiative Power</p>
        <h2 class="page-heading">Thermal Energy & Radiative Distribution</h2>
        <p class="page-subtext">Quantify the radiative heat energy (FRP) and thermal emission profiles across all detected wildfire events.</p>
      </div>
    </div>`;

  if (d.kpis && d.kpis.length) {
    html += kpis(d.kpis);
  }

  html += `
    <section class="grid page-grid">
      ${chartCard('Wildfire Intensity Distribution (Log-Binned Scale)', 'intensity-chart')}
      ${card('Wildfire Severity Classification Tiers', severityHtml, '', '<span style="font-size:12px;color:var(--muted);font-weight:600">Standard Classification</span>')}
    </section>

    <section class="grid page-grid" style="margin-top:20px">
      ${chartCard('Thermal Radiance Temperature Distribution (Channel I4 vs I5)', 'thermal-band-chart')}
      ${card('Top Extreme Mega-Fire Outbreaks', megaFiresHtml, '', '<span style="font-size:12px;color:var(--red);font-weight:600"><i class="fa-solid fa-fire"></i> Record Events</span>')}
    </section>

    <section class="grid secondary-grid" style="margin-top:20px">
      ${card('Radiative Energy Summary', statsHtml)}
      ${card('Thermal Energy Insights', insightsHtml)}
    </section>`;

  $('#content').innerHTML = html;

  // Render Log-binned FRP chart
  if (d.log_distribution && d.log_distribution.length) {
    plot('intensity-chart', [{
      x: d.log_distribution.map(x => x.label),
      y: d.log_distribution.map(x => x.count),
      type: 'bar',
      marker: {
        color: d.log_distribution.map(x => x.count),
        colorscale: [[0, '#fed7aa'], [0.4, '#f97316'], [0.8, '#ef4444'], [1, '#7c3aed']],
      },
    }], {
      xaxis: { title: 'Fire Radiative Power (MW Bins)', tickangle: -30, gridcolor: '#f1f5f9' },
      yaxis: { title: 'Detections Count', gridcolor: '#f1f5f9' },
    });
  } else if (d.histogram && d.histogram.length) {
    plot('intensity-chart', [{
      x: d.histogram.map(x => x.x), y: d.histogram.map(x => x.y),
      type: 'bar', marker: { color: '#ff8429' },
    }], { xaxis: { title: 'FRP (MW)' }, yaxis: { title: 'Detections' } });
  }

  // Render Thermal Band Chart
  if (d.thermal_bands && d.thermal_bands.available && d.thermal_bands.brightness_distribution?.length) {
    const tb = d.thermal_bands;
    const traces = [{
      name: 'Channel I4 (375m Firefront)',
      x: tb.brightness_distribution.map(x => x.x),
      y: tb.brightness_distribution.map(x => x.count),
      type: 'bar',
      marker: { color: 'rgba(239, 68, 68, 0.75)' },
    }];
    if (tb.t31_distribution && tb.t31_distribution.length) {
      traces.push({
        name: 'Channel I5 (11µm Ambient)',
        x: tb.t31_distribution.map(x => x.x),
        y: tb.t31_distribution.map(x => x.count),
        type: 'bar',
        marker: { color: 'rgba(14, 165, 233, 0.75)' },
      });
    }
    plot('thermal-band-chart', traces, {
      barmode: 'overlay',
      xaxis: { title: 'Brightness Temperature (Kelvin)', gridcolor: '#f1f5f9' },
      yaxis: { title: 'Detections', gridcolor: '#f1f5f9' },
      legend: { orientation: 'h', x: 0, y: 1.15 },
    });
  } else {
    $('#thermal-band-chart').innerHTML = empty('Dual-band brightness channels not present in this dataset.');
  }
}

async function confidencePage() {
  const d = await apiCached('confidence', filters());

  if (!d.available) {
    $('#content').innerHTML = `
      <div class="page-section-header">
        <div>
          <p class="page-eyebrow"><i class="fa-solid fa-shield-halved"></i> Quality Assurance & Sensor Reliability</p>
          <h2 class="page-heading">Confidence Classification & Sensor Verification</h2>
          <p class="page-subtext">Quantify sensor detection reliability, filter solar glint noise, and correlate algorithmic confidence with thermal radiance energy.</p>
        </div>
      </div>
      <section class="grid page-grid">
        ${card('Confidence Analysis', empty(d.message || 'No confidence telemetry available.'))}
      </section>`;
    return;
  }

  // 1. Confidence Classification Cards HTML
  let classListHtml = '';
  if (d.classes && d.classes.length) {
    classListHtml = `
      <div class="conf-list">
        ${d.classes.map(c => `
          <div class="conf-item ${c.class_name}">
            <div class="conf-top">
              <div class="conf-title-group">
                <span class="conf-pill">${esc(c.short)}</span>
                <span class="conf-name">${esc(c.label)}</span>
              </div>
              <div class="conf-count-pill">${c.count.toLocaleString()} <span>(${c.percentage}%)</span></div>
            </div>
            <div class="conf-desc">${esc(c.desc)}</div>
            <div class="conf-bar-wrap">
              <div class="conf-bar-fill" style="width:${Math.max(2, c.percentage)}%"></div>
            </div>
            <div class="conf-meta-grid">
              <div class="conf-meta-item">
                <span>Mean FRP</span>
                <strong>${c.avg_frp !== undefined ? `${c.avg_frp} MW` : 'N/A'}</strong>
              </div>
              <div class="conf-meta-item">
                <span>Max Radiative</span>
                <strong>${c.max_frp !== undefined ? `${c.max_frp.toLocaleString()} MW` : 'N/A'}</strong>
              </div>
              <div class="conf-meta-item">
                <span>Brightness Temp</span>
                <strong>${c.avg_brightness !== undefined ? `${c.avg_brightness} K` : 'N/A'}</strong>
              </div>
            </div>
          </div>
        `).join('')}
      </div>`;
  } else {
    classListHtml = empty('No classification classes available.');
  }

  // 2. Solar Glint & Diurnal False Alarm Analysis HTML
  let glintHtml = '';
  const lowClass = (d.classes || []).find(c => c.code === 'l');
  const highClass = (d.classes || []).find(c => c.code === 'h');
  const nomClass = (d.classes || []).find(c => c.code === 'n');

  if (lowClass && (lowClass.day_count !== undefined || lowClass.day_pct !== undefined)) {
    glintHtml = `
      <div class="glint-card">
        <div class="glint-stat-box">
          <div>
            <div style="font-size:12px;color:var(--muted);text-transform:uppercase;font-weight:700;letter-spacing:0.05em">Low Confidence Diurnal Glint Ratio</div>
            <div style="font-size:13px;color:var(--text);margin-top:2px">Daytime solar glint & warm bare soil bias</div>
          </div>
          <strong style="color:var(--orange)">${lowClass.day_pct || 0}% Day</strong>
        </div>

        <div style="display:flex;flex-direction:column;gap:10px">
          <div class="stat-row" style="padding:6px 0">
            <span style="display:flex;align-items:center;gap:8px">
              <span class="conf-pill" style="background:rgba(16,185,129,0.15);color:#059669;padding:2px 8px;font-size:10px">HIGH</span> Diurnal Ratio
            </span>
            <strong>${highClass ? `${highClass.day_pct}% Day · ${highClass.night_pct}% Night` : 'N/A'}</strong>
          </div>
          <div class="stat-row" style="padding:6px 0">
            <span style="display:flex;align-items:center;gap:8px">
              <span class="conf-pill" style="background:rgba(245,158,11,0.15);color:#d97706;padding:2px 8px;font-size:10px">NOMINAL</span> Diurnal Ratio
            </span>
            <strong>${nomClass ? `${nomClass.day_pct}% Day · ${nomClass.night_pct}% Night` : 'N/A'}</strong>
          </div>
          <div class="stat-row" style="padding:6px 0">
            <span style="display:flex;align-items:center;gap:8px">
              <span class="conf-pill" style="background:rgba(100,116,139,0.15);color:#475569;padding:2px 8px;font-size:10px">LOW</span> Diurnal Ratio
            </span>
            <strong style="color:var(--red)">${lowClass.day_pct}% Day · ${lowClass.night_pct}% Night</strong>
          </div>
        </div>

        <div style="background:var(--panel-2);border-radius:10px;padding:12px 14px;font-size:12px;color:var(--muted);line-height:1.5">
          <i class="fa-solid fa-triangle-exclamation" style="color:var(--orange);margin-right:6px"></i>
          <strong>Scientific Finding:</strong> Low-confidence pixels are predominantly flagged during daytime overpasses due to sub-pixel solar reflection off specular water bodies, dry desert surfaces, and sunlit cloud edges. At night, false alarm ambiguity drops close to zero.
        </div>
      </div>`;
  } else {
    glintHtml = empty('Diurnal observation fields not available for glint analysis.');
  }

  // 3. Operational Response Protocols HTML
  const protocolsHtml = `
    <div style="padding:16px 20px;display:flex;flex-direction:column;gap:12px">
      <div style="display:flex;gap:12px;align-items:flex-start">
        <div style="width:32px;height:32px;border-radius:8px;background:rgba(16,185,129,0.15);color:var(--green);display:grid;place-items:center;flex-shrink:0">
          <i class="fa-solid fa-fire-extinguisher"></i>
        </div>
        <div>
          <strong style="font-size:13px;color:var(--text)">High Confidence: Direct Operational Response</strong>
          <p style="font-size:12px;color:var(--muted);margin:2px 0 0;line-height:1.4">Thermal pixels reach saturation brightness (~367 K) with extreme FRP. Immediate airborne dispatch, suppression allocation, and wildfire containment alerts recommended.</p>
        </div>
      </div>

      <div style="display:flex;gap:12px;align-items:flex-start">
        <div style="width:32px;height:32px;border-radius:8px;background:rgba(245,158,11,0.15);color:var(--orange);display:grid;place-items:center;flex-shrink:0">
          <i class="fa-solid fa-chart-line"></i>
        </div>
        <div>
          <strong style="font-size:13px;color:var(--text)">Nominal Quality: Routine Monitoring & Modeling</strong>
          <p style="font-size:12px;color:var(--muted);margin:2px 0 0;line-height:1.4">Standard verified fire pixels under clear conditions. Optimal for spatial clustering, temporal burn-scar propagation, and automated boundary estimation.</p>
        </div>
      </div>

      <div style="display:flex;gap:12px;align-items:flex-start">
        <div style="width:32px;height:32px;border-radius:8px;background:rgba(100,116,139,0.15);color:#64748b;display:grid;place-items:center;flex-shrink:0">
          <i class="fa-solid fa-filter"></i>
        </div>
        <div>
          <strong style="font-size:13px;color:var(--text)">Low Confidence: Secondary Pass Filtering</strong>
          <p style="font-size:12px;color:var(--muted);margin:2px 0 0;line-height:1.4">Potential candidate thermal hotspots subject to glint or thin cirrus cloud scattering. Requires cross-verification with consecutive satellite overpasses before emergency escalation.</p>
        </div>
      </div>
    </div>`;

  // 4. Dynamic Scientific QA Insights HTML
  const insightsHtml = (d.insights && d.insights.length)
    ? `<div class="insights">
        ${d.insights.map(txt => `
          <div class="insight">
            <i class="fa-solid fa-shield-halved" style="color:var(--green);font-size:12px;margin-top:2px"></i>
            <div><p>${esc(txt)}</p></div>
          </div>
        `).join('')}
      </div>`
    : empty('No QA insights available.');

  let html = `
    <div class="page-section-header">
      <div>
        <p class="page-eyebrow"><i class="fa-solid fa-shield-halved"></i> Quality Assurance & Sensor Reliability</p>
        <h2 class="page-heading">Confidence Classification & Sensor Verification</h2>
        <p class="page-subtext">Quantify sensor detection reliability, filter solar glint noise, and correlate algorithmic confidence with thermal radiance energy.</p>
      </div>
    </div>`;

  if (d.kpis && d.kpis.length) {
    html += kpis(d.kpis);
  }

  html += `
    <section class="grid page-grid">
      ${card('Algorithm Quality Classification Breakdown', classListHtml, '', `<span class="platform-code-pill" style="color:var(--green);background:rgba(16,185,129,0.15)">${d.actionable_rate}% Actionable</span>`)}
      ${chartCard('Detection Quality Share', 'conf-donut-chart')}
    </section>

    <section class="grid page-grid" style="margin-top:20px">
      ${chartCard('Thermal Radiative Power (FRP) by Confidence Tier', 'conf-frp-chart')}
      ${card('Solar Glint & Diurnal False Alarm Analysis', glintHtml, '', '<span style="font-size:12px;color:var(--orange);font-weight:600"><i class="fa-solid fa-sun"></i> Optical Glint Filter</span>')}
    </section>

    <section class="grid secondary-grid" style="margin-top:20px">
      ${card('Operational Decision Protocols', protocolsHtml)}
      ${card('Detection Reliability Insights', insightsHtml)}
    </section>`;

  $('#content').innerHTML = html;

  // 5. Render Donut Chart
  if (d.classes && d.classes.length) {
    plot('conf-donut-chart', [{
      labels: d.classes.map(c => c.label),
      values: d.classes.map(c => c.count),
      type: 'pie',
      hole: 0.6,
      marker: {
        colors: d.classes.map(c => c.color || '#64748b'),
      },
      textinfo: 'label+percent',
      textposition: 'inside',
      insidetextorientation: 'horizontal',
      hovertemplate: '<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>',
    }], {
      showlegend: false,
      annotations: [{
        text: `<b>${d.actionable_rate}%</b><br><span style="font-size:11px;color:#64748b">Actionable</span>`,
        showarrow: false,
        font: { size: 16, color: 'var(--text)', family: 'Outfit' },
      }],
      margin: { t: 20, b: 20, l: 20, r: 20 },
    });

    // 6. Render FRP by Confidence Tier Chart
    const classesWithFrp = d.classes.filter(c => c.avg_frp !== undefined);
    if (classesWithFrp.length) {
      plot('conf-frp-chart', [
        {
          name: 'Average FRP (MW)',
          x: classesWithFrp.map(c => c.short),
          y: classesWithFrp.map(c => c.avg_frp),
          type: 'bar',
          marker: {
            color: classesWithFrp.map(c => c.color || '#f97316'),
          },
          text: classesWithFrp.map(c => `${c.avg_frp} MW`),
          textposition: 'auto',
        },
      ], {
        xaxis: { title: 'Confidence Tier', gridcolor: '#f1f5f9' },
        yaxis: { title: 'Average Radiative Power (MW)', gridcolor: '#f1f5f9' },
        margin: { t: 30, b: 40, l: 50, r: 20 },
      });
    } else {
      $('#conf-frp-chart').innerHTML = empty('Radiative power telemetry not available for confidence tiers.');
    }
  }
}

async function categoryPage(key, title, color) {
  const endpoint = key === 'sources' ? 'satellites' : key;
  const eyebrow = key === 'sources' ? 'Source Comparison' : 'Confidence Analysis';
  const icon = key === 'sources' ? 'fa-satellite' : 'fa-shield-halved';
  const d = await apiCached(endpoint, filters());
  $('#content').innerHTML = `
    <div class="page-section-header">
      <div>
        <p class="page-eyebrow"><i class="fa-solid ${icon}"></i> ${eyebrow}</p>
        <h2 class="page-heading">${esc(title)}</h2>
      </div>
    </div>
    <section class="grid page-grid">
      ${chartCard(title, 'category-chart')}
      ${card('Breakdown', d.available
        ? d.items.map(i => `<div class="stat-row"><span>${esc(i.label)}</span><strong>${i.value.toLocaleString()}</strong></div>`).join('')
        : empty(d.message))}
    </section>`;
  if (d.available) bar('category-chart', d.items, color);
}

async function sourcesPage() {
  const d = await apiCached('satellites', filters());

  if (!d.available) {
    $('#content').innerHTML = `
      <div class="page-section-header">
        <div>
          <p class="page-eyebrow"><i class="fa-solid fa-satellite"></i> Platform & Sensor Intelligence</p>
          <h2 class="page-heading">Satellite Observations & Sensor Telemetry</h2>
          <p class="page-subtext">Satellite observation platforms and sensor payload diagnostics.</p>
        </div>
      </div>
      <section class="grid page-grid">
        ${card('Source Comparison', empty(d.message || 'No satellite or sensor sources detected.'))}
      </section>`;
    return;
  }

  // 1. Platform List HTML
  let platformListHtml = '';
  if (d.platforms && d.platforms.length) {
    platformListHtml = `
      <div class="platform-list">
        ${d.platforms.map(p => `
          <div class="platform-item">
            <div class="platform-top">
              <div class="platform-title-group">
                <div class="platform-icon"><i class="fa-solid fa-satellite"></i></div>
                <div>
                  <h3 class="platform-name">${esc(p.name)}</h3>
                  <div class="platform-sensor-tag">
                    <span class="platform-code-pill">Code: ${esc(p.code)}</span>
                    <span>·</span>
                    <span>${esc(p.instrument)}</span>
                  </div>
                </div>
              </div>
              <div class="platform-share-pill">${p.count.toLocaleString()} <span>(${p.percentage}%)</span></div>
            </div>
            <div class="platform-bar-wrap">
              <div class="platform-bar-fill" style="width:${p.percentage}%"></div>
            </div>
            <div class="platform-meta-grid">
              <div class="platform-meta-item">
                <span>Sensor Payload</span>
                <strong>${esc(p.instrument.split(' ')[0])}</strong>
              </div>
              <div class="platform-meta-item">
                <span>Average FRP</span>
                <strong>${p.avg_frp ? `${p.avg_frp} MW` : 'N/A'}</strong>
              </div>
              <div class="platform-meta-item">
                <span>Peak FRP</span>
                <strong>${p.max_frp ? `${p.max_frp} MW` : 'N/A'}</strong>
              </div>
            </div>
          </div>
        `).join('')}
      </div>`;
  } else if (d.items && d.items.length) {
    platformListHtml = `
      <div class="insights">
        ${d.items.map(i => `
          <div class="stat-row">
            <span>${esc(i.label)}</span>
            <strong>${i.value.toLocaleString()}</strong>
          </div>
        `).join('')}
      </div>`;
  } else {
    platformListHtml = empty('No platforms detected.');
  }

  // 2. Day vs Night HTML
  let daynightHtml = '';
  if (d.daynight && d.daynight.available) {
    const dn = d.daynight;
    daynightHtml = `
      <div class="daynight-wrap">
        <div class="daynight-grid">
          <div class="daynight-box day">
            <div class="daynight-header">
              <span class="daynight-label"><i class="fa-solid fa-sun"></i> Solar Daytime</span>
              <span class="daynight-pct">${dn.day_pct}%</span>
            </div>
            <div class="daynight-sub">
              <span>${dn.day_count.toLocaleString()} detections</span>
              ${dn.day_avg_frp ? `<span>Avg: ${dn.day_avg_frp} MW</span>` : ''}
            </div>
          </div>
          <div class="daynight-box night">
            <div class="daynight-header">
              <span class="daynight-label"><i class="fa-solid fa-moon"></i> Nocturnal Thermal</span>
              <span class="daynight-pct">${dn.night_pct}%</span>
            </div>
            <div class="daynight-sub">
              <span>${dn.night_count.toLocaleString()} detections</span>
              ${dn.night_avg_frp ? `<span>Avg: ${dn.night_avg_frp} MW</span>` : ''}
            </div>
          </div>
        </div>
        <div class="daynight-split-bar">
          <div class="split-day" style="width:${dn.day_pct}%"></div>
          <div class="split-night" style="width:${dn.night_pct}%"></div>
        </div>
      </div>`;
  } else {
    daynightHtml = empty('Day/Night observation field not present in dataset.');
  }

  // 3. Confidence HTML
  let confHtml = '';
  if (d.confidence && d.confidence.available) {
    const c = d.confidence;
    confHtml = `
      <div class="daynight-wrap">
        <div class="stat-row" style="padding:12px 0">
          <span style="display:flex;align-items:center;gap:8px">
            <i class="fa-solid fa-shield-halved" style="color:var(--green)"></i> High Confidence
          </span>
          <strong>${c.high.toLocaleString()} <span style="font-size:12px;color:var(--muted);font-weight:500">(${c.high_pct}%)</span></strong>
        </div>
        <div class="stat-row" style="padding:12px 0">
          <span style="display:flex;align-items:center;gap:8px">
            <i class="fa-solid fa-circle-dot" style="color:var(--orange)"></i> Nominal Confidence
          </span>
          <strong>${c.nominal.toLocaleString()} <span style="font-size:12px;color:var(--muted);font-weight:500">(${c.nominal_pct}%)</span></strong>
        </div>
        <div class="stat-row" style="padding:12px 0">
          <span style="display:flex;align-items:center;gap:8px">
            <i class="fa-solid fa-circle-notch" style="color:var(--muted)"></i> Low / Unverified
          </span>
          <strong>${c.low.toLocaleString()} <span style="font-size:12px;color:var(--muted);font-weight:500">(${c.low_pct}%)</span></strong>
        </div>
      </div>`;
  } else {
    confHtml = empty('Detection confidence telemetry not detected.');
  }

  // 4. Viewing Geometry HTML
  let geometryHtml = '';
  if (d.geometry && d.geometry.available) {
    const g = d.geometry;
    geometryHtml = `
      <div class="geometry-wrap">
        <div class="geometry-grid">
          <div class="geometry-cell">
            <span>Pixel Footprint</span>
            <strong>${g.mean_footprint} km²</strong>
            <small>Mean ground area</small>
          </div>
          <div class="geometry-cell">
            <span>Cross-Track Scan</span>
            <strong>${g.mean_scan} km</strong>
            <small>Scan axis resolution</small>
          </div>
          <div class="geometry-cell">
            <span>Along-Track Track</span>
            <strong>${g.mean_track} km</strong>
            <small>Flight axis resolution</small>
          </div>
        </div>
        <div class="stat-row" style="padding:10px 0;border-top:1px solid var(--line)">
          <span>Near-Nadir Clarity (Scan ≤ 0.5 km)</span>
          <strong style="color:var(--cyan)">${g.nadir_pct}% of detections</strong>
        </div>
      </div>`;
  } else {
    geometryHtml = empty('Scan and track optical dimensions not detected in this file.');
  }

  // 5. Dynamic Insights HTML
  const insightsHtml = (d.insights && d.insights.length)
    ? `<div class="insights">
        ${d.insights.map(txt => `
          <div class="insight">
            <i class="fa-solid fa-satellite" style="color:var(--cyan);font-size:12px;margin-top:2px"></i>
            <div><p>${esc(txt)}</p></div>
          </div>
        `).join('')}
      </div>`
    : empty('No platform operational insights available.');

  let html = `
    <div class="page-section-header">
      <div>
        <p class="page-eyebrow"><i class="fa-solid fa-satellite"></i> Platform & Sensor Intelligence</p>
        <h2 class="page-heading">Satellite Observations & Sensor Telemetry</h2>
        <p class="page-subtext">Review active satellite platforms, orbital overpass patterns, day vs. night observations, and radiometer telemetry.</p>
      </div>
    </div>`;

  if (d.kpis && d.kpis.length) {
    html += kpis(d.kpis);
  }

  html += `
    <section class="grid page-grid">
      ${card('Active Satellite Platforms', platformListHtml, '', `<span class="platform-code-pill">${esc(d.pipeline_version || 'Collection 2')}</span>`)}
      ${chartCard('Orbital Overpass Schedule (UTC Acquisition Hours)', 'overpass-chart')}
    </section>

    <section class="grid page-grid" style="margin-top:20px">
      ${card('Day vs. Night Observation Split', daynightHtml)}
      ${card('Detection Confidence Profile', confHtml)}
    </section>

    <section class="grid secondary-grid" style="margin-top:20px">
      ${card('Viewing Geometry & Optical Resolution', geometryHtml)}
      ${card('Sensor Operational Insights', insightsHtml)}
    </section>`;

  $('#content').innerHTML = html;

  // Plotly Hourly Overpass chart
  if (d.hourly_overpass && d.hourly_overpass.length) {
    plot('overpass-chart', [{
      x: d.hourly_overpass.map(x => x.label),
      y: d.hourly_overpass.map(x => x.count),
      type: 'bar',
      marker: {
        color: d.hourly_overpass.map(x => x.count),
        colorscale: [[0, '#bae6fd'], [0.5, '#0ea5e9'], [1, '#6366f1']],
      },
    }], {
      xaxis: { title: 'Acquisition Hour (UTC)', tickangle: -45, gridcolor: '#f1f5f9' },
      yaxis: { title: 'Detections', gridcolor: '#f1f5f9' }
    });
  }
}

async function mapPage() {
  const d = await apiCached('map', filters());
  $('#content').innerHTML = card(
    'Interactive fire detection map',
    d.available ? '<div id="fire-map" class="map" style="height:calc(100vh - 170px);min-height:480px"></div>' : empty(d.message),
    'map-card'
  );
  if (d.available) renderMap(d);
}

function renderMiniHotspotMap(d) {
  if (state.miniMap) {
    try { state.miniMap.remove(); } catch (_) {}
    state.miniMap = null;
  }
  const el = document.getElementById('geo-mini-map');
  if (!el || !window.L) return;

  const centerLat = d.bounds?.center_lat || 20;
  const centerLon = d.bounds?.center_lon || 0;
  state.miniMap = L.map('geo-mini-map', { zoomControl: false, attributionControl: false }).setView([centerLat, centerLon], 3);
  L.control.zoom({ position: 'bottomright' }).addTo(state.miniMap);

  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    maxZoom: 18,
  }).addTo(state.miniMap);

  const bounds = [];

  // Plot hotspot clusters as prominent radiant markers
  if (d.clusters && d.clusters.length) {
    d.clusters.forEach((c, idx) => {
      const radius = Math.max(9, Math.min(24, 11 + (c.percentage * 0.9)));
      const color = idx === 0 ? '#ef4444' : idx < 3 ? '#ff8429' : '#f59e0b';
      const marker = L.circleMarker([c.lat, c.lon], {
        radius: radius,
        fillColor: color,
        color: '#ffffff',
        weight: 1.5,
        opacity: 0.95,
        fillOpacity: 0.7,
      }).bindPopup(`<b>${esc(c.label)}</b><br>Detections: <b>${c.count.toLocaleString()}</b> (${c.percentage}%)<br>${c.avg_frp ? `Avg FRP: ${c.avg_frp} MW<br>` : ''}${c.max_frp ? `Max FRP: ${c.max_frp} MW` : ''}`);
      marker.addTo(state.miniMap);
      bounds.push([c.lat, c.lon]);
    });
  }

  // Plot a sample of individual points with faint glow
  if (d.sample_points && d.sample_points.length) {
    d.sample_points.forEach(p => {
      const frp = p.frp || 0;
      const col = frp > 80 ? '#ef4444' : frp > 25 ? '#ff8429' : '#ffbd22';
      L.circleMarker([p.lat, p.lon], {
        radius: 3.5,
        fillColor: col,
        color: col,
        weight: 0,
        fillOpacity: 0.5,
      }).addTo(state.miniMap);
      bounds.push([p.lat, p.lon]);
    });
  }

  if (bounds.length) {
    state.miniMap.fitBounds(bounds, { padding: [35, 35], maxZoom: 6 });
  }
}

async function geography() {
  const d = await apiCached('geographic', filters());

  if (!d.available) {
    $('#content').innerHTML = `
      <div class="page-section-header">
        <div>
          <p class="page-eyebrow"><i class="fa-solid fa-earth-americas"></i> Geographic Insights</p>
          <h2 class="page-heading">Fire Hotspots & Spatial Intelligence</h2>
          <p class="page-subtext">Spatial pattern analysis, coordinate distributions, and hotspot clusters.</p>
        </div>
      </div>
      <section class="grid page-grid">
        ${card('Geographic Analytics', empty(d.message || 'No geographic coordinate data detected in the dataset.'))}
      </section>`;
    return;
  }

  // Render top cluster list HTML
  let clusterListHtml = '';
  if (d.clusters && d.clusters.length) {
    clusterListHtml = `
      <div class="hotspot-list">
        ${d.clusters.map((c, idx) => `
          <div class="hotspot-item">
            <div class="hotspot-rank">#${idx + 1}</div>
            <div class="hotspot-info">
              <div class="hotspot-title-row">
                <span class="hotspot-title">${esc(c.label)}</span>
                <span class="hotspot-count-pill">${c.count.toLocaleString()} <span>(${c.percentage}%)</span></span>
              </div>
              <div class="hotspot-bar-wrap">
                <div class="hotspot-bar-fill" style="width:${Math.max(6, Math.min(100, Math.round((c.relative_ratio || (c.percentage / d.clusters[0].percentage)) * 100)))}%"></div>
              </div>
              <div class="hotspot-sub">
                ${c.avg_frp ? `<span>Avg FRP: <strong>${c.avg_frp} MW</strong></span>` : ''}
                ${c.max_frp ? `<span>Peak FRP: <strong>${c.max_frp} MW</strong></span>` : ''}
                ${c.avg_brightness ? `<span>Brightness: <strong>${c.avg_brightness} K</strong></span>` : ''}
              </div>
            </div>
          </div>
        `).join('')}
      </div>`;
  } else if (d.items && d.items.length) {
    clusterListHtml = `
      <div class="insights">
        ${d.items.map(i => `
          <div class="stat-row">
            <span>${esc(i.label)}</span>
            <strong>${i.value.toLocaleString()}</strong>
          </div>`).join('')}
      </div>`;
  } else {
    clusterListHtml = empty('No spatial clusters detected in current filter view.');
  }

  // Quadrants breakdown HTML
  const q = d.quadrants || { north: 0, south: 0, east: 0, west: 0, north_pct: 0, south_pct: 0, east_pct: 0, west_pct: 0 };
  const quadrantHtml = `
    <div class="quadrant-grid">
      <div class="quadrant-box north">
        <div class="quadrant-header">
          <span class="quadrant-title"><i class="fa-solid fa-compass"></i> Northern Hemi</span>
          <span class="quadrant-pct">${q.north_pct}%</span>
        </div>
        <div class="quadrant-bar"><div class="quadrant-bar-fill" style="width:${q.north_pct}%"></div></div>
        <span class="quadrant-count">${q.north.toLocaleString()} detections</span>
      </div>
      <div class="quadrant-box south">
        <div class="quadrant-header">
          <span class="quadrant-title"><i class="fa-regular fa-compass"></i> Southern Hemi</span>
          <span class="quadrant-pct">${q.south_pct}%</span>
        </div>
        <div class="quadrant-bar"><div class="quadrant-bar-fill" style="width:${q.south_pct}%"></div></div>
        <span class="quadrant-count">${q.south.toLocaleString()} detections</span>
      </div>
      <div class="quadrant-box east">
        <div class="quadrant-header">
          <span class="quadrant-title"><i class="fa-solid fa-globe"></i> Eastern Hemi</span>
          <span class="quadrant-pct">${q.east_pct}%</span>
        </div>
        <div class="quadrant-bar"><div class="quadrant-bar-fill" style="width:${q.east_pct}%"></div></div>
        <span class="quadrant-count">${q.east.toLocaleString()} detections</span>
      </div>
      <div class="quadrant-box west">
        <div class="quadrant-header">
          <span class="quadrant-title"><i class="fa-regular fa-globe"></i> Western Hemi</span>
          <span class="quadrant-pct">${q.west_pct}%</span>
        </div>
        <div class="quadrant-bar"><div class="quadrant-bar-fill" style="width:${q.west_pct}%"></div></div>
        <span class="quadrant-count">${q.west.toLocaleString()} detections</span>
      </div>
    </div>`;

  // Dynamic Insights HTML
  const insightsHtml = (d.insights && d.insights.length)
    ? `<div class="insights">
        ${d.insights.map(txt => `
          <div class="insight">
            <i class="fa-solid fa-location-dot" style="color:var(--red);font-size:12px;margin-top:2px"></i>
            <div><p>${esc(txt)}</p></div>
          </div>
        `).join('')}
      </div>`
    : empty('No spatial patterns detected for this selection.');

  let html = `
    <div class="page-section-header">
      <div>
        <p class="page-eyebrow"><i class="fa-solid fa-earth-americas"></i> Geographic Insights</p>
        <h2 class="page-heading">Fire Hotspots & Spatial Intelligence</h2>
        <p class="page-subtext">Spatial density clustering, latitudinal-longitudinal profiles, and regional fire concentration patterns.</p>
      </div>
    </div>`;

  if (d.kpis && d.kpis.length) {
    html += kpis(d.kpis);
  }

  html += `
    <section class="grid page-grid">
      ${card('Top Fire Hotspot Clusters', clusterListHtml, '', '<span style="font-size:12px;color:var(--muted);font-weight:600">Density Grid</span>')}
      ${card('Spatial Cluster Map Preview', `
        <div class="geo-mini-map-wrap">
          <div id="geo-mini-map" class="geo-mini-map"></div>
          <div class="geo-map-badge"><i class="fa-solid fa-layer-group"></i> Active Fire Zones</div>
        </div>
      `, '', '<a href="#" id="open-full-map-btn" style="font-size:12px;color:var(--cyan);font-weight:600;text-decoration:none">Full Map Explorer <i class="fa-solid fa-arrow-right"></i></a>')}
    </section>

    <section class="grid page-grid" style="margin-top:20px">
      ${chartCard('Latitudinal Distribution (South to North)', 'geo-lat-chart')}
      ${chartCard('Longitudinal Distribution (West to East)', 'geo-lon-chart')}
    </section>

    <section class="grid secondary-grid" style="margin-top:20px">
      ${card('Hemispheric Distribution', quadrantHtml)}
      ${card('Spatial Pattern Insights', insightsHtml)}
    </section>`;

  $('#content').innerHTML = html;

  // Mini map rendering
  requestAnimationFrame(() => {
    renderMiniHotspotMap(d);
  });

  // Link to Map Explorer
  const mapBtn = $('#open-full-map-btn');
  if (mapBtn) {
    mapBtn.addEventListener('click', e => {
      e.preventDefault();
      const mapNav = document.querySelector('[data-view="map"]');
      if (mapNav) mapNav.click();
    });
  }

  // Plot Latitude distribution
  if (d.lat_distribution && d.lat_distribution.length) {
    plot('geo-lat-chart', [{
      x: d.lat_distribution.map(x => x.x),
      y: d.lat_distribution.map(x => x.count),
      type: 'bar',
      marker: {
        color: d.lat_distribution.map(x => x.count),
        colorscale: [[0, '#fecaca'], [0.5, '#f87171'], [1, '#dc2626']],
      },
    }], {
      xaxis: { title: 'Latitude (°)', gridcolor: '#f1f5f9' },
      yaxis: { title: 'Detections', gridcolor: '#f1f5f9' }
    });
  }

  // Plot Longitude distribution
  if (d.lon_distribution && d.lon_distribution.length) {
    plot('geo-lon-chart', [{
      x: d.lon_distribution.map(x => x.x),
      y: d.lon_distribution.map(x => x.count),
      type: 'bar',
      marker: {
        color: d.lon_distribution.map(x => x.count),
        colorscale: [[0, '#bae6fd'], [0.5, '#38bdf8'], [1, '#0284c7']],
      },
    }], {
      xaxis: { title: 'Longitude (°)', gridcolor: '#f1f5f9' },
      yaxis: { title: 'Detections', gridcolor: '#f1f5f9' }
    });
  }
}

async function dataPage() {
  const [schema, q, preview] = await Promise.all([
    apiCached('schema'),
    apiCached('data-quality', filters()),
    apiCached('data', filters()),
  ]);
  const kindColor = { geographic: 'badge-geo', numeric: 'badge-num', datetime: 'badge-dt', categorical: 'badge-cat' };
  const header = schema.fields.map(f => `<th>${esc(f.name)}</th>`).join('');
  const rows   = preview.rows.map(r => `<tr>${schema.fields.map(f => `<td>${esc(r[f.name])}</td>`).join('')}</tr>`).join('');
  $('#content').innerHTML = `
    <div class="page-section-header">
      <div>
        <p class="page-eyebrow"><i class="fa-solid fa-table-columns"></i> Data Overview</p>
        <h2 class="page-heading">Dataset Quality & Schema</h2>
        <p class="page-subtext">Inspect dataset health metrics, detected field types, and a live preview of filtered rows.</p>
      </div>
    </div>
    <section class="grid page-grid">
      ${card('Data quality', `
        <div class="stat-row"><span>Records / columns</span><strong>${q.records.toLocaleString()} / ${q.columns}</strong></div>
        <div class="stat-row"><span>Duplicate rows</span><strong class="${q.duplicates > 0 ? 'stat-warn' : 'stat-ok'}">${q.duplicates.toLocaleString()}</strong></div>
        <div class="stat-row"><span>Missing cells</span><strong class="${q.missing_total > 0 ? 'stat-warn' : 'stat-ok'}">${q.missing_total.toLocaleString()}</strong></div>
        <div class="stat-row"><span>Memory footprint</span><strong>${q.memory_mb} MB</strong></div>
      `)}
      ${card('Detected schema', `
        <div class="table-wrap"><table class="data-table">
          <thead><tr><th>Column</th><th>Type</th><th>Kind</th><th>Missing</th><th>Unique</th></tr></thead>
          <tbody>${schema.fields.map(f => `
            <tr>
              <td><strong>${esc(f.name)}</strong></td><td style="color:var(--muted)">${esc(f.type)}</td>
              <td><span class="badge ${kindColor[f.kind] || ''}">${esc(f.kind)}</span></td>
              <td>${f.missing_pct > 0 ? `<span class="stat-warn">${f.missing_pct}%</span>` : '0%'}</td>
              <td>${f.unique.toLocaleString()}</td>
            </tr>`).join('')}
          </tbody>
        </table></div>`, 'full')}
      ${card('Filtered data preview', `
        <div class="table-wrap"><table class="data-table">
          <thead><tr>${header}</tr></thead>
          <tbody>${rows}</tbody>
        </table></div>
        <div style="padding:12px 24px;font-size:12px;color:var(--muted);font-weight:500">Showing ${preview.rows.length} of ${preview.total.toLocaleString()} filtered rows</div>
      `, 'full')}
    </section>`;
}

function downloadPage() {
  $('#content').innerHTML = `
    <div class="page-section-header">
      <div>
        <p class="page-eyebrow"><i class="fa-solid fa-download"></i> Export</p>
        <h2 class="page-heading">Download Filtered Data</h2>
        <p class="page-subtext">Export a CSV of the rows matching your currently active filters.</p>
      </div>
    </div>
    ${card('Export the active view', `
      <div class="about">
        <p>The exported CSV includes exactly the rows matching your active filters and search query. Original values and column names are preserved.</p>
        <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:8px">
          <button id="download-csv" class="button primary" style="display:flex;align-items:center;gap:10px">
            <i class="fa-solid fa-download"></i> Download filtered CSV
          </button>
        </div>
      </div>
    `)}`;
  $('#download-csv').onclick = async () => {
    const btn = $('#download-csv');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Preparing…';
    try {
      const response = await fetch('/api/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(filters()),
      });
      if (!response.ok) throw new Error('Download failed');
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = 'firescope-filtered.csv'; a.click();
      URL.revokeObjectURL(url);
    } catch (e) { alert(e.message); }
    finally {
      btn.disabled = false;
      btn.innerHTML = '<i class="fa-solid fa-download"></i> Download filtered CSV';
    }
  };
}

function about() {
  const detected = state.schema?.detected || {};
  $('#content').innerHTML = `
    <div class="page-section-header">
      <div>
        <p class="page-eyebrow"><i class="fa-solid fa-circle-info"></i> About Dataset</p>
        <h2 class="page-heading">Project & Dataset Info</h2>
      </div>
    </div>
    ${card('About this dataset', `
      <div class="about">
        <h2>Independent fire-data analysis</h2>
        <p>FireScope is an independent exploratory analytics project. It does not represent, endorse, or use the branding of NASA or any satellite operator.</p>
        <p>The loaded dataset is a tabular active-fire detection export. The application inspects its schema at runtime and adapts available analyses to detected fields, rather than assuming a fixed format.</p>
        <h2>Detected capabilities</h2>
        <p>${Object.entries(detected).filter(([, v]) => v)
          .map(([k, v]) => `<span class="badge">${esc(k)}: ${esc(v)}</span>`).join(' ')
          || 'Dataset detection is in progress.'}</p>
        <h2>Future direction</h2>
        <p>The application separates loading, analytics, and presentation layers so forecasting, hotspot clustering, or risk modeling can be added later without redesigning the dashboard.</p>
      </div>
    `)}`;
}

/* ─── Upload & Train page ────────────────────────────────────────────────── */

function trainPage() {
  $('#content').innerHTML = `
    <div class="train-page">

      <p class="train-section-title">Step 1 — Dataset Management</p>
      <div class="drop-zone" id="train-drop-zone">
        <input type="file" id="train-file-input" accept=".csv,.xlsx,.xls">
        <div class="drop-zone-icon"><i class="fa-solid fa-cloud-arrow-up"></i></div>
        <h3>Drag &amp; drop a CSV or Excel file here to add</h3>
        <p>Upload a new dataset while keeping existing ones intact (<strong>.csv</strong>, <strong>.xlsx</strong>, <strong>.xls</strong>)</p>
      </div>

      <div class="dataset-lib-container">
        <div class="dataset-lib-header">
          <h4>Available Datasets (<span id="dataset-count-num">0</span>)</h4>
          <p>Keep multiple files on the server. Click <strong>Activate</strong> to switch datasets, or delete individual uploads anytime.</p>
        </div>
        <div class="dataset-items-list" id="dataset-items-list">
          <div class="empty">Loading datasets…</div>
        </div>
      </div>

      <p class="train-section-title">Step 2 — Select models to train</p>
      <div class="model-grid">
        <label class="model-card selected" id="mc-regression">
          <input type="checkbox" value="regression" checked>
          <div class="model-card-icon reg"><i class="fa-solid fa-fire"></i></div>
          <div>
            <h4>Fire Intensity Predictor</h4>
            <p>Predicts fire radiative power (FRP) from location, brightness, and confidence features.</p>
            <span class="algo-badge">Random Forest Regressor</span>
          </div>
        </label>
        <label class="model-card selected" id="mc-clustering">
          <input type="checkbox" value="clustering" checked>
          <div class="model-card-icon clu"><i class="fa-solid fa-location-crosshairs"></i></div>
          <div>
            <h4>Hotspot Clustering</h4>
            <p>Groups fire detections into geographic clusters to identify high-activity zones.</p>
            <span class="algo-badge">K-Means</span>
            <div class="k-slider-row">
              <label>Clusters (k)</label>
              <input type="range" id="k-slider" min="2" max="15" value="5">
              <span class="k-val" id="k-val">5</span>
            </div>
          </div>
        </label>
        <label class="model-card selected" id="mc-classification">
          <input type="checkbox" value="classification" checked>
          <div class="model-card-icon cls"><i class="fa-solid fa-shield-halved"></i></div>
          <div>
            <h4>Confidence Classifier</h4>
            <p>Classifies detection confidence (High / Medium / Low) from radiative power and brightness.</p>
            <span class="algo-badge">Random Forest Classifier</span>
          </div>
        </label>
      </div>

      <div class="train-btn-row">
        <button class="train-btn" id="train-btn">
          <span class="spinner"></span>
          <i class="fa-solid fa-play btn-icon"></i>
          Train Selected Models
        </button>
        <span id="train-status-msg" style="font-size:13px;color:var(--muted)"></span>
      </div>

      <div id="train-results" style="display:none">
        <p class="train-section-title">Results</p>
        <div class="results-grid" id="results-grid"></div>
      </div>

    </div>`;

  // ── drop zone wiring ──
  const zone = $('#train-drop-zone');
  const fileInput = $('#train-file-input');

  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f) handleTrainFile(f);
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleTrainFile(fileInput.files[0]);
  });

  // ── model card toggle ──
  document.querySelectorAll('.model-card').forEach(card => {
    card.addEventListener('click', () => {
      const cb = card.querySelector('input[type=checkbox]');
      cb.checked = !cb.checked;
      card.classList.toggle('selected', cb.checked);
    });
  });

  // ── k slider ──
  const kSlider = $('#k-slider'), kVal = $('#k-val');
  if (kSlider) kSlider.addEventListener('input', () => { kVal.textContent = kSlider.value; });

  // ── train button ──
  $('#train-btn').addEventListener('click', runTraining);

  // Load dataset library cards
  loadDatasetLibrary();
}

async function loadDatasetLibrary() {
  const container = $('#dataset-items-list');
  if (!container) return;
  try {
    const res = await fetch('/api/datasets');
    const data = await res.json();
    const datasets = data.datasets || [];
    const countBadge = $('#dataset-count-num');
    if (countBadge) countBadge.textContent = datasets.length;

    if (!datasets.length) {
      container.innerHTML = '<div class="empty">No datasets available. Drop a CSV or Excel file above to add one.</div>';
      return;
    }

    container.innerHTML = datasets.map(d => `
      <div class="dataset-item-card ${d.active ? 'active' : ''}">
        <div class="dataset-item-icon ${d.active ? 'active-icon' : ''}">
          <i class="fa-solid ${d.active ? 'fa-database' : 'fa-file-csv'}"></i>
        </div>
        <div class="dataset-item-info">
          <div class="dataset-item-name">
            <strong>${esc(d.name)}</strong>
            ${d.active ? '<span class="dataset-active-pill"><i class="fa-solid fa-check"></i> Active</span>' : ''}
          </div>
          <div class="dataset-item-meta">
            <span>${d.size_mb} MB</span>
            <span>·</span>
            <span>${d.source === 'archive' ? 'System Archive' : 'Uploaded File'}</span>
          </div>
        </div>
        <div class="dataset-item-actions">
          ${!d.active ? `
            <button class="action-btn switch-dataset-btn" data-path="${esc(d.path)}" data-name="${esc(d.name)}" type="button">
              <i class="fa-solid fa-arrow-right-arrow-left"></i> Activate
            </button>` : `
            <span class="active-indicator"><i class="fa-solid fa-circle-dot"></i> In Use</span>`}
          <button class="action-btn delete-dataset-btn" data-name="${esc(d.name)}" type="button" title="Delete this file from server">
            <i class="fa-solid fa-trash-can"></i> Delete
          </button>
        </div>
      </div>
    `).join('');

    container.querySelectorAll('.switch-dataset-btn').forEach(btn => {
      btn.addEventListener('click', () => switchDataset(btn.dataset.path, btn.dataset.name));
    });

    container.querySelectorAll('.delete-dataset-btn').forEach(btn => {
      btn.addEventListener('click', () => deleteDataset(btn.dataset.name));
    });

  } catch (err) {
    container.innerHTML = `<div class="error">${esc(err.message)}</div>`;
  }
}

async function switchDataset(path, name) {
  try {
    const res = await fetch('/api/datasets/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, name })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Failed to switch dataset');
    cacheClear();
    state.filters = {};
    const fc = document.getElementById('filter-count');
    if (fc) fc.textContent = '0';
    state.schema = data.dataset;
    updateDatasetBadges(data.active, data.dataset?.records);
    await loadDatasetLibrary();
    const optRes = await apiCached('filter-options');
    state.options = optRes.options;
    filterFields();
    updateDateRangeBadge();
    if (window.showToast) window.showToast('Dataset Activated', `Switched active dataset to ${name}`, 'fire');
  } catch (err) {
    alert('Could not switch dataset: ' + err.message);
  }
}

async function deleteDataset(filename) {
  if (!confirm(`Are you sure you want to delete "${filename}"? Other datasets will remain safe.`)) return;
  try {
    const res = await fetch(`/api/datasets/${encodeURIComponent(filename)}`, { method: 'DELETE' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Failed to delete dataset');
    cacheClear();
    state.filters = {};
    const fc = document.getElementById('filter-count');
    if (fc) fc.textContent = '0';
    state.schema = data.dataset;
    const activeName = (data.datasets && data.datasets.find(d => d.active)?.name) || state.schema?.dataset || 'No Dataset';
    updateDatasetBadges(activeName, data.dataset?.records);
    await loadDatasetLibrary();
    const optRes = await apiCached('filter-options');
    state.options = optRes.options;
    filterFields();
    updateDateRangeBadge();
    if (window.showToast) window.showToast('Dataset Deleted', data.message, 'bell');
  } catch (err) {
    alert('Could not delete dataset: ' + err.message);
  }
}

function updateDatasetBadges(name, count) {
  const headerTag = document.getElementById('header-dataset-name');
  if (headerTag) headerTag.textContent = name || 'No Dataset';
  const countEl = document.getElementById('dataset-count-val');
  if (countEl) countEl.textContent = count != null ? count.toLocaleString() : '-';
}

function updateDateRangeBadge() {
  const dateEl = document.getElementById('header-date-range');
  if (!dateEl) return;
  const start = state.filters.date_start;
  const end = state.filters.date_end;
  if (start && end) {
    dateEl.textContent = `${start} – ${end}`;
  } else if (start) {
    dateEl.textContent = `From ${start}`;
  } else if (end) {
    dateEl.textContent = `Until ${end}`;
  } else if (state.options?.dates?.min && state.options?.dates?.max) {
    dateEl.textContent = `${state.options.dates.min} – ${state.options.dates.max}`;
  } else {
    dateEl.textContent = 'All Available Dates';
  }
}

function handleTrainFile(file) {
  const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
  if (!['.csv', '.xlsx', '.xls'].includes(ext)) {
    if (window.showToast) window.showToast('Upload Error', 'Only CSV and Excel files are supported', 'bell');
    return;
  }
  const formData = new FormData();
  formData.append('dataset', file);
  if (window.showToast) window.showToast('Uploading', `Uploading ${file.name}…`, 'fire');

  fetch('/api/upload', { method: 'POST', body: formData })
    .then(r => r.json())
    .then(data => {
      if (data.error) {
        if (window.showToast) window.showToast('Upload Failed', data.error, 'bell');
        return;
      }
      cacheClear();
      state.filters = {};
      const fc = document.getElementById('filter-count');
      if (fc) fc.textContent = '0';
      state.schema = data.dataset;
      updateDatasetBadges(data.active, data.dataset?.records);
      loadDatasetLibrary();
      apiCached('filter-options').then(optRes => {
        state.options = optRes.options;
        filterFields();
        updateDateRangeBadge();
      });
      if (window.showToast) window.showToast('Dataset Added', `Added and activated "${file.name}"`, 'fire');
    })
    .catch(err => {
      if (window.showToast) window.showToast('Upload Failed', err.message, 'bell');
    });
}


async function runTraining() {
  const selectedModels = [...document.querySelectorAll('.model-card input[type=checkbox]:checked')]
    .map(cb => cb.value);
  if (!selectedModels.length) {
    $('#train-status-msg').textContent = 'Select at least one model.';
    return;
  }

  const btn = $('#train-btn');
  const kSlider = $('#k-slider');
  btn.classList.add('loading');
  btn.disabled = true;
  $('#train-status-msg').textContent = 'Training in progress…';

  try {
    const data = await fetch('/api/ml/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        models: selectedModels,
        n_clusters: kSlider ? parseInt(kSlider.value) : 5,
        n_estimators: 100,
      }),
    }).then(r => r.json());

    if (data.error) throw new Error(data.error);
    renderTrainResults(data.results);
    const count = Object.keys(data.results).length;
    $('#train-status-msg').textContent = `${count} model${count > 1 ? 's' : ''} trained successfully.`;
  } catch (e) {
    $('#train-status-msg').textContent = 'Training failed: ' + e.message;
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}

function renderTrainResults(results) {
  const container = $('#results-grid');
  $('#train-results').style.display = 'block';
  container.innerHTML = '';

  const modelMeta = {
    regression:     { icon: 'fa-fire', iconCls: 'reg',   color: '#ff8429' },
    clustering:     { icon: 'fa-location-crosshairs', iconCls: 'clu', color: '#22c7dc' },
    classification: { icon: 'fa-shield-halved', iconCls: 'cls', color: '#9b74e8' },
  };

  for (const [key, r] of Object.entries(results)) {
    const meta = modelMeta[key] || { icon: 'fa-robot', iconCls: 'reg', color: '#ff8429' };
    const card = document.createElement('div');
    card.className = 'result-card';

    if (!r.available) {
      card.innerHTML = `
        <div class="result-card-header">
          <div class="rc-icon model-card-icon ${meta.iconCls}"><i class="fa-solid ${meta.icon}"></i></div>
          <h3>${esc(r.label || key)}</h3>
        </div>
        <div class="unavail-card"><i class="fa-solid fa-circle-exclamation"></i><span>${esc(r.message)}</span></div>`;
      container.appendChild(card);
      continue;
    }

    // Build metric cells
    let metricCells = '';
    if (key === 'regression') {
      metricCells = `
        <div class="metric-cell"><div class="metric-val">${r.metrics.r2}</div><div class="metric-label">R²</div></div>
        <div class="metric-cell"><div class="metric-val">${r.metrics.mae} MW</div><div class="metric-label">MAE</div></div>
        <div class="metric-cell"><div class="metric-val">${r.metrics.rmse} MW</div><div class="metric-label">RMSE</div></div>`;
    } else if (key === 'clustering') {
      metricCells = `
        <div class="metric-cell"><div class="metric-val">${r.metrics.silhouette}</div><div class="metric-label">Silhouette</div></div>
        <div class="metric-cell"><div class="metric-val">${r.n_clusters}</div><div class="metric-label">Clusters</div></div>
        <div class="metric-cell"><div class="metric-val">${r.metrics.n_points.toLocaleString()}</div><div class="metric-label">Points</div></div>`;
    } else {
      metricCells = `
        <div class="metric-cell"><div class="metric-val">${(r.metrics.accuracy * 100).toFixed(1)}%</div><div class="metric-label">Accuracy</div></div>
        <div class="metric-cell"><div class="metric-val">${r.train_rows.toLocaleString()}</div><div class="metric-label">Train rows</div></div>
        <div class="metric-cell"><div class="metric-val">${r.test_rows.toLocaleString()}</div><div class="metric-label">Test rows</div></div>`;
    }

    const chartId = `train-chart-${key}`;
    const dlName  = key === 'regression' ? 'regression_frp' : key === 'clustering' ? 'clustering_kmeans' : 'classification_confidence';

    card.innerHTML = `
      <div class="result-card-header">
        <div class="rc-icon model-card-icon ${meta.iconCls}"><i class="fa-solid ${meta.icon}"></i></div>
        <h3>${esc(r.label)}</h3>
        <span class="rc-time">Trained in ${r.training_seconds}s</span>
      </div>
      <div class="metrics-row">${metricCells}</div>
      ${ (key === 'classification' && r.per_class?.length) ? `
        <table class="per-class-table">
          <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
          <tbody>${r.per_class.map(c => `
            <tr><td>${esc(c.label)}</td><td>${c.precision}</td><td>${c.recall}</td><td>${c.f1}</td><td>${c.support.toLocaleString()}</td></tr>
          `).join('')}</tbody>
        </table>` : '' }
      <div class="result-chart" id="${chartId}"></div>
      <div class="result-card-footer">
        <button class="download-btn" data-model="${dlName}">
          <i class="fa-solid fa-download"></i> Download .pkl
        </button>
      </div>`;

    container.appendChild(card);

    // Render chart
    requestAnimationFrame(() => {
      if (key === 'regression' && r.feature_importances?.length) {
        const fi = r.feature_importances.slice(0, 8);
        Plotly.newPlot(chartId,
          [{ x: fi.map(x => x.importance), y: fi.map(x => x.feature), type: 'bar', orientation: 'h', marker: { color: meta.color } }],
          { ...PLOT_DEFAULTS, xaxis: { title: 'Importance', gridcolor: '#213448' }, yaxis: { gridcolor: '#213448' }, margin: { l: 100, r: 16, t: 20, b: 32 } },
          { responsive: true, displayModeBar: false });
      } else if (key === 'clustering' && r.cluster_sizes?.length) {
        Plotly.newPlot(chartId,
          [{ x: r.cluster_sizes.map(c => c.cluster), y: r.cluster_sizes.map(c => c.count), type: 'bar', marker: { color: meta.color } }],
          { ...PLOT_DEFAULTS, xaxis: { gridcolor: '#213448' }, yaxis: { title: 'Detections', gridcolor: '#213448' }, margin: { l: 56, r: 16, t: 20, b: 42 } },
          { responsive: true, displayModeBar: false });
      } else if (key === 'classification' && r.feature_importances?.length) {
        const fi = r.feature_importances.slice(0, 8);
        Plotly.newPlot(chartId,
          [{ x: fi.map(x => x.importance), y: fi.map(x => x.feature), type: 'bar', orientation: 'h', marker: { color: meta.color } }],
          { ...PLOT_DEFAULTS, xaxis: { title: 'Importance', gridcolor: '#213448' }, yaxis: { gridcolor: '#213448' }, margin: { l: 100, r: 16, t: 20, b: 32 } },
          { responsive: true, displayModeBar: false });
      }
    });
  }

  // Download handlers
  container.querySelectorAll('.download-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const name = btn.dataset.model;
      const a = document.createElement('a');
      a.href = `/api/ml/download/${name}`;
      a.download = `${name}.pkl`;
      a.click();
    });
  });
}

const viewFns = {
  overview,
  map: mapPage,
  temporal,
  intensity,
  confidence: confidencePage,
  sources:    sourcesPage,
  geography,
  data:       dataPage,
  download:   downloadPage,
  train:      trainPage,
  about,
  alerts:     () => window.alertsView ? window.alertsView() : Promise.resolve(),
};

/* ─── Render engine ──────────────────────────────────────────────────────── */

async function render() {
  try {
    $('#error').classList.add('hidden');
    
    // Toggle hero banner visibility
    const hero = document.querySelector('.hero-banner');
    if (hero) {
      hero.style.display = state.view === 'overview' ? 'flex' : 'none';
    }

    // Show skeleton immediately for content-heavy views, plain spinner for others
    if (['overview', 'data'].includes(state.view)) {
      $('#content').innerHTML = loadingContent();
    } else {
      $('#content').innerHTML = '<div class="empty">Updating analytics…</div>';
    }
    await viewFns[state.view]();
  } catch (e) {
    $('#error').textContent = e.message;
    $('#error').classList.remove('hidden');
    $('#content').innerHTML = empty('Analytics could not be loaded. Check the dataset configuration and try again.');
  }
}

/* ─── Filter drawer ──────────────────────────────────────────────────────── */

function filterFields() {
  const o = state.options;
  const select = (key, label) => o[key]?.length ? `
    <div class="filter-group">
      <label>${label}</label>
      <select data-filter="${key}">
        <option value="">All values</option>
        ${o[key].map(v => `<option value="${esc(v)}" ${state.filters[key] === v ? 'selected' : ''}>${esc(v)}</option>`).join('')}
      </select>
    </div>` : '';

  let h = `
    <div class="filter-group">
      <label>Date range</label>
      <div class="range">
        <input data-filter="date_start" type="date" min="${o.dates?.min || ''}" max="${o.dates?.max || ''}" value="${state.filters.date_start || ''}">
        <input data-filter="date_end"   type="date" min="${o.dates?.min || ''}" max="${o.dates?.max || ''}" value="${state.filters.date_end || ''}">
      </div>
    </div>`;
  h += select('source', 'Source / satellite') + select('confidence', 'Confidence') + select('daynight', 'Day / night');
  for (const key of ['frp', 'brightness']) {
    if (o[key]) h += `
      <div class="filter-group">
        <label>${key.toUpperCase()} range</label>
        <div class="range">
          <input data-filter="${key}_min" type="number" placeholder="Min ${o[key].min.toFixed(1)}" value="${state.filters[key + '_min'] || ''}">
          <input data-filter="${key}_max" type="number" placeholder="Max ${o[key].max.toFixed(1)}" value="${state.filters[key + '_max'] || ''}">
        </div>
      </div>`;
  }
  $('#filter-fields').innerHTML = h;
}

/* ─── Event listeners ────────────────────────────────────────────────────── */

$('#nav').addEventListener('click', e => {
  const b = e.target.closest('[data-view]');
  if (!b) return;
  state.view = b.dataset.view;
  document.querySelectorAll('.nav-item').forEach(x => x.classList.toggle('active', x === b));
  const pageTitle = $('#page-title');
  if (pageTitle) pageTitle.textContent = b.textContent.trim();
  $('#sidebar').classList.remove('open');
  render();
});

$('#filter-toggle').onclick = () => { $('#drawer').classList.add('open'); $('#scrim').classList.add('show'); };
$('#filter-close').onclick = $('#scrim').onclick = () => { $('#drawer').classList.remove('open'); $('#scrim').classList.remove('show'); };

$('#apply-filters').onclick = () => {
  state.filters = {};
  document.querySelectorAll('[data-filter]').forEach(x => { if (x.value) state.filters[x.dataset.filter] = x.value; });
  $('#filter-count').textContent = Object.keys(state.filters).length;
  updateDateRangeBadge();
  cacheClear(); // invalidate cache when filters change
  $('#filter-close').click();
  render();
};

$('#reset-filters').onclick = () => {
  state.filters = {};
  cacheClear();
  filterFields();
  $('#filter-count').textContent = '0';
  updateDateRangeBadge();
};

const mobileMenuBtn = $('.mobile-menu');
if (mobileMenuBtn) mobileMenuBtn.onclick = () => $('#sidebar').classList.toggle('open');

let _searchTimer;
$('#search').addEventListener('input', e => {
  clearTimeout(_searchTimer);
  _searchTimer = setTimeout(() => {
    const q = e.target.value.trim();
    if (q) state.filters.search = q; else delete state.filters.search;
    $('#filter-count').textContent = Object.keys(state.filters).length;
    cacheClear();
    render();
  }, 450);
});

/* ─── Boot ───────────────────────────────────────────────────────────────── */

async function load() {
  const [schema, optionsRes] = await Promise.all([
    apiCached('schema'),
    apiCached('filter-options')
  ]);
  state.schema = schema;
  state.options = optionsRes.options;
  updateDatasetBadges(state.schema?.dataset, state.schema?.records);
  updateDateRangeBadge();

  const headerPicker = $('#header-dataset-picker');
  if (headerPicker) {
    headerPicker.addEventListener('click', () => {
      const trainNav = document.querySelector('[data-view="train"]');
      if (trainNav) trainNav.click();
    });
  }

  const headerDatePicker = $('#header-date-picker');
  if (headerDatePicker) {
    headerDatePicker.addEventListener('click', () => {
      $('#drawer').classList.add('open');
      $('#scrim').classList.add('show');
      const startInput = document.querySelector('[data-filter="date_start"]');
      if (startInput) startInput.focus();
    });
  }

  // Share schema/options with the alerts module for dropdown population
  if (window.setAlertContext) window.setAlertContext(state.schema, state.options);
  filterFields();
  render();
}

load();


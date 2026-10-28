/**
 * FireScope — Alerts & Notifications Module
 *
 * Responsibilities:
 *  - Render the Alerts manager view (create, list, toggle, delete rules)
 *  - Poll /api/alerts/check every 60 s and show toast notifications
 *  - Badge count on the sidebar nav item
 */

/* ─── Constants ─────────────────────────────────────────────────────────── */

const POLL_INTERVAL_MS = 60_000;
const API_ALERTS = '/api/alerts';

/* ─── Internal state ────────────────────────────────────────────────────── */

let _pollTimer = null;
let _firingIds = new Set(); // IDs of already-toasted alerts this session

/* ─── Toast system ──────────────────────────────────────────────────────── */

function showToast(title, message, type = 'fire') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const id = `toast-${Date.now()}`;
  const icon = type === 'fire' ? 'fa-fire-flame-curved' : 'fa-bell';
  const toast = document.createElement('div');
  toast.className = 'toast toast-enter';
  toast.id = id;
  toast.setAttribute('role', 'alert');
  toast.innerHTML = `
    <div class="toast-icon"><i class="fa-solid ${icon}"></i></div>
    <div class="toast-body">
      <div class="toast-title">${escHtml(title)}</div>
      <div class="toast-msg">${escHtml(message)}</div>
    </div>
    <button class="toast-close" aria-label="Dismiss" onclick="dismissToast('${id}')">×</button>
    <div class="toast-progress"></div>
  `;
  container.appendChild(toast);

  // Trigger enter animation
  requestAnimationFrame(() => toast.classList.remove('toast-enter'));

  // Auto-dismiss after 8 s
  setTimeout(() => dismissToast(id), 8000);
}

function dismissToast(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.add('toast-exit');
  setTimeout(() => el.remove(), 350);
}

/* ─── Badge ─────────────────────────────────────────────────────────────── */

function setBadge(count) {
  let badge = document.getElementById('alerts-badge');
  if (!badge) return;
  if (count > 0) {
    badge.textContent = count > 99 ? '99+' : count;
    badge.classList.remove('hidden');
  } else {
    badge.classList.add('hidden');
  }
}

/* ─── Polling ────────────────────────────────────────────────────────────── */

async function pollAlerts() {
  try {
    const res = await fetch(`${API_ALERTS}/check`, { method: 'POST' });
    if (!res.ok) return;
    const data = await res.json();
    const firing = data.firing || [];

    setBadge(firing.length);

    for (const alert of firing) {
      if (!_firingIds.has(alert.id)) {
        _firingIds.add(alert.id);
        showToast(
          `🔥 Alert: ${alert.name}`,
          `${(alert.matched_count || 0).toLocaleString()} matching fire detections found`
        );
      }
    }

    // Remove IDs that are no longer firing so they can re-toast next time
    const firingSet = new Set(firing.map(a => a.id));
    for (const id of _firingIds) {
      if (!firingSet.has(id)) _firingIds.delete(id);
    }
  } catch (_) {
    // Network error — silently skip
  }
}

function startPolling() {
  if (_pollTimer) return; // already running
  pollAlerts(); // immediate first check
  _pollTimer = setInterval(pollAlerts, POLL_INTERVAL_MS);
}

function stopPolling() {
  if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
}

/* ─── API helpers ────────────────────────────────────────────────────────── */

async function apiAlerts(method, path = '', body = null) {
  const opts = {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  };
  const r = await fetch(API_ALERTS + path, opts);
  const json = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(json.error || 'Request failed');
  return json;
}

/* ─── Alert rule card HTML ──────────────────────────────────────────────── */

function conditionTags(rule) {
  const tags = [];
  if (rule.confidence)     tags.push(`Confidence: ${rule.confidence}`);
  if (rule.source)         tags.push(`Source: ${rule.source}`);
  if (rule.daynight)       tags.push(`Day/Night: ${rule.daynight}`);
  if (rule.frp_min != null)       tags.push(`FRP ≥ ${rule.frp_min}`);
  if (rule.frp_max != null)       tags.push(`FRP ≤ ${rule.frp_max}`);
  if (rule.brightness_min != null) tags.push(`Brightness ≥ ${rule.brightness_min}`);
  if (!tags.length) tags.push('All fires');
  return tags.map(t => `<span class="alert-tag">${escHtml(t)}</span>`).join('');
}

function ruleCardHtml(rule) {
  const active = rule.active !== false;
  return `
  <div class="alert-card ${active ? '' : 'alert-card--inactive'}" data-id="${escHtml(rule.id)}">
    <div class="alert-card-header">
      <div class="alert-card-title">
        <i class="fa-solid fa-bell${active ? '' : '-slash'} alert-card-icon"></i>
        <span class="alert-card-name">${escHtml(rule.name)}</span>
        <span class="alert-status-pill ${active ? 'pill-active' : 'pill-inactive'}">${active ? 'Active' : 'Paused'}</span>
      </div>
      <div class="alert-card-actions">
        <button class="alert-action-btn" title="${active ? 'Pause' : 'Enable'} alert" data-action="toggle" data-id="${escHtml(rule.id)}">
          <i class="fa-solid ${active ? 'fa-pause' : 'fa-play'}"></i>
        </button>
        <button class="alert-action-btn alert-action-btn--danger" title="Delete alert" data-action="delete" data-id="${escHtml(rule.id)}">
          <i class="fa-solid fa-trash"></i>
        </button>
      </div>
    </div>
    <div class="alert-conditions">${conditionTags(rule)}</div>
    ${rule.email ? `<div class="alert-email"><i class="fa-regular fa-envelope"></i> ${escHtml(rule.email)}</div>` : ''}
    <div class="alert-meta">Created ${new Date(rule.created_at).toLocaleDateString()}
      ${rule.last_fired ? ` · Last fired ${new Date(rule.last_fired).toLocaleString()}` : ''}
      ${rule.last_count ? ` · ${rule.last_count.toLocaleString()} records` : ''}
    </div>
  </div>`;
}

/* ─── Alerts view renderer ──────────────────────────────────────────────── */

async function alertsView() {
  const content = document.getElementById('content');
  content.innerHTML = '<div class="empty">Loading alerts…</div>';

  let rules = [];
  try {
    const data = await apiAlerts('GET');
    rules = data.rules || [];
  } catch (e) {
    content.innerHTML = `<div class="error">${escHtml(e.message)}</div>`;
    return;
  }

  const schema = window._alertSchema || {};
  const opts = window._alertOptions || {};

  const sourceOptions = (opts.source || []).map(v => `<option value="${escHtml(v)}">${escHtml(v)}</option>`).join('');
  const confOptions  = (opts.confidence || []).map(v => `<option value="${escHtml(v)}">${escHtml(v)}</option>`).join('');
  const dnOptions    = (opts.daynight || []).map(v => `<option value="${escHtml(v)}">${escHtml(v)}</option>`).join('');

  content.innerHTML = `
  <div class="alerts-page">
    <!-- Header -->
    <div class="alerts-page-header">
      <div>
        <div class="alerts-page-eyebrow">Monitoring</div>
        <h2 class="alerts-page-title">Fire Alert Rules</h2>
        <p class="alerts-page-sub">Create threshold-based alerts. The dashboard checks active rules every 60 seconds and notifies you in-browser — and optionally by email.</p>
      </div>
      <div class="alerts-poll-status" id="alerts-poll-status">
        <span class="status-dot"></span> Live monitoring active
      </div>
    </div>

    <!-- Create form -->
    <div class="alert-form-card" id="alert-form-card">
      <div class="alert-form-header">
        <i class="fa-solid fa-circle-plus"></i>
        <span>New Alert Rule</span>
      </div>
      <form id="alert-form" class="alert-form" autocomplete="off">
        <div class="alert-form-grid">
          <div class="filter-group af-full">
            <label for="af-name">Alert name <span class="required">*</span></label>
            <input id="af-name" name="name" type="text" placeholder="e.g. High-FRP fires in Asia" required>
          </div>
          <div class="filter-group">
            <label for="af-frp-min">Min FRP (MW)</label>
            <input id="af-frp-min" name="frp_min" type="number" min="0" step="0.1" placeholder="e.g. 100">
          </div>
          <div class="filter-group">
            <label for="af-frp-max">Max FRP (MW)</label>
            <input id="af-frp-max" name="frp_max" type="number" min="0" step="0.1" placeholder="optional">
          </div>
          <div class="filter-group">
            <label for="af-brightness">Min Brightness (K)</label>
            <input id="af-brightness" name="brightness_min" type="number" min="0" step="1" placeholder="e.g. 320">
          </div>
          <div class="filter-group">
            <label for="af-confidence">Confidence level</label>
            <select id="af-confidence" name="confidence">
              <option value="">Any</option>
              ${confOptions}
            </select>
          </div>
          <div class="filter-group">
            <label for="af-source">Satellite source</label>
            <select id="af-source" name="source">
              <option value="">Any</option>
              ${sourceOptions}
            </select>
          </div>
          <div class="filter-group">
            <label for="af-daynight">Day / Night</label>
            <select id="af-daynight" name="daynight">
              <option value="">Any</option>
              ${dnOptions}
            </select>
          </div>
          <div class="filter-group af-full">
            <label for="af-email">Notify email (optional)</label>
            <input id="af-email" name="email" type="email" placeholder="you@example.com — requires SMTP config in .env">
          </div>
        </div>
        <div class="alert-form-footer">
          <div id="af-error" class="af-error hidden"></div>
          <button type="submit" class="train-btn" id="af-submit">
            <span class="btn-icon"><i class="fa-solid fa-plus"></i></span>
            <span class="spinner"></span>
            Create Alert
          </button>
        </div>
      </form>
    </div>

    <!-- Rules list -->
    <div class="alert-list-header">
      <span>${rules.length} rule${rules.length !== 1 ? 's' : ''}</span>
      <button class="button secondary" id="check-now-btn" style="padding:8px 16px;font-size:13px;">
        <i class="fa-solid fa-rotate"></i> Check now
      </button>
    </div>
    <div id="alert-rules-list" class="alert-rules-list">
      ${rules.length
        ? rules.map(ruleCardHtml).join('')
        : '<div class="empty" style="margin-top:12px">No alert rules yet. Create one above.</div>'}
    </div>
  </div>`;

  attachAlertFormHandler();
  attachRuleListHandlers();
  document.getElementById('check-now-btn').addEventListener('click', async () => {
    await pollAlerts();
    await reloadRulesList();
  });
}

/* ─── Form submit ────────────────────────────────────────────────────────── */

function attachAlertFormHandler() {
  const form = document.getElementById('alert-form');
  if (!form) return;
  form.addEventListener('submit', async e => {
    e.preventDefault();
    const btn = document.getElementById('af-submit');
    const errEl = document.getElementById('af-error');
    errEl.classList.add('hidden');
    btn.classList.add('loading');
    btn.disabled = true;

    const fd = new FormData(form);
    const body = {};
    for (const [k, v] of fd.entries()) { if (v !== '') body[k] = v; }

    try {
      await apiAlerts('POST', '', body);
      form.reset();
      await reloadRulesList();
      // Immediate check after creating a rule
      await pollAlerts();
    } catch (err) {
      errEl.textContent = err.message;
      errEl.classList.remove('hidden');
    } finally {
      btn.classList.remove('loading');
      btn.disabled = false;
    }
  });
}

/* ─── Rule list actions ──────────────────────────────────────────────────── */

function attachRuleListHandlers() {
  const list = document.getElementById('alert-rules-list');
  if (!list) return;
  list.addEventListener('click', async e => {
    const btn = e.target.closest('[data-action]');
    if (!btn) return;
    const { action, id } = btn.dataset;
    btn.disabled = true;
    try {
      if (action === 'delete') {
        if (!confirm('Delete this alert rule?')) return;
        await apiAlerts('DELETE', `/${id}`);
      } else if (action === 'toggle') {
        await apiAlerts('PATCH', `/${id}/toggle`);
      }
      await reloadRulesList();
      await pollAlerts();
    } catch (err) {
      showToast('Error', err.message, 'error');
    } finally {
      btn.disabled = false;
    }
  });
}

async function reloadRulesList() {
  const data = await apiAlerts('GET');
  const rules = data.rules || [];
  const list = document.getElementById('alert-rules-list');
  if (!list) return;
  list.innerHTML = rules.length
    ? rules.map(ruleCardHtml).join('')
    : '<div class="empty" style="margin-top:12px">No alert rules yet. Create one above.</div>';
  const hdr = document.querySelector('.alert-list-header span');
  if (hdr) hdr.textContent = `${rules.length} rule${rules.length !== 1 ? 's' : ''}`;
  attachRuleListHandlers();
}

/* ─── Utilities ──────────────────────────────────────────────────────────── */

function escHtml(v) {
  return String(v ?? '').replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[c]));
}

/* ─── Bootstrap ──────────────────────────────────────────────────────────── */

// Expose for dashboard.js viewFns dispatch
window.alertsView = alertsView;
window.startAlertsPolling = startPolling;
window.stopAlertsPolling  = stopPolling;

// Store schema/options set by dashboard.js boot so the form can populate dropdowns
window.setAlertContext = (schema, options) => {
  window._alertSchema  = schema;
  window._alertOptions = options;
};

// Start polling immediately when script loads
startPolling();

'use strict';

const state = {
  apiBase: '',
  payload: null,
  uploadedFile: null,
  uploadedCsvText: null,
  activeTierFilter: 'ALL',
  activeGradeFilter: 'ALL',
  searchQuery: '',
  interventions: {},
  sortCol: 5,
  sortDir: 'descending',
};

const els = {
  // Auth session indicators on dashboard header
  userIdDisplay: document.getElementById('user-id-display'),
  userRoleDisplay: document.getElementById('user-role-display'),
  userAvatarInitials: document.getElementById('user-avatar-initials'),
  signOutBtn: document.getElementById('sign-out-btn'),

  statusPill: document.getElementById('pipeline-status-pill'),
  statusText: document.getElementById('pipeline-status-text'),
  imputationSelect: document.getElementById('imputation-select'),
  dedupSelect: document.getElementById('dedup-select'),
  dropzone: document.getElementById('csv-dropzone'),
  fileInput: document.getElementById('csv-file-input'),
  dropzoneTitle: document.getElementById('dropzone-title'),
  runBtn: document.getElementById('run-pipeline-btn'),
  downloadTemplateBtn: document.getElementById('download-template-btn'),
  downloadCleanBtn: document.getElementById('download-clean-btn'),
  activeSourceBadge: document.getElementById('active-source-badge'),
  modelBadgePill: document.getElementById('model-badge-pill'),
  errorBanner: document.getElementById('error-banner'),

  // KPIs
  kpiHygiene: document.getElementById('kpi-hygiene'),
  kpiHygieneSub: document.getElementById('kpi-hygiene-sub'),
  kpiFlagged: document.getElementById('kpi-flagged'),
  kpiFlaggedSub: document.getElementById('kpi-flagged-sub'),
  kpiRetention: document.getElementById('kpi-retention'),
  kpiRetentionSub: document.getElementById('kpi-retention-sub'),
  kpiFixes: document.getElementById('kpi-fixes'),
  kpiFixesSub: document.getElementById('kpi-fixes-sub'),

  // Overview charts
  totalStudentsPill: document.getElementById('total-students-pill'),
  riskDonutContainer: document.getElementById('risk-donut-container'),
  qualityDeltaPill: document.getElementById('quality-delta-pill'),
  hygieneBarsContainer: document.getElementById('hygiene-bars-container'),
  scatterChartContainer: document.getElementById('scatter-chart-container'),
  gradeBreakdownContainer: document.getElementById('grade-breakdown-container'),
  featureImportanceContainer: document.getElementById('feature-importance-container'),
  cvMetricsRow: document.getElementById('cv-metrics-row'),
  modelSubtitle: document.getElementById('model-subtitle'),

  // Cleaning tab
  cleaningCounters: document.getElementById('cleaning-counters'),
  dupCountBadge: document.getElementById('dup-count-badge'),
  scaleCountBadge: document.getElementById('scale-count-badge'),
  imputedCountBadge: document.getElementById('imputed-count-badge'),
  feeFixBadge: document.getElementById('fee-fix-badge'),
  duplicatesTbody: document.getElementById('duplicates-tbody'),
  scalesTbody: document.getElementById('scales-tbody'),
  imputedTbody: document.getElementById('imputed-tbody'),
  feeFixesTbody: document.getElementById('fee-fixes-tbody'),
  diffTbody: document.getElementById('diff-tbody'),
  onlyModifiedToggle: document.getElementById('only-modified-diff-toggle'),

  // Directory tab
  studentSearch: document.getElementById('student-search'),
  gradeFilter: document.getElementById('grade-filter'),
  studentsTable: document.getElementById('students-table'),
  studentsTbody: document.getElementById('students-tbody'),
  tableSortStatus: document.getElementById('table-sort-status'),

  // Sandbox tab
  singleForm: document.getElementById('single-record-form'),
  sandboxResultBody: document.getElementById('sandbox-result-body'),
  sandboxResetBtn: document.getElementById('sandbox-reset-btn'),

  // Modal
  modal: document.getElementById('student-modal'),
  closeModalBtn: document.getElementById('close-modal-btn'),
  modalStudentId: document.getElementById('modal-student-id'),
  modalStudentTitle: document.getElementById('modal-student-title'),
  modalStudentGrade: document.getElementById('modal-student-grade'),
  modalBody: document.getElementById('modal-body'),
};

function esc(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function formatMoney(amount, currencyHint) {
  const sym = (currencyHint || '').includes('USD') ? '$' : '₹';
  return `${sym}${Math.round(Number(amount || 0)).toLocaleString('en-IN')}`;
}

function tierBadgeHtml(tier) {
  const map = {
    'High Risk': { cls: 'badge-high', icon: '▲' },
    'Medium Risk': { cls: 'badge-medium', icon: '◆' },
    'Low Risk': { cls: 'badge-low', icon: '✓' },
  };
  const info = map[tier] || { cls: 'badge-neutral', icon: '•' };
  return `<span class="badge ${info.cls}"><span aria-hidden="true">${info.icon}</span> ${esc(tier)}</span>`;
}

function setLoading(isLoading, label) {
  els.statusPill.classList.toggle('is-loading', isLoading);
  els.statusText.textContent = label;
  els.runBtn.disabled = isLoading || !state.uploadedFile;
}

function showError(message) {
  if (!message) {
    els.errorBanner.hidden = true;
    els.errorBanner.textContent = '';
    return;
  }
  els.errorBanner.hidden = false;
  els.errorBanner.textContent = message;
}

async function parseApiResponse(response) {
  const rawText = await response.text();
  if (!rawText || !rawText.trim()) {
    if (response.status === 404 || response.status === 405) {
      throw new Error(
        `FastAPI backend route unavailable (HTTP ${response.status}). Start the server with "python -m uvicorn app.main:app --host 127.0.0.1 --port 8000" and open http://127.0.0.1:8000.`
      );
    }
    throw new Error(`Server returned an empty response (HTTP ${response.status}).`);
  }

  let data;
  try {
    data = JSON.parse(rawText);
  } catch {
    throw new Error(
      `Server returned a non-JSON response (HTTP ${response.status}). Make sure the FastAPI server is running at http://127.0.0.1:8000.`
    );
  }

  if (!response.ok) {
    throw new Error(data.detail || `Request failed with HTTP ${response.status}.`);
  }
  return data;
}

/* Accessible Tabs */
function initTabs() {
  const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
  const panels = Array.from(document.querySelectorAll('[role="tabpanel"]'));

  function activateTab(targetTab) {
    for (const t of tabs) {
      t.setAttribute('aria-selected', String(t === targetTab));
    }
    for (const p of panels) {
      p.hidden = p.id !== targetTab.dataset.panel;
    }
  }

  for (const tab of tabs) {
    tab.addEventListener('click', () => activateTab(tab));
    tab.addEventListener('keydown', (e) => {
      const idx = tabs.indexOf(tab);
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        const next = tabs[(idx + 1) % tabs.length];
        next.focus();
        activateTab(next);
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        const prev = tabs[(idx - 1 + tabs.length) % tabs.length];
        prev.focus();
        activateTab(prev);
      }
    });
  }
}

/* Initial Empty State & Model Health */
function renderEmptyDashboard() {
  const emptyPrompt = (text) => `<div class="sandbox-result-empty">${esc(text)}</div>`;

  els.riskDonutContainer.innerHTML = emptyPrompt('Upload a school CSV file above to calculate High, Medium, and Low dropout risk tier distribution.');
  els.hygieneBarsContainer.innerHTML = emptyPrompt('Upload a CSV file to compare raw data defects against Python-cleaned outputs.');
  els.scatterChartContainer.innerHTML = emptyPrompt('Upload a CSV file to plot attendance percentage vs. standardized academic marks.');
  els.gradeBreakdownContainer.innerHTML = emptyPrompt('Upload a CSV file to inspect risk distribution across grade cohorts.');

  els.cleaningCounters.innerHTML = '<span class="badge badge-neutral">Awaiting CSV upload</span>';
  els.duplicatesTbody.innerHTML = '<tr><td colspan="4">Upload a CSV file to inspect detected duplicates.</td></tr>';
  els.scalesTbody.innerHTML = '<tr><td colspan="4">Upload a CSV file to inspect grading scale conversions.</td></tr>';
  els.imputedTbody.innerHTML = '<tr><td colspan="5">Upload a CSV file to inspect imputed missing cells.</td></tr>';
  els.feeFixesTbody.innerHTML = '<tr><td colspan="4">Upload a CSV file to inspect currency and fee status fixes.</td></tr>';
  els.diffTbody.innerHTML = '<tr><td colspan="6">Upload a CSV file to inspect cell-by-cell transformations.</td></tr>';
  els.studentsTbody.innerHTML = '<tr><td colspan="8">No student cohort uploaded yet. Upload a CSV above or use the Manual Student Predictor tab.</td></tr>';
}

function renderModelSection(modelMetrics, featureImportances) {
  if (!modelMetrics) return;
  els.modelBadgePill.textContent = `Ensemble ROC-AUC ${modelMetrics.roc_auc}`;
  els.modelSubtitle.textContent = `${modelMetrics.model_name} • ${modelMetrics.cv_folds}-fold CV`;

  if (Array.isArray(featureImportances)) {
    els.featureImportanceContainer.innerHTML = featureImportances.slice(0, 5).map(f => `
      <div>
        <div class="bar-row-meta">
          <span>${esc(f.label)}</span>
          <span class="tabular">${f.importance_pct}%</span>
        </div>
        <div class="bar-track"><div class="bar-fill" style="inline-size: ${Math.max(5, Math.min(100, f.importance_pct * 2.2))}%"></div></div>
      </div>
    `).join('');
  }

  els.cvMetricsRow.innerHTML = `
    <span class="cv-chip">Accuracy: <strong>${modelMetrics.accuracy_pct}%</strong></span>
    <span class="cv-chip">ROC-AUC: <strong>${modelMetrics.roc_auc}</strong></span>
    <span class="cv-chip">Precision: <strong>${modelMetrics.precision_pct}%</strong></span>
    <span class="cv-chip">Recall: <strong>${modelMetrics.recall_pct}%</strong></span>
  `;
}

async function loadInitialHealth() {
  const candidates = window.location.protocol === 'file:'
    ? ['http://127.0.0.1:8000']
    : ['', 'http://127.0.0.1:8000'];

  for (const base of candidates) {
    try {
      const res = await fetch(`${base}/api/health`);
      const data = await parseApiResponse(res);
      if (data && data.status === 'ok') {
        state.apiBase = base;
        renderModelSection(data.model, data.feature_importances);
        return;
      }
    } catch {
      // Try next candidate
    }
  }

  els.statusText.textContent = 'Backend offline';
  showError(
    'Python FastAPI backend is not reachable. Start it in terminal with "python -m uvicorn app.main:app --host 127.0.0.1 --port 8000" and open http://127.0.0.1:8000.'
  );
}

/* API Upload Pipeline Execution */
async function runUploadAnalysis() {
  if (!state.uploadedFile || state.uploadedCsvText === null) {
    showError('Please choose or drop a CSV file first.');
    return;
  }

  showError('');
  setLoading(true, 'Running Python pipeline...');

  try {
    const fd = new FormData();
    const csvBlob = new Blob([state.uploadedCsvText], { type: 'text/csv;charset=utf-8' });
    fd.append('file', csvBlob, state.uploadedFile.name || 'upload.csv');
    fd.append('imputation_method', els.imputationSelect.value);
    fd.append('dedup_mode', els.dedupSelect.value);

    const response = await fetch(`${state.apiBase}/api/analyze/upload`, { method: 'POST', body: fd });
    const data = await parseApiResponse(response);

    state.payload = data;
    renderAll();
    setLoading(false, `Cleaned ${data.cleaning_summary.clean_rows} students`);
  } catch (err) {
    setLoading(false, 'Pipeline error');
    showError(err.message || 'Unable to run Python cleaning pipeline.');
  }
}

/* Renderers */
function renderAll() {
  const data = state.payload;
  if (!data) return;

  els.downloadCleanBtn.disabled = !data.cleaned_csv;
  els.activeSourceBadge.textContent = `Active Upload: ${state.uploadedFile?.name || 'Uploaded CSV'}`;

  renderKpis(data);
  renderOverviewCharts(data);
  renderCleaningTab(data);
  populateGradeFilter(data);
  renderStudentsDirectory();
}

function renderKpis(data) {
  const s = data.cleaning_summary;
  const c = data.cohort_analytics;

  els.kpiHygiene.textContent = `${s.data_quality_before_pct}% → ${s.data_quality_after_pct}%`;
  els.kpiHygieneSub.textContent = `${s.duplicates_removed_count} duplicates dropped • ${s.missing_values_before.total} nulls fixed`;

  const flagged = c.at_risk_flagged;
  els.kpiFlagged.textContent = `${flagged} (${c.tier_counts['High Risk']} High)`;
  els.kpiFlaggedSub.textContent = `${c.high_risk_pct}% critical risk • ${c.overdue_students_count} overdue fee accounts`;

  els.kpiRetention.textContent = `${c.retention_forecast_pct}%`;
  els.kpiRetentionSub.textContent = `${c.total_students} unique cleaned students (from ${s.raw_rows} raw rows)`;

  const totalTransforms = s.scale_conversions_count + s.cells_imputed_count + s.fee_fixes_count;
  els.kpiFixes.textContent = `${totalTransforms} fixes`;
  els.kpiFixesSub.textContent = `${s.scale_conversions_count} scale unified • ${s.cells_imputed_count} imputed (${s.imputation_method.replace('_', ' ')})`;
}

function renderOverviewCharts(data) {
  const c = data.cohort_analytics;
  const s = data.cleaning_summary;

  els.totalStudentsPill.textContent = `${c.total_students} students`;
  const lift = Math.max(0, (s.data_quality_after_pct - s.data_quality_before_pct)).toFixed(1);
  els.qualityDeltaPill.textContent = `+${lift}% data health`;

  // 1. Donut Chart SVG
  const total = Math.max(1, c.total_students);
  const high = c.tier_counts['High Risk'];
  const med = c.tier_counts['Medium Risk'];
  const low = c.tier_counts['Low Risk'];

  const r = 54;
  const circ = 2 * Math.PI * r;
  const highLen = (high / total) * circ;
  const medLen = (med / total) * circ;
  const lowLen = (low / total) * circ;

  els.riskDonutContainer.innerHTML = `
    <svg viewBox="0 0 148 148" width="148" height="148" role="img" aria-label="Risk tiers: ${high} High, ${med} Medium, ${low} Low">
      <circle cx="74" cy="74" r="${r}" fill="none" stroke="var(--bg-subtle)" stroke-width="18" />
      <circle cx="74" cy="74" r="${r}" fill="none" stroke="#10b981" stroke-width="18"
        stroke-dasharray="${lowLen} ${circ}" stroke-dashoffset="0" transform="rotate(-90 74 74)" />
      <circle cx="74" cy="74" r="${r}" fill="none" stroke="#f59e0b" stroke-width="18"
        stroke-dasharray="${medLen} ${circ}" stroke-dashoffset="${-lowLen}" transform="rotate(-90 74 74)" />
      <circle cx="74" cy="74" r="${r}" fill="none" stroke="#ef4444" stroke-width="18"
        stroke-dasharray="${highLen} ${circ}" stroke-dashoffset="${-(lowLen + medLen)}" transform="rotate(-90 74 74)" />
      <text x="74" y="70" text-anchor="middle" font-size="22" font-weight="800" fill="var(--ink-primary)">${c.at_risk_flagged}</text>
      <text x="74" y="88" text-anchor="middle" font-size="10.5" font-weight="600" fill="var(--ink-muted)">AT-RISK</text>
    </svg>
    <div class="donut-legend">
      <div class="donut-legend-row">
        <span>${tierBadgeHtml('High Risk')}</span>
        <strong class="tabular">${high} (${((high / total) * 100).toFixed(0)}%)</strong>
      </div>
      <div class="donut-legend-row">
        <span>${tierBadgeHtml('Medium Risk')}</span>
        <strong class="tabular">${med} (${((med / total) * 100).toFixed(0)}%)</strong>
      </div>
      <div class="donut-legend-row">
        <span>${tierBadgeHtml('Low Risk')}</span>
        <strong class="tabular">${low} (${((low / total) * 100).toFixed(0)}%)</strong>
      </div>
    </div>
  `;

  // 2. Data Hygiene Bars
  const hygieneItems = [
    {
      label: 'Raw vs. Cleaned Quality Index',
      value: `${s.data_quality_before_pct}% → ${s.data_quality_after_pct}%`,
      pct: s.data_quality_after_pct,
      cls: 'is-emerald',
    },
    {
      label: 'Missing Values Imputed (Attendance / Marks / Fees)',
      value: `${s.missing_values_before.total} / ${s.missing_values_before.total} fixed`,
      pct: 100,
      cls: 'is-amber',
    },
    {
      label: 'Duplicate Records Purged',
      value: `${s.duplicates_removed_count} of ${s.raw_rows} raw rows`,
      pct: Math.min(100, (s.duplicates_removed_count / Math.max(1, s.raw_rows)) * 100 * 3),
      cls: 'is-rose',
    },
    {
      label: 'Heterogeneous Grading Scales Normalized',
      value: `${s.scale_conversions_count} standardized to 0–100%`,
      pct: Math.min(100, (s.scale_conversions_count / Math.max(1, s.clean_rows)) * 100),
      cls: '',
    },
  ];
  els.hygieneBarsContainer.innerHTML = hygieneItems.map(item => `
    <div>
      <div class="bar-row-meta">
        <span>${esc(item.label)}</span>
        <span class="tabular">${esc(item.value)}</span>
      </div>
      <div class="bar-track"><div class="bar-fill ${item.cls}" style="inline-size: ${Math.max(4, item.pct)}%"></div></div>
    </div>
  `).join('');

  // 3. Scatter Quadrant SVG (Attendance vs Marks)
  const w = 780;
  const h = 270;
  const padL = 46;
  const padR = 20;
  const padT = 18;
  const padB = 34;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const toX = (att) => padL + (Math.min(100, Math.max(20, att)) - 20) / 80 * plotW;
  const toY = (marks) => padT + (1 - (Math.min(100, Math.max(15, marks)) - 15) / 85) * plotH;

  const dots = data.students.map((st) => {
    const cx = toX(st.attendance_pct).toFixed(1);
    const cy = toY(st.marks_pct).toFixed(1);
    const fill = st.risk_tier === 'High Risk' ? '#ef4444' : (st.risk_tier === 'Medium Risk' ? '#f59e0b' : '#10b981');
    return `
      <circle
        class="scatter-dot"
        data-student-id="${esc(st.student_id)}"
        cx="${cx}" cy="${cy}" r="6.5"
        fill="${fill}" stroke="var(--bg-elevated)" stroke-width="1.75"
        tabindex="0" role="button"
        aria-label="${esc(st.student_name)} (${esc(st.student_id)}): ${st.attendance_pct}% attendance, ${st.marks_pct}% marks, ${st.dropout_probability_pct}% risk"
      >
        <title>${esc(st.student_name)} (${esc(st.student_id)})&#10;Attendance: ${st.attendance_pct}% | Marks: ${st.marks_pct}%&#10;Dropout Risk: ${st.dropout_probability_pct}% (${st.risk_tier})</title>
      </circle>
    `;
  }).join('');

  const x72 = toX(72).toFixed(1);
  const y52 = toY(52).toFixed(1);

  els.scatterChartContainer.innerHTML = `
    <svg viewBox="0 0 ${w} ${h}" width="100%" style="inline-size: 100%; max-inline-size: 100%; block-size: auto; max-block-size: 290px;" role="img" aria-label="Scatter chart comparing attendance percentage and cleaned academic marks">
      <rect x="${padL}" y="${padT}" width="${x72 - padL}" height="${plotH}" fill="rgb(239 68 68 / 0.04)" />
      <line x1="${padL}" y1="${y52}" x2="${w - padR}" y2="${y52}" stroke="var(--border-strong)" stroke-dasharray="4 4" />
      <line x1="${x72}" y1="${padT}" x2="${x72}" y2="${h - padB}" stroke="var(--border-strong)" stroke-dasharray="4 4" />
      <text x="${padL + 8}" y="${padT + 14}" font-size="10.5" font-weight="600" fill="var(--risk-high)">CRITICAL ZONE (&lt;72% Att)</text>
      <text x="${w - padR - 8}" y="${padT + 14}" text-anchor="end" font-size="10.5" font-weight="600" fill="var(--risk-low)">ON-TRACK COHORT</text>

      <line x1="${padL}" y1="${h - padB}" x2="${w - padR}" y2="${h - padB}" stroke="var(--border-strong)" />
      <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${h - padB}" stroke="var(--border-strong)" />

      ${[20, 40, 60, 80, 100].map(v => `
        <text x="${toX(v)}" y="${h - 12}" text-anchor="middle" font-size="10" fill="var(--ink-muted)">${v}%</text>
      `).join('')}
      ${[20, 40, 60, 80, 100].map(v => `
        <text x="${padL - 8}" y="${toY(v) + 3}" text-anchor="end" font-size="10" fill="var(--ink-muted)">${v}%</text>
      `).join('')}
      <text x="${padL + plotW / 2}" y="${h - 1}" text-anchor="middle" font-size="10.5" font-weight="600" fill="var(--ink-secondary)">Cleaned Attendance %</text>
      ${dots}
    </svg>
  `;

  els.scatterChartContainer.querySelectorAll('.scatter-dot').forEach((dot) => {
    const open = () => openStudentModal(dot.dataset.studentId);
    dot.addEventListener('click', open);
    dot.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        open();
      }
    });
  });

  // 4. Grade Breakdown Bars
  els.gradeBreakdownContainer.innerHTML = c.grade_breakdown.map(g => {
    const cls = g.avg_risk_pct >= 50 ? 'is-rose' : (g.avg_risk_pct >= 30 ? 'is-amber' : 'is-emerald');
    return `
      <div>
        <div class="bar-row-meta">
          <span>${esc(g.grade_level)} <small style="color:var(--ink-muted)">(${g.total} students • ${g.high} high risk)</small></span>
          <span class="tabular">Avg Risk ${g.avg_risk_pct}%</span>
        </div>
        <div class="bar-track"><div class="bar-fill ${cls}" style="inline-size: ${Math.max(4, g.avg_risk_pct)}%"></div></div>
      </div>
    `;
  }).join('');

  renderModelSection(data.model_metrics, data.feature_importances);
}

function renderCleaningTab(data) {
  const s = data.cleaning_summary;
  const curr = data.cohort_analytics.currency_hint;

  els.cleaningCounters.innerHTML = `
    <span class="badge badge-indigo">Raw Rows: ${s.raw_rows} → Clean: ${s.clean_rows}</span>
    <span class="badge badge-rose">Duplicates Removed: ${s.duplicates_removed_count}</span>
    <span class="badge badge-amber">Missing Cells Imputed: ${s.cells_imputed_count}</span>
    <span class="badge badge-teal">Scales Standardized: ${s.scale_conversions_count}</span>
  `;

  els.dupCountBadge.textContent = `${data.duplicates_removed.length} removed`;
  els.scaleCountBadge.textContent = `${data.scale_conversions.length} converted`;
  els.imputedCountBadge.textContent = `${data.imputed_cells.length} cells`;
  els.feeFixBadge.textContent = `${data.fee_fixes.length} fixed`;

  els.duplicatesTbody.innerHTML = data.duplicates_removed.length
    ? data.duplicates_removed.map(d => `
      <tr>
        <td class="tabular">${esc(d.student_id)}</td>
        <td>${esc(d.student_name)}</td>
        <td class="tabular">Drop #${d.dropped_row} → Keep #${d.kept_row}</td>
        <td>${esc(d.reason)}</td>
      </tr>
    `).join('')
    : '<tr><td colspan="4">No duplicate records found in this dataset.</td></tr>';

  els.scalesTbody.innerHTML = data.scale_conversions.length
    ? data.scale_conversions.map(sc => `
      <tr>
        <td><strong>${esc(sc.student_id)}</strong> ${esc(sc.student_name)}</td>
        <td class="tabular raw-strike">${esc(sc.raw_marks)}</td>
        <td><span class="badge badge-indigo">${esc(sc.detected_scale)}</span></td>
        <td class="tabular"><strong>${sc.normalized_pct}%</strong> (${sc.normalized_letter} / ${sc.normalized_gpa4} GPA)</td>
      </tr>
    `).join('')
    : '<tr><td colspan="4">All marks were already on 0–100% scale.</td></tr>';

  els.imputedTbody.innerHTML = data.imputed_cells.length
    ? data.imputed_cells.map(im => `
      <tr>
        <td><strong>${esc(im.student_id)}</strong> ${esc(im.student_name)}</td>
        <td class="tabular">${esc(im.column)}</td>
        <td class="tabular raw-strike">${esc(im.raw_value)}</td>
        <td class="tabular"><strong>${im.column === 'fee_due' ? formatMoney(im.imputed_value, curr) : `${im.imputed_value}%`}</strong></td>
        <td>${esc(im.method)}</td>
      </tr>
    `).join('')
    : '<tr><td colspan="5">No missing values required imputation.</td></tr>';

  els.feeFixesTbody.innerHTML = data.fee_fixes.length
    ? data.fee_fixes.map(ff => `
      <tr>
        <td><strong>${esc(ff.student_id)}</strong> ${esc(ff.student_name)}</td>
        <td class="tabular">${esc(ff.field)}</td>
        <td class="tabular"><span class="raw-strike">${esc(ff.raw_value)}</span> → <strong>${esc(ff.cleaned_value)}</strong></td>
        <td>${esc(ff.note)}</td>
      </tr>
    `).join('')
    : '<tr><td colspan="4">No currency or fee status conflicts detected.</td></tr>';

  renderDiffTable();
}

function renderDiffTable() {
  if (!state.payload) return;
  const onlyMod = els.onlyModifiedToggle.checked;
  const curr = state.payload.cohort_analytics.currency_hint;
  const rows = onlyMod
    ? state.payload.row_lineages.filter(r => r.was_modified)
    : state.payload.row_lineages;

  els.diffTbody.innerHTML = rows.length
    ? rows.map(r => {
      const attChanged = r.attendance_imputed || parseFloat(String(r.raw_attendance).replace('%', '')) !== r.clean_attendance_pct;
      const feeNum = Number(String(r.raw_fee_due).replace(/[^0-9.-]/g, ''));
      const feeChanged = r.fee_imputed || r.raw_fee_due === '∅' || Number.isNaN(feeNum) || feeNum !== r.clean_fee_due || r.raw_fee_status !== r.clean_fee_status;
      return `
      <tr>
        <td class="tabular"><strong>${esc(r.student_id)}</strong></td>
        <td>${esc(r.student_name)}<br><small style="color:var(--ink-muted)">${esc(r.grade_level)}</small></td>
        <td>
          <div class="diff-pair">
            ${attChanged ? `<span class="raw-strike">${esc(r.raw_attendance)}</span><span>→</span>` : ''}
            <span class="clean-val">${r.clean_attendance_pct}%</span>
            ${r.attendance_imputed ? '<span class="cell-flag">Imputed</span>' : ''}
          </div>
        </td>
        <td>
          <div class="diff-pair">
            ${r.detected_scale !== 'Percentage (0-100)' ? `<span class="raw-strike">${esc(r.raw_marks)}</span><span>→</span>` : ''}
            <span class="clean-val">${r.clean_marks_pct}% (${esc(r.clean_letter)})</span>
            ${r.marks_imputed ? '<span class="cell-flag">Imputed</span>' : `<span class="cell-flag cell-flag-scale">${esc(r.detected_scale)}</span>`}
          </div>
        </td>
        <td>
          <div class="diff-pair">
            ${feeChanged ? `<span class="raw-strike">${esc(r.raw_fee_due)} (${esc(r.raw_fee_status)})</span><span>→</span>` : ''}
            <span class="clean-val">${formatMoney(r.clean_fee_due, curr)} (${esc(r.clean_fee_status)})</span>
            ${r.fee_imputed ? '<span class="cell-flag">Imputed</span>' : ''}
          </div>
        </td>
        <td>
          <div class="tag-list">
            ${r.transformations.length ? r.transformations.map(t => `<span class="mini-tag">${esc(t)}</span>`).join('') : '<span class="mini-tag">Clean as received</span>'}
          </div>
        </td>
      </tr>
    `;
    }).join('')
    : '<tr><td colspan="6">No modified rows match the filter.</td></tr>';
}

/* Directory Table */
function populateGradeFilter(data) {
  const grades = [...new Set(data.students.map(s => s.grade_level))].sort();
  const prev = els.gradeFilter.value;
  els.gradeFilter.innerHTML = `<option value="ALL">All Grades (${data.students.length})</option>` +
    grades.map(g => `<option value="${esc(g)}">${esc(g)}</option>`).join('');
  if (grades.includes(prev)) els.gradeFilter.value = prev;
}

function renderStudentsDirectory() {
  if (!state.payload) return;
  const curr = state.payload.cohort_analytics.currency_hint;
  const q = state.searchQuery.trim().toLowerCase();

  let list = state.payload.students.filter(s => {
    if (state.activeTierFilter !== 'ALL' && s.risk_tier !== state.activeTierFilter) return false;
    if (state.activeGradeFilter !== 'ALL' && s.grade_level !== state.activeGradeFilter) return false;
    if (q) {
      const hay = `${s.student_id} ${s.student_name} ${s.grade_level} ${s.primary_driver}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });

  const factor = state.sortDir === 'ascending' ? 1 : -1;
  list.sort((a, b) => {
    if (state.sortCol === 0) return factor * a.student_name.localeCompare(b.student_name);
    if (state.sortCol === 1) return factor * a.grade_level.localeCompare(b.grade_level);
    if (state.sortCol === 2) return factor * (a.attendance_pct - b.attendance_pct);
    if (state.sortCol === 3) return factor * (a.marks_pct - b.marks_pct);
    if (state.sortCol === 4) return factor * (a.fee_due - b.fee_due);
    return factor * (a.dropout_probability_pct - b.dropout_probability_pct);
  });

  els.studentsTbody.innerHTML = list.length
    ? list.map(s => {
      const actionStatus = state.interventions[s.student_id] || (s.risk_tier === 'Low Risk' ? 'Monitoring' : 'Unassigned');
      return `
        <tr>
          <td>
            <strong>${esc(s.student_name)}</strong><br>
            <small class="tabular" style="color:var(--ink-muted)">${esc(s.student_id)}</small>
          </td>
          <td>${esc(s.grade_level)}</td>
          <td class="tabular">
            ${s.attendance_pct}%
            ${s.attendance_imputed ? '<span class="cell-flag">Imputed</span>' : ''}
          </td>
          <td class="tabular">
            <strong>${s.marks_pct}%</strong> <small>(${esc(s.letter_grade)})</small>
            ${s.marks_imputed ? '<span class="cell-flag">Imputed</span>' : ''}
          </td>
          <td class="tabular">
            ${formatMoney(s.fee_due, curr)}<br>
            <small style="color:var(--ink-muted)">${esc(s.fee_status)}</small>
          </td>
          <td class="tabular">
            ${tierBadgeHtml(s.risk_tier)}<br>
            <strong style="font-size:0.82rem">${s.dropout_probability_pct}% prob</strong>
          </td>
          <td>
            <div style="max-inline-size: 270px; font-size:0.8rem">${esc(s.primary_driver)}</div>
            <small class="mini-tag" style="margin-block-start:0.25rem; display:inline-block">Counselor: ${esc(actionStatus)}</small>
          </td>
          <td>
            <button type="button" class="btn btn-secondary btn-sm inspect-student-btn" data-student-id="${esc(s.student_id)}">
              Inspect
            </button>
          </td>
        </tr>
      `;
    }).join('')
    : '<tr><td colspan="8">No students match the selected filters.</td></tr>';

  els.studentsTbody.querySelectorAll('.inspect-student-btn').forEach(btn => {
    btn.addEventListener('click', () => openStudentModal(btn.dataset.studentId));
  });
}

/* Sortable Table Headers */
function initSortableHeaders() {
  if (!els.studentsTable) return;
  const headers = Array.from(els.studentsTable.querySelectorAll('thead th'));
  headers.forEach((th) => {
    const btn = th.querySelector('.sort-btn');
    if (!btn) return;
    btn.addEventListener('click', () => {
      const col = Number(btn.dataset.col);
      const nextDir = th.getAttribute('aria-sort') === 'ascending' ? 'descending' : 'ascending';
      headers.forEach(h => h.removeAttribute('aria-sort'));
      th.setAttribute('aria-sort', nextDir);
      state.sortCol = col;
      state.sortDir = nextDir;
      renderStudentsDirectory();
      if (els.tableSortStatus) {
        els.tableSortStatus.textContent = `Sorted by ${btn.textContent.trim()}, ${nextDir}`;
      }
    });
  });
}

/* Native Modal Student Inspector */
function openStudentModal(studentId) {
  if (!state.payload) return;
  const student = state.payload.students.find(s => s.student_id === studentId);
  if (!student) return;

  const curr = state.payload.cohort_analytics.currency_hint;
  const lin = student.lineage || {};
  const currentStatus = state.interventions[student.student_id] || (student.risk_tier === 'Low Risk' ? 'Monitoring' : 'Unassigned');

  els.modalStudentId.textContent = student.student_id;
  els.modalStudentTitle.textContent = student.student_name;
  els.modalStudentGrade.textContent = `${student.grade_level} • Dropout Probability: ${student.dropout_probability_pct}%`;

  const bd = student.factor_breakdown;
  els.modalBody.innerHTML = `
    <div style="display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:0.75rem;">
      ${tierBadgeHtml(student.risk_tier)}
      <div class="tabular" style="font-size:0.85rem; color:var(--ink-secondary)">
        Att: <strong>${student.attendance_pct}%</strong> &bull; Marks: <strong>${student.marks_pct}% (${esc(student.letter_grade)} / ${student.gpa4} GPA)</strong> &bull; Fees: <strong>${formatMoney(student.fee_due, curr)} (${esc(student.fee_status)})</strong>
      </div>
    </div>

    <div class="dialog-section">
      <h2>Python Data Cleaning Lineage (Before → After)</h2>
      <div class="bars-stack" style="gap:0.5rem; font-size:0.84rem;">
        <div>Attendance: <span class="raw-strike">${esc(lin.raw_attendance || '—')}</span> → <strong class="tabular">${student.attendance_pct}%</strong> ${student.attendance_imputed ? '<span class="cell-flag">Imputed</span>' : ''}</div>
        <div>Marks / Grade: <span class="raw-strike">${esc(lin.raw_marks || '—')}</span> <span class="badge badge-indigo">${esc(student.detected_scale)}</span> → <strong class="tabular">${student.marks_pct}% (${esc(student.letter_grade)})</strong></div>
        <div>Fee Ledger: <span class="raw-strike">${esc(lin.raw_fee_due || '—')} (${esc(lin.raw_fee_status || '—')})</span> → <strong class="tabular">${formatMoney(student.fee_due, curr)} (${esc(student.fee_status)})</strong></div>
      </div>
      ${(lin.transformations && lin.transformations.length) ? `
        <div class="tag-list" style="margin-block-start:0.65rem;">
          ${lin.transformations.map(t => `<span class="mini-tag">${esc(t)}</span>`).join('')}
        </div>
      ` : ''}
    </div>

    <div class="dialog-section">
      <h2>Explainable Risk Factor Breakdown</h2>
      <div class="bars-stack" style="gap:0.55rem;">
        <div>
          <div class="bar-row-meta"><span>Attendance Impact</span><span class="tabular">${bd.attendance_share}%</span></div>
          <div class="bar-track"><div class="bar-fill is-rose" style="inline-size:${Math.max(3, bd.attendance_share)}%"></div></div>
        </div>
        <div>
          <div class="bar-row-meta"><span>Academic Marks Impact</span><span class="tabular">${bd.academic_share}%</span></div>
          <div class="bar-track"><div class="bar-fill is-amber" style="inline-size:${Math.max(3, bd.academic_share)}%"></div></div>
        </div>
        <div>
          <div class="bar-row-meta"><span>Fee Delinquency Impact</span><span class="tabular">${bd.fee_share}%</span></div>
          <div class="bar-track"><div class="bar-fill" style="inline-size:${Math.max(3, bd.fee_share)}%"></div></div>
        </div>
      </div>
      <ul style="margin-block-start:0.75rem; padding-inline-start:1.2rem; font-size:0.85rem; display:grid; gap:0.35rem;">
        ${student.risk_drivers.map(d => `<li>${esc(d)}</li>`).join('')}
      </ul>
    </div>

    <div class="dialog-section">
      <h2>Recommended Counselor Interventions</h2>
      <ul style="padding-inline-start:1.2rem; font-size:0.86rem; display:grid; gap:0.35rem; margin-block-end:0.85rem;">
        ${student.recommended_interventions.map(i => `<li>${esc(i)}</li>`).join('')}
      </ul>
      <label for="modal-intervention-select" class="field-label">Assign Intervention Workflow Status</label>
      <select id="modal-intervention-select" class="form-control" style="margin-block-start:0.35rem;">
        ${['Unassigned', 'Outreach Scheduled', 'Tutoring Active', 'Fee Plan Set', 'Monitoring', 'Resolved'].map(opt => `
          <option value="${opt}" ${opt === currentStatus ? 'selected' : ''}>${opt}</option>
        `).join('')}
      </select>
    </div>
  `;

  document.getElementById('modal-intervention-select').addEventListener('change', (e) => {
    state.interventions[student.student_id] = e.target.value;
    renderStudentsDirectory();
  });

  els.modal.showModal();
}

els.closeModalBtn.addEventListener('click', () => els.modal.close());
if (typeof HTMLDialogElement !== 'undefined' && HTMLDialogElement.prototype && !('closedBy' in HTMLDialogElement.prototype)) {
  const isOutside = (e) => {
    const r = els.modal.getBoundingClientRect();
    return e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom;
  };
  let pressedOut = false;
  els.modal.addEventListener('pointerdown', (e) => { pressedOut = isOutside(e); });
  els.modal.addEventListener('pointerup', (e) => { if (pressedOut && isOutside(e)) els.modal.close(); });
}

/* Manual Single Record Predictor */
function initSandbox() {
  els.sandboxResetBtn.addEventListener('click', () => {
    els.sandboxResultBody.innerHTML = 'Enter student details on the left and submit to inspect how Python normalizes the grading scale, reconciles fee conflicts, imputes missing fields, and scores dropout risk.';
  });

  els.singleForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(els.singleForm);
    const payload = {
      student_id: String(formData.get('student_id') || '').trim(),
      student_name: String(formData.get('student_name') || '').trim(),
      grade_level: String(formData.get('grade_level') || '').trim(),
      attendance: String(formData.get('attendance') || ''),
      marks_raw: String(formData.get('marks_raw') || ''),
      fee_due: String(formData.get('fee_due') || ''),
      fee_status: String(formData.get('fee_status') || ''),
      imputation_method: String(formData.get('imputation_method') || 'cohort_median'),
      reference_csv: state.uploadedCsvText || null,
    };

    els.sandboxResultBody.innerHTML = '<p>Running Python cleaning &amp; ML scoring...</p>';
    try {
      const res = await fetch(`${state.apiBase}/api/analyze/single`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await parseApiResponse(res);
      if (!data.student) throw new Error('Could not evaluate student.');

      const st = data.student;
      const lin = data.lineage || {};
      const curr = String(payload.fee_due || '').includes('$') ? 'USD' : 'INR';

      els.sandboxResultBody.innerHTML = `
        <div style="display:grid; gap:1rem;">
          <div style="display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:0.65rem;">
            <div>
              <strong style="font-size:1.1rem;">${esc(st.student_name)}</strong>
              <span class="tabular" style="color:var(--ink-muted)">(${esc(st.student_id)} &bull; ${esc(st.grade_level)})</span>
            </div>
            ${tierBadgeHtml(st.risk_tier)}
          </div>

          <div class="kpi-card ${st.risk_tier === 'High Risk' ? 'kpi-rose' : (st.risk_tier === 'Medium Risk' ? 'kpi-amber' : 'kpi-emerald')}">
            <div class="kpi-label">Predicted Dropout Probability</div>
            <div class="kpi-value tabular">${st.dropout_probability_pct}%</div>
            <p class="kpi-sub">${esc(st.primary_driver)}</p>
          </div>

          <div class="dialog-section">
            <h2>Python Cleaning Trace</h2>
            <div class="tag-list">
              ${(lin.transformations && lin.transformations.length)
                ? lin.transformations.map(t => `<span class="mini-tag">${esc(t)}</span>`).join('')
                : '<span class="mini-tag">No formatting issues needed repair</span>'}
            </div>
            <div class="tabular" style="margin-block-start:0.75rem; font-size:0.84rem;">
              Cleaned Vector → Att: <strong>${st.attendance_pct}%</strong> | Marks: <strong>${st.marks_pct}% (${esc(st.letter_grade)})</strong> | Fee: <strong>${formatMoney(st.fee_due, curr)} (${esc(st.fee_status)})</strong>
            </div>
          </div>

          <div class="dialog-section">
            <h2>Counselor Action Plan</h2>
            <ul style="padding-inline-start:1.2rem; font-size:0.85rem; display:grid; gap:0.35rem;">
              ${st.recommended_interventions.map(i => `<li>${esc(i)}</li>`).join('')}
            </ul>
          </div>
        </div>
      `;
    } catch (err) {
      els.sandboxResultBody.innerHTML = `<div class="alert-banner">${esc(err.message)}</div>`;
    }
  });
}

/* File Selection & Events */
async function handleSelectedFile(file) {
  if (!file) return;
  state.uploadedFile = file;
  state.uploadedCsvText = await file.text();
  els.dropzoneTitle.textContent = `Selected: ${file.name}`;
  els.runBtn.disabled = false;
  await runUploadAnalysis();
}

els.runBtn.addEventListener('click', runUploadAnalysis);

els.dropzone.addEventListener('click', () => els.fileInput.click());
els.dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    els.fileInput.click();
  }
});

els.dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  els.dropzone.classList.add('is-dragover');
});

els.dropzone.addEventListener('dragleave', () => {
  els.dropzone.classList.remove('is-dragover');
});

els.dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  els.dropzone.classList.remove('is-dragover');
  const file = e.dataTransfer?.files?.[0];
  if (file) handleSelectedFile(file);
});

els.fileInput.addEventListener('change', () => {
  const file = els.fileInput.files?.[0];
  if (file) handleSelectedFile(file);
});

els.onlyModifiedToggle.addEventListener('change', renderDiffTable);
els.studentSearch.addEventListener('input', (e) => {
  state.searchQuery = e.target.value;
  renderStudentsDirectory();
});
els.gradeFilter.addEventListener('change', (e) => {
  state.activeGradeFilter = e.target.value;
  renderStudentsDirectory();
});

document.querySelectorAll('.pill-filter').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.pill-filter').forEach(b => b.classList.remove('is-active'));
    btn.classList.add('is-active');
    state.activeTierFilter = btn.dataset.tier;
    renderStudentsDirectory();
  });
});

els.downloadTemplateBtn.addEventListener('click', () => {
  window.location.href = `${state.apiBase}/api/template.csv`;
});

els.downloadCleanBtn.addEventListener('click', () => {
  if (!state.payload?.cleaned_csv) return;
  const baseName = (state.uploadedFile?.name || 'students').replace(/\.csv$/i, '');
  const blob = new Blob([state.payload.cleaned_csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${baseName}_cleaned_risk_scored.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
});

function initAuth() {
  const getStoredUser = () => {
    try {
      const stored = sessionStorage.getItem('edushield_user') || localStorage.getItem('edushield_user');
      return stored ? JSON.parse(stored) : null;
    } catch {
      return null;
    }
  };

  const user = getStoredUser();
  if (user) {
    if (els.userIdDisplay) els.userIdDisplay.textContent = user.id;
    if (els.userRoleDisplay) els.userRoleDisplay.textContent = user.role;
    if (els.userAvatarInitials) {
      const clean = (user.id || 'FA').replace(/[^a-zA-Z0-9 ]/g, '').trim();
      const parts = clean.split(/\s+/);
      els.userAvatarInitials.textContent = parts.length > 1
        ? (parts[0][0] + parts[1][0]).toUpperCase()
        : clean.slice(0, 2).toUpperCase() || 'FA';
    }
  }

  els.signOutBtn?.addEventListener('click', () => {
    try {
      sessionStorage.removeItem('edushield_user');
      localStorage.removeItem('edushield_user');
    } catch {}
    const isGitHubPages = window.location.hostname.includes('github.io');
    window.location.href = isGitHubPages ? 'index.html' : '/login';
  });
}

initTabs();
initSortableHeaders();
initSandbox();
initAuth();
renderEmptyDashboard();
loadInitialHealth();

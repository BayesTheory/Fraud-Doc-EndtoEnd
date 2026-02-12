/* ═══════════════════════════════════════════════════
   Bayes Fraud-Doc Pipeline — Frontend Logic
   ═══════════════════════════════════════════════════ */

const API = '';

// ─── Router ───
function initRouter() {
    window.addEventListener('hashchange', route);
    route();
}

function route() {
    const hash = window.location.hash || '#/analyze';
    const page = hash.replace('#/', '') || 'analyze';

    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

    const pageEl = document.getElementById(`page-${page}`);
    const navEl = document.querySelector(`[data-page="${page}"]`);

    if (pageEl) pageEl.classList.add('active');
    if (navEl) navEl.classList.add('active');

    if (page === 'dashboard' && sessionStorage.getItem('dash-auth') === 'true') {
        document.getElementById('dash-login').classList.add('hidden');
        document.getElementById('dash-content').classList.remove('hidden');
        loadCases();
    }
}

// ─── File Upload ───
let selectedFile = null;

function initUpload() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const btnAnalyze = document.getElementById('btn-analyze');
    const previewClear = document.getElementById('preview-clear');

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });
    btnAnalyze.addEventListener('click', analyzeDocument);
    // runDemo is called via onclick
    previewClear.addEventListener('click', clearFile);
}

function setFile(file) {
    selectedFile = file;
    const preview = document.getElementById('preview-box');
    const img = document.getElementById('preview-img');
    const name = document.getElementById('preview-name');

    img.src = URL.createObjectURL(file);
    name.textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KB`;
    preview.classList.remove('hidden');
    document.getElementById('btn-analyze').disabled = false;
}

function clearFile() {
    selectedFile = null;
    document.getElementById('preview-box').classList.add('hidden');
    document.getElementById('btn-analyze').disabled = true;
    document.getElementById('file-input').value = '';
}

// ─── Analyze ───
async function analyzeDocument() {
    if (!selectedFile) return;
    showLoading();

    const formData = new FormData();
    formData.append('file', selectedFile);

    const steps = ['quality', 'ocr', 'rules', 'llm'];
    let stepIdx = 0;
    const stepTimer = setInterval(() => {
        if (stepIdx < steps.length) {
            document.querySelectorAll('.load-step').forEach((s, i) => {
                if (i < stepIdx) s.classList.add('done');
                else if (i === stepIdx) s.classList.add('active');
            });
            stepIdx++;
        }
    }, 1500);

    try {
        const res = await fetch(`${API}/api/v1/analyze`, { method: 'POST', body: formData });
        clearInterval(stepTimer);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        showResults(data);
    } catch (err) {
        clearInterval(stepTimer);
        showError(err.message);
    }
}

function showLoading() {
    document.getElementById('results-placeholder').classList.add('hidden');
    document.getElementById('results-content').classList.add('hidden');
    document.getElementById('loading-state').classList.remove('hidden');
    document.querySelectorAll('.load-step').forEach(s => { s.classList.remove('active', 'done'); });
}

function showError(msg) {
    document.getElementById('loading-state').classList.add('hidden');
    const el = document.getElementById('results-content');
    el.classList.remove('hidden');
    el.innerHTML = `<div class="result-card"><h4>❌ Error</h4><p>${msg}</p></div>`;
}

function showResults(data) {
    document.getElementById('loading-state').classList.add('hidden');
    const el = document.getElementById('results-content');
    el.classList.remove('hidden');

    const d = data.final_decision;
    const dClass = d === 'APPROVED' ? 'approved' : d === 'REJECTED' ? 'rejected' : 'review';
    const dIcon = d === 'APPROVED' ? '✅' : d === 'REJECTED' ? '🚫' : '⚠️';
    const score = (data.final_score * 100).toFixed(0);

    let html = `
    <div class="decision-banner ${dClass}">
      <div style="display:flex;align-items:center;gap:16px">
        <div class="decision-icon">${dIcon}</div>
        <div class="decision-text">
          <h3>${d}</h3>
          <p>${data.case_id} · ${data.total_latency_ms?.toFixed(0) || 0}ms</p>
        </div>
      </div>
      <div class="decision-score">${score}%</div>
    </div>`;

    // Metrics row
    html += `<div class="metrics-grid">`;
    if (data.quality) {
        html += `<div class="metric-item"><div class="metric-label">Quality Score</div><div class="metric-value">${(data.quality.quality_score * 100).toFixed(0)}%</div><div class="metric-sub">${data.quality.recommendation}</div></div>`;
    }
    if (data.ocr) {
        html += `<div class="metric-item"><div class="metric-label">OCR Confidence</div><div class="metric-value">${(data.ocr.avg_confidence * 100).toFixed(1)}%</div><div class="metric-sub">${data.ocr.ocr_engine || ''}</div></div>`;
    }
    if (data.rules) {
        html += `<div class="metric-item"><div class="metric-label">Rules</div><div class="metric-value">${data.rules.rules_passed}/${data.rules.rules_total}</div><div class="metric-sub">${data.rules.risk_level}</div></div>`;
    }
    if (data.llm && !data.llm.error) {
        html += `<div class="metric-item"><div class="metric-label">Fraud Prob.</div><div class="metric-value" style="color:${data.llm.fraud_probability > 0.5 ? 'var(--red)' : 'var(--green)'}">${(data.llm.fraud_probability * 100).toFixed(0)}%</div><div class="metric-sub">${data.llm.recommendation || ''}</div></div>`;
    }
    html += `</div>`;

    // LLM Analysis
    if (data.llm && !data.llm.error) {
        html += `<div class="result-card"><h4>🧠 AI Fraud Analysis</h4><div class="llm-section">`;
        html += `<div class="llm-assessment">${data.llm.assessment || ''}</div>`;
        if (data.llm.anomalies?.length) {
            html += `<div class="llm-anomalies">${data.llm.anomalies.map(a => `<span class="anomaly-tag">${a}</span>`).join('')}</div>`;
        }
        if (data.llm.reasoning) {
            html += `<div class="llm-reasoning">${data.llm.reasoning}</div>`;
        }
        html += `</div></div>`;
    }

    // OCR Fields
    if (data.ocr?.fields?.length) {
        html += `<div class="result-card"><h4>📝 OCR Fields <span style="font-weight:400;color:var(--text-muted)">· ${data.ocr.doc_type_detected || ''}</span></h4><div class="field-list">`;
        for (const f of data.ocr.fields) {
            html += `<div class="field-item"><span class="field-name">${f.name}</span><span><span class="field-value">${f.value}</span><span class="field-conf">${(f.confidence * 100).toFixed(0)}%</span></span></div>`;
        }
        html += `</div></div>`;
    }

    // Rules Violations
    if (data.rules?.violations?.length) {
        html += `<div class="result-card"><h4>⚖️ Rule Violations</h4>`;
        for (const v of data.rules.violations) {
            html += `<div class="violation-item"><span class="violation-sev ${v.severity}">${v.severity}</span><div><div class="violation-text">${v.rule_name}</div><div class="violation-detail">${v.detail}</div></div></div>`;
        }
        html += `</div>`;
    }

    // Latencies
    if (data.stage_latencies) {
        html += `<div class="latency-row">`;
        for (const [k, v] of Object.entries(data.stage_latencies)) {
            html += `<span>${k}: ${v.toFixed(0)}ms</span>`;
        }
        html += `<span>total: ${data.total_latency_ms?.toFixed(0) || 0}ms</span></div>`;
    }

    // Raw JSON
    html += `<div class="raw-json"><span class="raw-json-toggle" onclick="this.nextElementSibling.classList.toggle('open')">{ } View Raw JSON ▸</span><pre class="code-block raw-json-content">${JSON.stringify(data, null, 2)}</pre></div>`;

    el.innerHTML = html;
}

// ─── Demo ───
// ─── Demo Data ───
const DEMO_CASES = {
    approved: {
        case_id: "demo-approved-001",
        final_decision: "APPROVED",
        final_score: 0.98,
        total_latency_ms: 3240,
        quality: { quality_score: 0.96, recommendation: "ACCEPT" },
        ocr: {
            avg_confidence: 0.99,
            ocr_engine: "Hybrid (PaddleOCR v5 + EasyOCR)",
            doc_type_detected: "PASSPORT",
            fields: [
                { name: "primary_identifier", value: "SILVA", confidence: 0.99 },
                { name: "secondary_identifier", value: "MARIA", confidence: 0.99 },
                { name: "document_number", value: "BR1234567", confidence: 0.99 },
                { name: "nationality", value: "BRA", confidence: 0.99 },
                { name: "date_of_birth", value: "10.05.1988", confidence: 0.99 }
            ]
        },
        rules: { rules_passed: 10, rules_total: 10, risk_level: "LOW", violations: [] },
        llm: {
            fraud_probability: 0.02,
            assessment: "Perfect document. All checksums pass, data is consistent, and no anomalies detected.",
            recommendation: "APPROVE",
            anomalies: []
        }
    },
    rejected: {
        case_id: "demo-fraud-002",
        final_decision: "REJECTED",
        final_score: 0.12,
        total_latency_ms: 4100,
        quality: { quality_score: 0.88, recommendation: "ACCEPT" },
        ocr: {
            avg_confidence: 0.92,
            ocr_engine: "Hybrid (PaddleOCR v5 + EasyOCR)",
            doc_type_detected: "PASSPORT",
            fields: [
                { name: "primary_identifier", value: "SMITH", confidence: 0.95 },
                { name: "document_number", value: "X12345678", confidence: 0.90 }
            ]
        },
        rules: {
            rules_passed: 7, rules_total: 10, risk_level: "CRITICAL",
            violations: [
                { severity: "CRITICAL", rule_name: "Document Number Checksum", detail: "Checksum mismatch (expected 8, got 5)" },
                { severity: "HIGH", rule_name: "Cross-Check", detail: "Surname mismatch VIZ vs MRZ" }
            ]
        },
        llm: {
            fraud_probability: 0.95,
            assessment: "Strong evidence of tampering. Checksum failure and name mismatch are critical red flags.",
            recommendation: "REJECT",
            anomalies: ["Checksum failure", "Name mismatch"]
        }
    },
    review: {
        case_id: "demo-review-003",
        final_decision: "REVIEW",
        final_score: 0.55,
        total_latency_ms: 3800,
        quality: { quality_score: 0.45, recommendation: "REVIEW" },
        ocr: {
            avg_confidence: 0.72,
            ocr_engine: "Hybrid (PaddleOCR v5 + EasyOCR)",
            doc_type_detected: "PASSPORT",
            fields: [
                { name: "primary_identifier", value: "TANAKA", confidence: 0.72 },
                { name: "document_number", value: "TK9988776", confidence: 0.68 }
            ]
        },
        rules: { rules_passed: 10, rules_total: 10, risk_level: "LOW", violations: [] },
        llm: {
            fraud_probability: 0.20,
            assessment: "Document rules pass, but image quality is very low (blur). Cannot reliably authenticate security features.",
            recommendation: "REVIEW",
            anomalies: ["High Blur Detected", "Low OCR Confidence"]
        }
    }
};

async function runDemo(type) {
    if (!type || !DEMO_CASES[type]) type = 'approved';

    showLoading();

    // Simulate pipeline steps visual
    const steps = ['quality', 'ocr', 'rules', 'llm'];
    let stepIdx = 0;

    // Faster for demo
    const stepTimer = setInterval(() => {
        if (stepIdx < steps.length) {
            document.querySelectorAll('.load-step').forEach((s, i) => {
                if (i < stepIdx) s.classList.add('done');
                else if (i === stepIdx) s.classList.add('active');
            });
            stepIdx++;
        }
    }, 400);

    // Simulate network delay
    setTimeout(() => {
        clearInterval(stepTimer);
        showResults(DEMO_CASES[type]);
    }, 2000);
}

// ─── Dashboard ───
function initDashboard() {
    document.getElementById('btn-login').addEventListener('click', login);
    document.getElementById('dash-password').addEventListener('keydown', e => { if (e.key === 'Enter') login(); });
    document.getElementById('btn-refresh').addEventListener('click', loadCases);
    document.getElementById('btn-send').addEventListener('click', sendChat);
    document.getElementById('chat-input').addEventListener('keydown', e => { if (e.key === 'Enter') sendChat(); });
}

async function login() {
    const pwd = document.getElementById('dash-password').value;
    try {
        const res = await fetch(`${API}/api/v1/auth`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ password: pwd }),
        });
        const data = await res.json();
        if (data.success) {
            sessionStorage.setItem('dash-auth', 'true');
            document.getElementById('dash-login').classList.add('hidden');
            document.getElementById('dash-content').classList.remove('hidden');
            document.getElementById('login-error').classList.add('hidden');
            loadCases();
        } else {
            document.getElementById('login-error').classList.remove('hidden');
        }
    } catch (err) {
        document.getElementById('login-error').textContent = err.message;
        document.getElementById('login-error').classList.remove('hidden');
    }
}

async function loadCases() {
    try {
        const res = await fetch(`${API}/api/v1/cases`);
        const data = await res.json();
        const cases = data.cases || [];

        // Stats
        document.getElementById('stat-total').textContent = cases.length;
        document.getElementById('stat-approved').textContent = cases.filter(c => c.final_decision === 'APPROVED').length;
        document.getElementById('stat-rejected').textContent = cases.filter(c => c.final_decision === 'REJECTED').length;
        document.getElementById('stat-review').textContent = cases.filter(c => c.final_decision === 'REVIEW').length;

        // Cases table
        const table = document.getElementById('cases-table');
        if (!cases.length) {
            table.innerHTML = '<p class="muted">No cases yet. Upload a document to get started.</p>';
            return;
        }

        table.innerHTML = cases.map(c => {
            const d = c.final_decision;
            const badgeClass = d === 'APPROVED' ? 'badge-approved' : d === 'REJECTED' ? 'badge-rejected' : 'badge-review';
            const name = c.ocr?.fields?.find(f => f.name === 'primary_identifier')?.value || 'Unknown';
            const score = c.final_score != null ? (c.final_score * 100).toFixed(0) + '%' : '—';
            const time = c.timestamp ? new Date(c.timestamp).toLocaleTimeString() : '';
            return `<div class="case-row" onclick='showCaseDetail(${JSON.stringify(c.case_id)})'>
        <span class="case-id">${(c.run_id || c.case_id || '').slice(0, 8)}</span>
        <span class="case-name">${name}</span>
        <span class="badge ${badgeClass}">${d}</span>
        <span class="case-score">${score}</span>
        <span class="case-time">${time}</span>
      </div>`;
        }).join('');
    } catch (err) {
        document.getElementById('cases-table').innerHTML = `<p class="muted">Error: ${err.message}</p>`;
    }
}

function showCaseDetail(caseId) {
    // Navigate to analyze page and show the case
    window.location.hash = '#/analyze';
    fetch(`${API}/api/v1/cases/${caseId}`)
        .then(r => r.json())
        .then(data => {
            document.getElementById('results-placeholder').classList.add('hidden');
            showResults(data);
        });
}

async function sendChat() {
    const input = document.getElementById('chat-input');
    const msg = input.value.trim();
    if (!msg) return;
    input.value = '';

    const messages = document.getElementById('chat-messages');

    // User message
    messages.innerHTML += `<div class="chat-msg user"><div class="msg-avatar">👤</div><div class="msg-bubble">${escapeHtml(msg)}</div></div>`;
    messages.scrollTop = messages.scrollHeight;

    // Loading indicator
    const loadingId = 'chat-loading-' + Date.now();
    messages.innerHTML += `<div class="chat-msg bot" id="${loadingId}"><div class="msg-avatar">🤖</div><div class="msg-bubble"><em>Thinking...</em></div></div>`;
    messages.scrollTop = messages.scrollHeight;

    try {
        const res = await fetch(`${API}/api/v1/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg }),
        });
        const data = await res.json();

        const loadingEl = document.getElementById(loadingId);
        if (loadingEl) loadingEl.remove();

        const reply = formatMarkdown(data.reply || 'No response');
        const meta = data.latency_ms ? `<span style="font-size:11px;color:var(--text-muted);display:block;margin-top:8px">${data.model} · ${data.latency_ms.toFixed(0)}ms</span>` : '';
        messages.innerHTML += `<div class="chat-msg bot"><div class="msg-avatar">🤖</div><div class="msg-bubble">${reply}${meta}</div></div>`;
        messages.scrollTop = messages.scrollHeight;
    } catch (err) {
        const loadingEl = document.getElementById(loadingId);
        if (loadingEl) loadingEl.remove();
        messages.innerHTML += `<div class="chat-msg bot"><div class="msg-avatar">🤖</div><div class="msg-bubble" style="color:var(--red)">Error: ${err.message}</div></div>`;
    }
}

// ─── API Page ───
async function testHealth() {
    const el = document.getElementById('health-response');
    el.textContent = 'Loading...';
    try {
        const res = await fetch(`${API}/health`);
        const data = await res.json();
        el.textContent = JSON.stringify(data, null, 2);
    } catch (err) {
        el.textContent = `Error: ${err.message}`;
    }
}
// Make it global
window.testHealth = testHealth;

// ─── Helpers ───
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function formatMarkdown(text) {
    // Basic markdown: bold, italic, lists, code
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code style="background:var(--bg-input);padding:2px 6px;border-radius:3px;font-family:var(--mono);font-size:12px">$1</code>')
        .replace(/^- (.*)/gm, '• $1')
        .replace(/\n/g, '<br>');
}

// ─── Init ───
document.addEventListener('DOMContentLoaded', () => {
    initRouter();
    initUpload();
    initDashboard();
});

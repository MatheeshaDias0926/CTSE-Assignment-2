/* ═══════════════════════════════════════════════════════════════
   Smart Factory MAS — Dashboard JavaScript
   SSE client, DOM updates, and pipeline state management
   ═══════════════════════════════════════════════════════════════ */

const DEFAULT_QUERY =
    "Analyse the automotive production data for all production lines. " +
    "Identify efficiency bottlenecks using PTP metrics, check raw material " +
    "inventory for shortages, and provide a comprehensive optimization plan " +
    "with actionable recommendations.";

// ── DOM References ────────────────────────────────────────────────
const els = {
    queryInput: () => document.getElementById("query-input"),
    runBtn: () => document.getElementById("run-btn"),
    evalBtn: () => document.getElementById("eval-btn"),
    statusDot: () => document.getElementById("status-dot"),
    statusText: () => document.getElementById("status-text"),
    timer: () => document.getElementById("timer"),
    pipelineSteps: () => document.querySelectorAll(".pipeline-step"),
    statsSection: () => document.getElementById("stats-section"),
    resultsSection: () => document.getElementById("results-section"),
    ptpBody: () => document.getElementById("ptp-table-body"),
    inventoryBody: () => document.getElementById("inventory-table-body"),
    reportContent: () => document.getElementById("report-content"),
    evalSection: () => document.getElementById("eval-section"),
    evalScores: () => document.getElementById("eval-scores"),
    evalOverall: () => document.getElementById("eval-overall-score"),
    evalSummary: () => document.getElementById("eval-summary"),
    statRecords: () => document.getElementById("stat-records"),
    statLines: () => document.getElementById("stat-lines"),
    statDowntime: () => document.getElementById("stat-downtime"),
    statElapsed: () => document.getElementById("stat-elapsed"),
};

let timerInterval = null;
let startTime = null;

// ── Initialization ────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    els.queryInput().value = DEFAULT_QUERY;
    checkHealth();
    loadDataSummary();

    els.runBtn().addEventListener("click", startWorkflow);
    els.evalBtn().addEventListener("click", runEvaluation);

    // Allow Enter key to start workflow
    els.queryInput().addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            startWorkflow();
        }
    });
});

// ── Health Check ──────────────────────────────────────────────────
async function checkHealth() {
    try {
        const res = await fetch("/api/health");
        const data = await res.json();
        if (data.status === "ok") {
            els.statusDot().className = "status-dot online";
            els.statusText().textContent = `Ollama Online · ${data.models[0] || "llama3:8b"}`;
        } else {
            setOffline();
        }
    } catch {
        setOffline();
    }
}

function setOffline() {
    els.statusDot().className = "status-dot offline";
    els.statusText().textContent = "Ollama Offline";
}

// ── Load Data Summary ─────────────────────────────────────────────
async function loadDataSummary() {
    try {
        const res = await fetch("/api/data/summary");
        const data = await res.json();
        if (!data.error) {
            els.statRecords().textContent = data.total_records;
            els.statLines().textContent = data.production_lines.length;
            els.statDowntime().textContent = `${data.total_downtime}m`;
        }
    } catch {
        /* silent */
    }
}

// ── Start Workflow (SSE) ──────────────────────────────────────────
function startWorkflow() {
    const query = els.queryInput().value.trim();
    if (!query) return;

    // Reset UI
    resetPipeline();
    els.runBtn().disabled = true;
    els.runBtn().innerHTML = '<span class="spinner"></span> Running...';
    els.resultsSection().classList.add("hidden");
    els.evalSection().classList.add("hidden");
    els.statsSection().classList.remove("hidden");

    // Start timer
    startTime = Date.now();
    els.timer().textContent = "⏱ 0.0s";
    els.timer().classList.remove("hidden");
    timerInterval = setInterval(updateTimer, 100);

    // Connect SSE
    const url = `/api/run/stream?query=${encodeURIComponent(query)}`;
    const source = new EventSource(url);

    source.addEventListener("step", (e) => {
        const data = JSON.parse(e.data);
        handleStep(data);
    });

    source.addEventListener("complete", (e) => {
        const data = JSON.parse(e.data);
        handleComplete(data);
        source.close();
    });

    source.addEventListener("error", (e) => {
        // Check if it's an actual error event with data
        if (e.data) {
            const data = JSON.parse(e.data);
            handleError(data.message);
        }
        source.close();
    });

    source.onerror = () => {
        // SSE connection error (stream ended)
        stopTimer();
        els.runBtn().disabled = false;
        els.runBtn().innerHTML = '🚀 Run Analysis';
    };
}

// ── Handle Step Event ─────────────────────────────────────────────
function handleStep(data) {
    const { agent, detail, ptp_results, inventory, findings, final_report, optimization_plan } = data;

    // Update pipeline step
    const stepEl = document.querySelector(`[data-agent="${agent}"]`);
    if (stepEl) {
        // Mark previous running steps as done
        document.querySelectorAll(".pipeline-step.running").forEach((el) => {
            if (el !== stepEl) {
                el.classList.remove("running");
                el.classList.add("done");
                const statusEl = el.querySelector(".step-status");
                if (statusEl && !statusEl.textContent.includes("✓")) {
                    statusEl.textContent = "✓ Complete";
                }
            }
        });

        stepEl.classList.remove("idle");
        stepEl.classList.add("running");
        const statusEl = stepEl.querySelector(".step-status");
        if (statusEl) statusEl.textContent = detail;
    }

    // Populate PTP results
    if (ptp_results && ptp_results.length > 0) {
        populatePTP(ptp_results);
        els.resultsSection().classList.remove("hidden");
    }

    // Populate inventory
    if (inventory && inventory.results) {
        populateInventory(inventory.results);
    }

    // Store findings
    if (findings) {
        document.getElementById("bottleneck-findings").textContent = findings;
        document.getElementById("findings-card").classList.remove("hidden");
    }

    // Store optimization plan
    if (optimization_plan) {
        els.reportContent().textContent = optimization_plan;
        document.getElementById("report-card").classList.remove("hidden");
    }

    // Final report
    if (final_report) {
        els.reportContent().textContent = final_report;
    }
}

// ── Handle Complete ───────────────────────────────────────────────
function handleComplete(data) {
    stopTimer();

    // Mark all steps as done
    document.querySelectorAll(".pipeline-step").forEach((el) => {
        el.classList.remove("running");
        el.classList.add("done");
    });

    // Update final status
    const synthStep = document.querySelector('[data-agent="synthesizer"]');
    if (synthStep) {
        const statusEl = synthStep.querySelector(".step-status");
        if (statusEl) statusEl.textContent = "✓ Complete";
    }

    // Show results
    els.resultsSection().classList.remove("hidden");
    els.evalSection().classList.remove("hidden");

    if (data.final_report) {
        els.reportContent().textContent = data.final_report;
        document.getElementById("report-card").classList.remove("hidden");
    }

    els.statElapsed().textContent = `${data.elapsed}s`;

    // Re-enable button
    els.runBtn().disabled = false;
    els.runBtn().innerHTML = '🚀 Run Analysis';
}

// ── Handle Error ──────────────────────────────────────────────────
function handleError(message) {
    stopTimer();
    els.runBtn().disabled = false;
    els.runBtn().innerHTML = '🚀 Run Analysis';

    // Mark current running step as error
    document.querySelectorAll(".pipeline-step.running").forEach((el) => {
        el.classList.remove("running");
        el.classList.add("error");
        const statusEl = el.querySelector(".step-status");
        if (statusEl) statusEl.textContent = `✗ Error`;
    });

    alert(`Workflow Error: ${message}`);
}

// ── Populate PTP Table ────────────────────────────────────────────
function populatePTP(results) {
    const body = els.ptpBody();
    body.innerHTML = "";

    results.forEach((r) => {
        const ptp = r.ptp_percentage != null ? r.ptp_percentage.toFixed(2) : "N/A";
        const cls = (r.classification || "").toLowerCase().replace("_", "-");
        const badgeClass =
            cls === "critical" ? "badge-critical"
            : cls === "warning" ? "badge-warning"
            : cls === "on-target" ? "badge-on-target"
            : "badge-exceeding";

        body.innerHTML += `
            <tr>
                <td style="font-weight:600;color:var(--text-primary)">${r.line_id || "—"}</td>
                <td>${ptp}%</td>
                <td>${r.planned_total != null ? r.planned_total : r.planned || "—"}</td>
                <td>${r.actual_total != null ? r.actual_total : r.actual || "—"}</td>
                <td><span class="badge ${badgeClass}">${r.classification || "—"}</span></td>
            </tr>`;
    });
}

// ── Populate Inventory Table ──────────────────────────────────────
function populateInventory(results) {
    const body = els.inventoryBody();
    body.innerHTML = "";

    results.forEach((r) => {
        const deficit = (r.stock_qty - r.min_threshold).toFixed(0);
        body.innerHTML += `
            <tr>
                <td style="font-weight:600;color:var(--text-primary)">${r.material_name || r.name || "—"}</td>
                <td>${r.stock_qty}</td>
                <td>${r.min_threshold}</td>
                <td style="color:var(--accent-red);font-weight:600">${deficit}</td>
                <td>${r.supplier || "—"}</td>
            </tr>`;
    });
}

// ── Evaluation ────────────────────────────────────────────────────
async function runEvaluation() {
    const btn = els.evalBtn();
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Evaluating...';

    try {
        const res = await fetch("/api/evaluate", { method: "POST" });
        const data = await res.json();

        if (data.error) {
            alert(`Evaluation Error: ${data.error}`);
            return;
        }

        const eval_ = data.evaluation;
        const scoresContainer = els.evalScores();
        scoresContainer.innerHTML = "";

        const criteria = [
            { key: "factual_grounding", label: "Factual" },
            { key: "actionable_specificity", label: "Actionable" },
            { key: "hallucination_detection", label: "No Halluc." },
            { key: "completeness", label: "Complete" },
            { key: "logical_coherence", label: "Coherent" },
        ];

        let total = 0;
        let count = 0;

        criteria.forEach((c) => {
            const item = eval_.criteria ? eval_.criteria[c.key] : null;
            const score = item ? item.score : 0;
            total += score;
            count++;

            scoresContainer.innerHTML += `
                <div class="eval-score-item">
                    <div class="eval-score-label">${c.label}</div>
                    <div class="eval-score-bar">
                        <div class="eval-score-fill" style="width:${score * 20}%"></div>
                    </div>
                    <div class="eval-score-value">${score}/5</div>
                </div>`;
        });

        const avg = count > 0 ? (total / count).toFixed(1) : "0.0";
        els.evalOverall().textContent = `${avg}/5.0`;

        if (eval_.summary) {
            els.evalSummary().textContent = eval_.summary;
            els.evalSummary().classList.remove("hidden");
        }
    } catch (e) {
        alert(`Evaluation Error: ${e.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '📋 Run Evaluation';
    }
}

// ── Utilities ─────────────────────────────────────────────────────
function resetPipeline() {
    document.querySelectorAll(".pipeline-step").forEach((el) => {
        el.classList.remove("running", "done", "error");
        el.classList.add("idle");
        const statusEl = el.querySelector(".step-status");
        if (statusEl) statusEl.textContent = "Waiting...";
    });
    els.ptpBody().innerHTML = "";
    els.inventoryBody().innerHTML = "";
    els.reportContent().textContent = "";
    const findingsEl = document.getElementById("bottleneck-findings");
    if (findingsEl) findingsEl.textContent = "";
}

function updateTimer() {
    if (startTime) {
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        els.timer().textContent = `⏱ ${elapsed}s`;
    }
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

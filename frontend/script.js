// ---------------------------------------------------------------------------
// Backend API base URL.
//
// IMPORTANT: update this before deploying the frontend anywhere other than
// your own machine. Pointing this at "localhost" from a deployed frontend
// (e.g. Vercel) is the #1 cause of ERR_CONNECTION_REFUSED — the visitor's
// browser has no Flask server running on ITS localhost.
//   - Local dev:      http://127.0.0.1:5000
//   - Deployed API:   https://your-backend.onrender.com  (etc.)
// ---------------------------------------------------------------------------
const API_BASE = (() => {
  if (location.hostname === "localhost" || location.hostname === "127.0.0.1") {
    return "http://127.0.0.1:5000";
  }
  // TODO: replace with your deployed backend URL
  return "https://customer-churn-prediction-beryl-six.vercel.app";
})();

const form = document.getElementById("churnForm");
const runBtn = document.getElementById("runBtn");
const resetBtn = document.getElementById("resetBtn");

const gaugeFill = document.getElementById("gaugeFill");
const gaugeNeedle = document.getElementById("gaugeNeedle");
const gaugeValue = document.getElementById("gaugeValue");

const verdictLabel = document.getElementById("verdictLabel");
const verdictDetail = document.getElementById("verdictDetail");
const driversBox = document.getElementById("driversBox");
const driversList = document.getElementById("driversList");

const logBox = document.getElementById("logBox");
const apiStatusDot = document.getElementById("apiStatusDot");
const apiStatusText = document.getElementById("apiStatusText");

const GAUGE_CIRCUMFERENCE = 314.159; // path length of the half-circle arc

function log(message, tag = "SYS", cls = "") {
  const line = document.createElement("div");
  line.className = `log-line ${cls}`;
  line.innerHTML = `<span class="log-tag">${tag}</span><span>${message}</span>`;
  logBox.appendChild(line);
  logBox.scrollTop = logBox.scrollHeight;
}

function setGauge(probability) {
  // probability: 0..1
  const offset = GAUGE_CIRCUMFERENCE * (1 - probability);
  gaugeFill.style.strokeDashoffset = offset;

  // needle sweeps from -90deg (0%) to +90deg (100%)
  const angle = -90 + probability * 180;
  gaugeNeedle.style.transform = `rotate(${angle}deg)`;

  let color = "var(--green)";
  if (probability >= 0.66) color = "var(--red)";
  else if (probability >= 0.33) color = "var(--amber)";
  gaugeFill.style.stroke = color;

  gaugeValue.textContent = `${Math.round(probability * 100)}%`;
}

function resetGauge() {
  gaugeFill.style.stroke = "var(--text-faint)";
  gaugeFill.style.strokeDashoffset = GAUGE_CIRCUMFERENCE;
  gaugeNeedle.style.transform = "rotate(-90deg)";
  gaugeValue.textContent = "--";
}

function renderResult(data) {
  setGauge(data.churn_probability);

  verdictLabel.classList.remove("at-risk", "retained");
  if (data.will_churn) {
    verdictLabel.textContent = "AT-RISK SUBSCRIBER";
    verdictLabel.classList.add("at-risk");
    verdictDetail.textContent =
      `Model estimates a ${Math.round(data.churn_probability * 100)}% chance this subscriber churns. ` +
      `Consider a retention offer targeting the drivers below.`;
  } else {
    verdictLabel.textContent = "LIKELY RETAINED";
    verdictLabel.classList.add("retained");
    verdictDetail.textContent =
      `Model estimates only a ${Math.round(data.churn_probability * 100)}% churn risk — ` +
      `this subscriber profile looks stable.`;
  }

  driversList.innerHTML = "";
  const maxImportance = Math.max(...data.top_drivers.map((d) => d.importance));
  data.top_drivers.forEach((d) => {
    const li = document.createElement("li");
    li.className = "driver-row";
    const pct = Math.round((d.importance / maxImportance) * 100);
    li.innerHTML = `
      <span class="driver-name">${d.feature}</span>
      <span class="driver-bar-track"><span class="driver-bar-fill" style="width:${pct}%"></span></span>
      <span class="driver-pct">${(d.importance * 100).toFixed(1)}%</span>
    `;
    driversList.appendChild(li);
  });
  driversBox.hidden = false;
}

function formToPayload() {
  const fd = new FormData(form);
  const payload = {};
  for (const [key, value] of fd.entries()) {
    if (key === "SeniorCitizen") {
      payload[key] = Number(value);
    } else if (key === "tenure") {
      payload[key] = Number(value);
    } else if (key === "MonthlyCharges" || key === "TotalCharges") {
      payload[key] = Number(value);
    } else {
      payload[key] = value;
    }
  }
  return payload;
}

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) throw new Error("bad status");
    apiStatusDot.classList.add("online");
    apiStatusText.textContent = "Model online";
    log("Connected to prediction API.", "SYS", "success");
  } catch (err) {
    apiStatusDot.classList.add("offline");
    apiStatusText.textContent = "API unreachable";
    log(
      `Could not reach backend at ${API_BASE}. Is the Flask server running?`,
      "ERR",
      "error"
    );
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const payload = formToPayload();

  runBtn.disabled = true;
  runBtn.querySelector(".btn-run-label").textContent = "Running…";
  log("Submitting subscriber profile for scoring…");

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Prediction failed.");
    }

    renderResult(data);
    log(`Prediction: ${data.prediction} (${Math.round(data.churn_probability * 100)}% risk).`, "OK", "success");
  } catch (err) {
    log(err.message, "ERR", "error");
    verdictLabel.textContent = "DIAGNOSTIC FAILED";
    verdictLabel.classList.remove("retained");
    verdictLabel.classList.add("at-risk");
    verdictDetail.textContent = err.message;
  } finally {
    runBtn.disabled = false;
    runBtn.querySelector(".btn-run-label").textContent = "Run diagnostic";
  }
});

resetBtn.addEventListener("click", () => {
  form.reset();
  resetGauge();
  verdictLabel.classList.remove("at-risk", "retained");
  verdictLabel.textContent = "AWAITING INPUT";
  verdictDetail.textContent = "Fill in the subscriber panel and run a diagnostic to score churn risk.";
  driversBox.hidden = true;
  log("Form reset.");
});

resetGauge();
checkHealth();

const API_BASE =
  window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost"
    ? "http://127.0.0.1:7860"
    : "https://Lammyde-email-spam-classifier.hf.space";

const emailText  = document.getElementById("emailText");
const btnCheck   = document.getElementById("btnCheck");
const btnHam     = document.getElementById("btnHam");
const btnSpam    = document.getElementById("btnSpam");
const resultLine = document.getElementById("resultLine");
const confBar    = document.getElementById("confBar");
const errBox     = document.getElementById("errBox");

// ── Helpers ───────────────────────────────────────────────────────────────────

function setError(msg = "") {
  errBox.textContent = msg;
}

function setLoading(btn, isLoading, loadingText, normalText) {
  btn.disabled    = isLoading;
  btn.textContent = isLoading ? loadingText : normalText;
}

function resetResultUI() {
  resultLine.textContent      = "Result: —";
  resultLine.style.color      = "#e7eefc";
  confBar.style.width          = "0%";
  confBar.style.backgroundColor = "#AB0B4B";
}

function setResult(label, spamProb, confidence) {
  const isSpam = String(label).toLowerCase() === "spam";
  const confPct = Number.isFinite(confidence)
    ? confidence
    : Math.round(spamProb * 10000) / 100;

  resultLine.innerHTML          = `Result: <strong>${String(label).toUpperCase()}</strong> — ${confPct}% confidence`;
  resultLine.style.color        = isSpam ? "#AB0B4B" : "#2ecc71";
  confBar.style.backgroundColor = isSpam ? "#AB0B4B" : "#2ecc71";
  confBar.style.width           = `${Math.max(0, Math.min(100, confPct))}%`;
}

// ── Random sample ─────────────────────────────────────────────────────────────

async function fetchSample(type) {
  setError("");
  const targetBtn  = type === "ham" ? btnHam : btnSpam;
  const normalText = type === "ham" ? "Random HAM" : "Random SPAM";

  setLoading(targetBtn, true, "Loading... (may take a moment)", normalText);

  const wakeupTimer = setTimeout(() => {
    setError("⏳ Server is waking up, please wait...");
  }, 5000);

  try {
    const res  = await fetch(`${API_BASE}/sample?label=${type}`);
    const data = await res.json();

    if (!res.ok) {
      setError(data.error || "Failed to fetch sample.");
      return;
    }

    emailText.value = data.text || "";
    resetResultUI();
    setError("");
  } catch (e) {
    setError("Backend not reachable. Please try again.");
  } finally {
    clearTimeout(wakeupTimer);
    setLoading(btnHam,  false, "", "Random HAM");
    setLoading(btnSpam, false, "", "Random SPAM");
  }
}

btnHam.addEventListener("click",  () => fetchSample("ham"));
btnSpam.addEventListener("click", () => fetchSample("spam"));

// ── Predict ───────────────────────────────────────────────────────────────────

btnCheck.addEventListener("click", async () => {
  setError("");
  const text = (emailText.value || "").trim();

  if (!text) {
    setError("Paste an email first.");
    return;
  }

  setLoading(btnCheck, true, "Checking... (may take a moment)", "Check");

  const wakeupTimer = setTimeout(() => {
    setError("⏳ Server is waking up, please wait...");
  }, 5000);

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text }),
    });

    const data = await res.json();

    if (!res.ok) {
      setError(data.error || "Prediction failed.");
      return;
    }

    const label           = data.label ?? data.prediction ?? "ham";
    const spamProbability = Number(data.spam_probability ?? data.probability ?? 0);
    const confidence      = Number(data.confidence ?? 0);

    setError("");
    setResult(label, spamProbability, confidence);
  } catch (e) {
    setError("Backend not reachable. Please try again.");
  } finally {
    clearTimeout(wakeupTimer);
    setLoading(btnCheck, false, "", "Check");
  }
});

// ── Evaluation Metrics ────────────────────────────────────────────────────────
// Formulas per Chapter 3:
//   Accuracy  : (TP + TN) / (TP + TN + FP + FN)
//   Precision : TP / (TP + FP)
//   Recall    : TP / (TP + FN)
//   F1-Score  : 2 × (Precision × Recall) / (Precision + Recall)
//   ROC-AUC   : Area under ROC curve across all classification thresholds

async function fetchMetric(metricKey, metricLabel) {
  const resultBox  = document.getElementById("metricResult");
  const nameEl     = document.getElementById("metricName");
  const valueEl    = document.getElementById("metricValue");
  const formulaEl  = document.getElementById("metricFormula");

  // Show loading state
  resultBox.style.display  = "block";
  nameEl.textContent       = metricLabel;
  valueEl.textContent      = "⏳";
  formulaEl.textContent    = "";

  // Disable all metric buttons while fetching
  const metricBtns = document.querySelectorAll(".metric-btn");
  metricBtns.forEach(b => b.disabled = true);

  try {
    const res  = await fetch(`${API_BASE}/metrics/${metricKey}`);
    const data = await res.json();

    if (!res.ok || data.error) {
      nameEl.textContent   = metricLabel;
      valueEl.textContent  = "—";
      formulaEl.textContent = data.error || "Could not retrieve metric.";
      return;
    }

    // Display value as percentage
    nameEl.textContent    = data.metric;
    valueEl.textContent   = (data.value * 100).toFixed(2) + "%";
    formulaEl.textContent = "Formula: " + data.formula;

  } catch (e) {
    nameEl.textContent    = metricLabel;
    valueEl.textContent   = "—";
    formulaEl.textContent = "Backend not reachable. Please try again.";
  } finally {
    metricBtns.forEach(b => b.disabled = false);
  }
}

const API_BASE =
  window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost"
    ? "http://127.0.0.1:8000"
    : "https://final-year-project-production-0aa4.up.railway.app/";

const emailText = document.getElementById("emailText");
const btnCheck  = document.getElementById("btnCheck");
const btnHam    = document.getElementById("btnHam");
const btnSpam   = document.getElementById("btnSpam");
const resultLine = document.getElementById("resultLine");
const confBar   = document.getElementById("confBar");
const errBox    = document.getElementById("errBox");

function setError(msg = "") {
  errBox.textContent = msg;
}

function setLoading(btn, isLoading, loadingText, normalText) {
  btn.disabled = isLoading;
  btn.textContent = isLoading ? loadingText : normalText;
}

function resetResultUI() {
  resultLine.textContent = "Result: —";
  resultLine.style.color = "#e7eefc";
  confBar.style.width = "0%";
  confBar.style.backgroundColor = "#AB0B4B";
}

function setResult(label, spamProb, confidence) {
  const isSpam = String(label).toLowerCase() === "spam";
  const confPct = Number.isFinite(confidence)
    ? confidence
    : Math.round(spamProb * 10000) / 100;

  resultLine.innerHTML = `Result: <strong>${String(label).toUpperCase()}</strong> — ${confPct}% confidence`;
  resultLine.style.color = isSpam ? "#AB0B4B" : "#2ecc71";
  confBar.style.backgroundColor = isSpam ? "#AB0B4B" : "#2ecc71";
  confBar.style.width = `${Math.max(0, Math.min(100, confPct))}%`;
}

async function fetchSample(type) {
  setError("");
  const targetBtn = type === "ham" ? btnHam : btnSpam;
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
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
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
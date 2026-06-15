const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const ERR_BACKEND_DOWN =
  "Layanan belum siap. Coba lagi sebentar.";
const ERR_EXTRACT_TIMEOUT =
  "Proses terlalu lama. Coba gunakan gambar yang lebih jelas atau input manual.";
const EXTRACT_TIMEOUT_MS = 60_000;

async function _fetch(path, options = {}) {
  let res;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, options);
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new Error(ERR_EXTRACT_TIMEOUT);
    }
    throw new Error(ERR_BACKEND_DOWN);
  }
  if (!res.ok) {
    let detail = `Error ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch { /* ignore */ }
    throw new Error(detail);
  }
  return res.json();
}

export const healthCheck = () => _fetch("/health");

export const extractTransaction = (file, selectedType = "auto") => {
  const typeMap = {
    "receipt":    "receipt",
    "screenshot": "screenshot",
    "auto":       "auto",
  };
  const backendType = typeMap[selectedType] ?? "auto";

  const form = new FormData();
  form.append("file", file);
  form.append("selected_type", backendType);

  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), EXTRACT_TIMEOUT_MS);

  return _fetch("/extract", {
    method: "POST",
    body: form,
    signal: controller.signal,
  }).finally(() => window.clearTimeout(timeoutId));
};

export const getTransactions = () => _fetch("/transactions");

export const saveTransaction = (payload) =>
  _fetch("/transactions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

export const updateTransactionCategory = (id, category) =>
  _fetch(`/transactions/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ category }),
  });

export const deleteTransactionById = (id) =>
  _fetch(`/transactions/${id}`, { method: "DELETE" });

export const deleteTransactions = () =>
  _fetch("/transactions", { method: "DELETE" });

export const getInsights = (monthlyIncome = 0) =>
  _fetch(`/insights${monthlyIncome > 0 ? `?monthly_income=${monthlyIncome}` : ""}`);

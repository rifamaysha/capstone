import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  BarChart2,
  Calendar,
  RefreshCw,
  ShoppingBag,
  TrendingUp,
  Upload,
  Wallet,
  PiggyBank,
  BadgeDollarSign,
  PencilLine,
  Plus,
  Trash2,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import Header from "../components/Header.jsx";
import MetricCard from "../components/MetricCard.jsx";
import EmptyState from "../components/EmptyState.jsx";
import TransactionTable from "../components/TransactionTable.jsx";
import { getInsights } from "../api/client.js";
import { chartColors, metricTone } from "../styles/theme.js";

const INCOME_STORAGE_KEY = "smart_expense_income_entries";

function formatRpShort(value) {
  const n = Number(value || 0);
  if (n >= 1_000_000) return `Rp ${(n / 1_000_000).toFixed(1)}jt`;
  if (n >= 1_000) return `Rp ${(n / 1_000).toFixed(0)}rb`;
  return `Rp ${n.toLocaleString("id-ID")}`;
}

function formatRp(value) {
  return `Rp ${Number(value || 0).toLocaleString("id-ID")}`;
}

function formatDate(isoString) {
  const d = new Date(isoString);
  return d.toLocaleDateString("id-ID", { day: "numeric", month: "short", year: "numeric" });
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#fff",
      border: "1px solid var(--color-border)",
      borderRadius: 10,
      padding: "10px 12px",
      fontSize: 12,
      boxShadow: "var(--shadow)",
    }}>
      <div style={{ fontWeight: 800, marginBottom: 4 }}>{label}</div>
      <div style={{ color: "var(--color-primary)", fontWeight: 800 }}>
        {formatRp(payload[0].value)}
      </div>
    </div>
  );
}

// ── Date helpers for trend chart ────────────────────────────────────────────
function toYMD(date) {
  // Format JS Date → "YYYY-MM-DD" using LOCAL time (avoids UTC off-by-one).
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

function fromYMD(ymd) {
  // Parse "YYYY-MM-DD" as local Date (new Date("YYYY-MM-DD") parses as UTC).
  if (!ymd || ymd.length < 10) return null;
  const [y, m, d] = ymd.slice(0, 10).split("-").map(Number);
  if (!y || !m || !d) return null;
  return new Date(y, m - 1, d);
}

function addDays(date, n) {
  const d = new Date(date);
  d.setDate(d.getDate() + n);
  return d;
}

function formatTickDate(ymd) {
  const d = fromYMD(ymd);
  if (!d) return ymd;
  return d.toLocaleDateString("id-ID", { day: "numeric", month: "short" });
}

function TrendTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const item = payload[0].payload;
  const d = fromYMD(item.date);
  const label = d
    ? d.toLocaleDateString("id-ID", { weekday: "short", day: "numeric", month: "short", year: "numeric" })
    : item.date;
  return (
    <div style={{
      background: "#fff",
      border: "1px solid var(--color-border)",
      borderRadius: 10,
      padding: "10px 12px",
      fontSize: 12,
      boxShadow: "var(--shadow)",
    }}>
      <div style={{ fontWeight: 800, marginBottom: 4 }}>{label}</div>
      <div style={{ color: "var(--color-primary)", fontWeight: 800 }}>
        {formatRp(item.total)}
      </div>
      {item.count > 0 && (
        <div style={{ color: "var(--color-muted)", marginTop: 2 }}>
          {item.count} transaksi
        </div>
      )}
    </div>
  );
}

function IncomeModal({ entries, onAdd, onDelete, onClose }) {
  const [inputValue, setInputValue] = useState("");
  const [label, setLabel] = useState("");
  const [showHistory, setShowHistory] = useState(false);

  const handleAdd = () => {
    const parsed = Number(String(inputValue || "").replace(/[^\d.]/g, "")) || 0;
    if (parsed <= 0) return;
    onAdd({
      id: Date.now(),
      amount: parsed,
      label: label.trim() || "Pemasukan",
      date: new Date().toISOString(),
    });
    setInputValue("");
    setLabel("");
  };

  const total = entries.reduce((s, e) => s + e.amount, 0);

  return (
    <div
      style={{
        position: "fixed", inset: 0,
        background: "rgba(7,21,40,0.35)",
        zIndex: 1000,
        display: "flex", alignItems: "center", justifyContent: "center",
        padding: 20,
      }}
      onClick={onClose}
    >
      <div
        style={{
          background: "var(--color-card)",
          borderRadius: "var(--radius-lg)",
          padding: "28px 28px 24px",
          width: "100%", maxWidth: 460,
          boxShadow: "var(--shadow)",
          maxHeight: "90vh",
          overflowY: "auto",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div style={{ fontSize: 20, fontWeight: 800, marginBottom: 4 }}>
          Tambah Pemasukan
        </div>
        <div style={{ color: "var(--color-muted)", fontSize: 14, marginBottom: 20 }}>
          Bisa ditambahkan berkali-kali, misalnya uang jajan awal bulan + uang tambahan.
        </div>

        {/* Input form */}
        <div style={{ display: "flex", flexDirection: "column", gap: 10, marginBottom: 16 }}>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">Keterangan (opsional)</label>
            <input
              className="form-input"
              type="text"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder="Contoh: Uang jajan Juni, Bonus, dll"
              onKeyDown={(e) => e.key === "Enter" && handleAdd()}
            />
          </div>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">Jumlah (Rp)</label>
            <div style={{ display: "flex", gap: 8 }}>
              <input
                className="form-input"
                type="number"
                min="0"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Contoh: 500000"
                autoFocus
                onKeyDown={(e) => e.key === "Enter" && handleAdd()}
                style={{ flex: 1 }}
              />
              <button className="btn btn-primary" onClick={handleAdd} style={{ whiteSpace: "nowrap" }}>
                <Plus size={16} />
                Tambah
              </button>
            </div>
          </div>
        </div>

        {/* Total */}
        {entries.length > 0 && (
          <div style={{
            background: "var(--color-success-soft)",
            border: "1px solid #b6f0df",
            borderRadius: "var(--radius)",
            padding: "12px 16px",
            marginBottom: 12,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}>
            <span style={{ fontSize: 13, color: "var(--color-success)", fontWeight: 600 }}>
              Total Pemasukan
            </span>
            <span style={{ fontSize: 18, fontWeight: 900, color: "var(--color-success)" }}>
              {formatRp(total)}
            </span>
          </div>
        )}

        {/* History toggle */}
        {entries.length > 0 && (
          <div>
            <button
              onClick={() => setShowHistory(!showHistory)}
              style={{
                background: "none", border: "none", cursor: "pointer",
                color: "var(--color-muted)", fontSize: 13, fontWeight: 600,
                display: "flex", alignItems: "center", gap: 4,
                padding: "4px 0", marginBottom: showHistory ? 8 : 0,
              }}
            >
              {showHistory ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              {entries.length} entri pemasukan
            </button>

            {showHistory && (
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {entries.map((entry) => (
                  <div
                    key={entry.id}
                    style={{
                      display: "flex", alignItems: "center",
                      justifyContent: "space-between",
                      background: "var(--color-background)",
                      borderRadius: "var(--radius-sm)",
                      padding: "10px 12px",
                    }}
                  >
                    <div>
                      <div style={{ fontWeight: 700, fontSize: 14 }}>{entry.label}</div>
                      <div style={{ fontSize: 12, color: "var(--color-muted)" }}>{formatDate(entry.date)}</div>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      <span style={{ fontWeight: 800, fontSize: 15 }}>{formatRp(entry.amount)}</span>
                      <button
                        onClick={() => onDelete(entry.id)}
                        style={{
                          background: "none", border: "none", cursor: "pointer",
                          color: "var(--color-danger)", padding: 4, borderRadius: 6,
                          display: "flex", alignItems: "center",
                        }}
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 20 }}>
          <button className="btn btn-ghost btn-sm" onClick={onClose}>
            Tutup
          </button>
        </div>
      </div>
    </div>
  );
}

export default function Dashboard({ onNavigate }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [incomeEntries, setIncomeEntries] = useState(() => {
    try {
      const saved = localStorage.getItem(INCOME_STORAGE_KEY);
      const parsed = saved ? JSON.parse(saved) : [];
      // Bug fix: jika legacy localStorage menyimpan nilai non-array
      // (mis. dulu hanya total income sebagai number), .reduce() akan crash.
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  });
  const [showIncomeModal, setShowIncomeModal] = useState(false);

  // ── Trend date range state (default: 7 hari terakhir) ─────────────────────
  const [trendStart, setTrendStart] = useState(() => toYMD(addDays(new Date(), -6)));
  const [trendEnd, setTrendEnd] = useState(() => toYMD(new Date()));

  const totalIncome = incomeEntries.reduce((s, e) => s + e.amount, 0);

  const load = async (income = totalIncome) => {
    setLoading(true);
    setError("");
    try {
      setData(await getInsights(income));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const saveEntries = (entries) => {
    setIncomeEntries(entries);
    localStorage.setItem(INCOME_STORAGE_KEY, JSON.stringify(entries));
    const newTotal = entries.reduce((s, e) => s + e.amount, 0);
    load(newTotal);
  };

  const handleAddEntry = (entry) => {
    saveEntries([...incomeEntries, entry]);
  };

  const handleDeleteEntry = (id) => {
    saveEntries(incomeEntries.filter((e) => e.id !== id));
  };

  const summary = data?.summary || {};
  const breakdown = data?.category_breakdown || [];
  const recent = data?.recent_transactions || [];
  const hasData = (summary.transaction_count || 0) > 0;

  const totalExpense = Number(summary.total_expense || 0);
  const remainingBalance = totalIncome > 0 ? Math.max(0, totalIncome - totalExpense) : null;
  const rawUsedPercent = totalIncome > 0 ? Math.round((totalExpense / totalIncome) * 100) : 0;
  const usedPercent = Math.min(100, rawUsedPercent); // capped for progress bar
  const isOverBudget = totalIncome > 0 && totalExpense > totalIncome;

  // ── Trend chart data: filter daily_expenses ke range, isi tanggal kosong dengan 0 ──
  const dailyExpenses = data?.daily_expenses || [];
  const todayStr = toYMD(new Date());

  const trendSeries = useMemo(() => {
    const s = fromYMD(trendStart);
    const e = fromYMD(trendEnd);
    if (!s || !e) return [];
    // Auto-swap kalau user pilih start > end
    const [from, to] = s <= e ? [s, e] : [e, s];

    // Map tanggal → {total, count} dari data backend
    const map = new Map();
    for (const item of dailyExpenses) {
      map.set(item.date, {
        total: Number(item.total) || 0,
        count: Number(item.count) || 0,
      });
    }

    // Generate setiap hari dalam range, fill 0 kalau tidak ada transaksi
    const out = [];
    for (let d = new Date(from); d <= to; d = addDays(d, 1)) {
      const key = toYMD(d);
      const hit = map.get(key);
      out.push({
        date: key,
        total: hit?.total || 0,
        count: hit?.count || 0,
      });
    }
    return out;
  }, [dailyExpenses, trendStart, trendEnd]);

  const trendTotal = trendSeries.reduce((s, x) => s + x.total, 0);
  const trendAvg = trendSeries.length > 0 ? trendTotal / trendSeries.length : 0;
  const trendDaysWithSpending = trendSeries.filter((x) => x.total > 0).length;

  const applyQuickRange = (days) => {
    setTrendEnd(toYMD(new Date()));
    setTrendStart(toYMD(addDays(new Date(), -(days - 1))));
  };

  return (
    <>
      {showIncomeModal && (
        <IncomeModal
          entries={incomeEntries}
          onAdd={handleAddEntry}
          onDelete={handleDeleteEntry}
          onClose={() => setShowIncomeModal(false)}
        />
      )}

      <Header
        title="Dashboard"
        subtitle="Pantau pengeluaran dari struk dan screenshot pembayaran."
        actions={
          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn btn-secondary btn-sm" onClick={() => setShowIncomeModal(true)}>
              <BadgeDollarSign size={15} />
              {totalIncome > 0 ? formatRpShort(totalIncome) : "Tambah Pemasukan"}
            </button>
            <button className="btn btn-ghost btn-sm" onClick={() => load()} disabled={loading}>
              <RefreshCw size={15} />
              Perbarui
            </button>
          </div>
        }
      />

      <div className="page-body">
        {error && <div className="alert alert-error mb-4">{error}</div>}

        {loading ? (
          <div className="loading-bar">
            <span className="spinner" />
            Memuat dashboard...
          </div>
        ) : !hasData ? (
          <>
            {totalIncome === 0 && (
              <div className="alert" style={{
                background: "var(--color-primary-soft)",
                color: "var(--color-primary)",
                border: "1px solid #d0ceff",
                marginBottom: 16,
                display: "flex", alignItems: "center", gap: 10,
              }}>
                <PiggyBank size={18} />
                <span>
                  Belum ada pemasukan.{" "}
                  <button onClick={() => setShowIncomeModal(true)} style={{
                    background: "none", border: "none",
                    color: "var(--color-primary)", fontWeight: 700,
                    cursor: "pointer", padding: 0, textDecoration: "underline",
                  }}>
                    Tambah sekarang
                  </button>{" "}
                  untuk mulai melacak sisa uangmu.
                </span>
              </div>
            )}
            <div className="card card-body">
              <EmptyState
                title="Belum ada transaksi"
                description="Upload gambar transaksi pertama untuk mulai melihat ringkasan."
                action={
                  <button className="btn btn-primary" onClick={() => onNavigate("upload")}>
                    <Upload size={16} />
                    Upload & Proses
                  </button>
                }
              />
            </div>
          </>
        ) : (
          <>
            {/* ── Sisa Uang Banner ── */}
            {totalIncome > 0 && (
              <div style={{
                background: isOverBudget ? "var(--color-danger-soft)" : "var(--color-success-soft)",
                border: `1px solid ${isOverBudget ? "#fbc5d0" : "#b6f0df"}`,
                borderRadius: "var(--radius-lg)",
                padding: "18px 24px",
                marginBottom: 24,
                display: "flex", alignItems: "center",
                justifyContent: "space-between",
                flexWrap: "wrap", gap: 16,
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
                  <div style={{
                    background: isOverBudget ? "#ef456522" : "#12a77a22",
                    borderRadius: "50%", width: 44, height: 44,
                    display: "flex", alignItems: "center", justifyContent: "center",
                  }}>
                    <PiggyBank size={22} color={isOverBudget ? "var(--color-danger)" : "var(--color-success)"} />
                  </div>
                  <div>
                    <div style={{
                      fontSize: 13, fontWeight: 600, marginBottom: 2,
                      color: isOverBudget ? "var(--color-danger)" : "var(--color-success)",
                    }}>
                      {isOverBudget
                        ? <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}><AlertTriangle size={13} />Pengeluaran Melebihi Pemasukan!</span>
                        : <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}><Wallet size={13} />Sisa Uang Kamu</span>}
                    </div>
                    <div style={{
                      fontSize: 26, fontWeight: 900, letterSpacing: "-0.5px",
                      color: isOverBudget ? "var(--color-danger)" : "var(--color-success)",
                    }}>
                      {isOverBudget
                        ? `- ${formatRp(totalExpense - totalIncome)}`
                        : formatRp(remainingBalance)}
                    </div>
                    <div style={{ fontSize: 13, color: "var(--color-muted)", marginTop: 2 }}>
                      dari {incomeEntries.length} pemasukan · total {formatRp(totalIncome)} · terpakai {usedPercent}%
                    </div>
                  </div>
                </div>

                {/* Progress bar */}
                <div style={{ flex: "1 1 200px", minWidth: 180, maxWidth: 320 }}>
                  <div style={{ fontSize: 12, color: "var(--color-muted)", marginBottom: 6 }}>
                    {formatRp(totalExpense)} dari {formatRp(totalIncome)}
                  </div>
                  <div style={{ height: 10, background: "rgba(0,0,0,0.08)", borderRadius: 999, overflow: "hidden" }}>
                    <div style={{
                      height: "100%",
                      width: `${Math.min(100, usedPercent)}%`,
                      background: isOverBudget ? "var(--color-danger)" : "var(--color-success)",
                      borderRadius: 999,
                      transition: "width 0.5s ease",
                    }} />
                  </div>
                  <div style={{ fontSize: 12, color: "var(--color-muted)", marginTop: 4 }}>
                    {isOverBudget
                      ? `Kelebihan ${rawUsedPercent - 100}% dari pemasukan`
                      : `Sisa ${100 - usedPercent}% belum terpakai`}
                  </div>
                </div>

                <button
                  className="btn btn-ghost btn-sm"
                  onClick={() => setShowIncomeModal(true)}
                  style={{ alignSelf: "flex-start" }}
                >
                  <Plus size={14} />
                  Tambah
                </button>
              </div>
            )}

            {/* No income prompt */}
            {totalIncome === 0 && (
              <div style={{
                background: "var(--color-primary-soft)",
                border: "1px solid #d0ceff",
                borderRadius: "var(--radius)",
                padding: "14px 18px", marginBottom: 20,
                display: "flex", alignItems: "center",
                justifyContent: "space-between", gap: 12, flexWrap: "wrap",
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <PiggyBank size={18} color="var(--color-primary)" />
                  <span style={{ fontSize: 14, color: "var(--color-primary)", fontWeight: 600 }}>
                    Tambahkan pemasukan / uang jajan untuk melihat sisa uangmu.
                  </span>
                </div>
                <button className="btn btn-primary btn-sm" onClick={() => setShowIncomeModal(true)}>
                  <Plus size={14} />
                  Tambah Pemasukan
                </button>
              </div>
            )}

            {/* ── Metric Cards ── */}
            <div className="metric-grid">
              <MetricCard
                label="Total Pengeluaran"
                value={formatRp(summary.total_expense)}
                icon={Wallet}
                {...metricTone.primary}
              />
              {totalIncome > 0 ? (
                <MetricCard
                  label="Sisa Uang"
                  value={isOverBudget ? `- ${formatRp(totalExpense - totalIncome)}` : formatRp(remainingBalance)}
                  sub={isOverBudget ? "Melebihi pemasukan!" : `${100 - usedPercent}% belum terpakai`}
                  icon={PiggyBank}
                  {...(isOverBudget ? metricTone.danger : metricTone.success)}
                />
              ) : (
                <MetricCard
                  label="Jumlah Transaksi"
                  value={summary.transaction_count || 0}
                  sub="transaksi tersimpan"
                  icon={ShoppingBag}
                  {...metricTone.success}
                />
              )}
              <MetricCard
                label="Kategori Terbesar"
                value={breakdown[0]?.category_display || summary.top_category_display || "-"}
                sub={breakdown[0] ? `${breakdown[0].percentage}% pengeluaran` : ""}
                icon={BarChart2}
                {...metricTone.warning}
              />
              <MetricCard
                label="Rata-rata Transaksi"
                value={formatRp(summary.average_transaction)}
                icon={TrendingUp}
                {...metricTone.violet}
              />
            </div>

            {/* ── Tren Pengeluaran Harian ── */}
            <div className="card card-body section-gap">
              <div style={{
                display: "flex", flexWrap: "wrap",
                alignItems: "flex-start", justifyContent: "space-between",
                gap: 16, marginBottom: 16,
              }}>
                <div>
                  <div className="card-title" style={{ marginBottom: 4 }}>Tren Pengeluaran Harian</div>
                  <div style={{ color: "var(--color-muted)", fontSize: 14 }}>
                    Total pengeluaran per hari dalam rentang tanggal yang kamu pilih.
                  </div>
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                  <button className="btn btn-ghost btn-sm" onClick={() => applyQuickRange(7)}>7 hari</button>
                  <button className="btn btn-ghost btn-sm" onClick={() => applyQuickRange(14)}>14 hari</button>
                  <button className="btn btn-ghost btn-sm" onClick={() => applyQuickRange(30)}>30 hari</button>
                </div>
              </div>

              <div style={{
                display: "flex", flexWrap: "wrap",
                gap: 14, marginBottom: 18, alignItems: "center",
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <Calendar size={15} color="var(--color-muted)" />
                  <span style={{ fontSize: 13, color: "var(--color-muted)", fontWeight: 600 }}>Dari</span>
                  <input
                    type="date"
                    className="form-input"
                    value={trendStart}
                    max={todayStr}
                    onChange={(e) => setTrendStart(e.target.value)}
                    style={{ width: "auto", padding: "8px 10px" }}
                  />
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ fontSize: 13, color: "var(--color-muted)", fontWeight: 600 }}>Sampai</span>
                  <input
                    type="date"
                    className="form-input"
                    value={trendEnd}
                    max={todayStr}
                    onChange={(e) => setTrendEnd(e.target.value)}
                    style={{ width: "auto", padding: "8px 10px" }}
                  />
                </div>
              </div>

              {trendSeries.length > 0 && trendTotal > 0 ? (
                <>
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart data={trendSeries} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                      <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" vertical={false} />
                      <XAxis
                        dataKey="date"
                        tick={{ fontSize: 11, fill: "var(--color-muted)" }}
                        tickLine={false} axisLine={false}
                        tickFormatter={formatTickDate}
                        interval="preserveStartEnd"
                        minTickGap={24}
                      />
                      <YAxis
                        tick={{ fontSize: 11, fill: "var(--color-muted)" }}
                        tickLine={false} axisLine={false} width={56}
                        tickFormatter={formatRpShort}
                      />
                      <Tooltip content={<TrendTooltip />} />
                      <Line
                        type="monotone"
                        dataKey="total"
                        stroke="var(--color-primary)"
                        strokeWidth={2.5}
                        dot={{ r: 3, fill: "var(--color-primary)" }}
                        activeDot={{ r: 5 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>

                  <div style={{
                    display: "flex", flexWrap: "wrap", gap: 28,
                    marginTop: 14, paddingTop: 14,
                    borderTop: "1px solid var(--color-border)",
                  }}>
                    <div>
                      <div style={{ fontSize: 12, color: "var(--color-muted)", fontWeight: 600 }}>Total Periode</div>
                      <div style={{ fontSize: 16, fontWeight: 850, marginTop: 2 }}>{formatRp(trendTotal)}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: 12, color: "var(--color-muted)", fontWeight: 600 }}>Rata-rata / hari</div>
                      <div style={{ fontSize: 16, fontWeight: 850, marginTop: 2 }}>{formatRp(trendAvg)}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: 12, color: "var(--color-muted)", fontWeight: 600 }}>Hari Berbelanja</div>
                      <div style={{ fontSize: 16, fontWeight: 850, marginTop: 2 }}>
                        {trendDaysWithSpending} dari {trendSeries.length} hari
                      </div>
                    </div>
                  </div>
                </>
              ) : (
                <div style={{
                  padding: "44px 20px", textAlign: "center",
                  color: "var(--color-muted)", fontSize: 14,
                }}>
                  Tidak ada pengeluaran pada rentang tanggal ini.
                </div>
              )}
            </div>

            <div className="two-col section-gap">
              <div className="card card-body">
                <div className="card-title">Pengeluaran per Kategori</div>
                {breakdown.length ? (
                  <ResponsiveContainer width="100%" height={240}>
                    <BarChart data={breakdown} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                      <XAxis
                        dataKey="category_display"
                        tick={{ fontSize: 11, fill: "var(--color-muted)" }}
                        tickLine={false} axisLine={false} interval={0}
                        tickFormatter={(v) => v.length > 10 ? `${v.slice(0, 10)}...` : v}
                      />
                      <YAxis
                        tick={{ fontSize: 11, fill: "var(--color-muted)" }}
                        tickLine={false} axisLine={false} width={46}
                        tickFormatter={formatRpShort}
                      />
                      <Tooltip content={<ChartTooltip />} />
                      <Bar dataKey="total" radius={[8, 8, 0, 0]}>
                        {breakdown.map((_, i) => (
                          <Cell key={i} fill={chartColors[i % chartColors.length]} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div style={{ color: "var(--color-muted)" }}>Belum ada kategori.</div>
                )}
              </div>

              <div className="card card-body">
                <div className="card-title">Catatan Bulan Ini</div>
                <div className="card-subtitle">Ringkasan dari transaksi tersimpan.</div>
                {breakdown.slice(0, 5).map((item, index) => (
                  <div
                    key={item.category}
                    className="flex items-center justify-between"
                    style={{
                      padding: "11px 0",
                      borderBottom: index < Math.min(breakdown.length, 5) - 1
                        ? "1px solid var(--color-border)" : "none",
                    }}
                  >
                    <div className="flex items-center gap-3">
                      <span style={{
                        width: 10, height: 10, borderRadius: "50%",
                        background: chartColors[index % chartColors.length],
                      }} />
                      <span style={{ fontWeight: 750 }}>{item.category_display}</span>
                    </div>
                    <div className="text-right">
                      <div style={{ fontWeight: 850 }}>{formatRp(item.total)}</div>
                      <div style={{ color: "var(--color-muted)", fontSize: 13 }}>
                        {item.percentage}% - {item.count} transaksi
                      </div>
                    </div>
                  </div>
                ))}
                {summary.top_merchant && (
                  <div className="alert alert-info mt-4">
                    Toko yang paling sering tercatat: <strong>{summary.top_merchant}</strong>
                  </div>
                )}
              </div>
            </div>

            <div className="card card-body">
              <div className="flex items-center justify-between" style={{ marginBottom: 16 }}>
                <div>
                  <div className="card-title" style={{ marginBottom: 4 }}>Transaksi Terbaru</div>
                  <div style={{ color: "var(--color-muted)", fontSize: 14 }}>
                    Transaksi terakhir yang berhasil disimpan.
                  </div>
                </div>
                <button className="btn btn-secondary btn-sm" onClick={() => onNavigate("history")}>
                  Lihat Semua
                </button>
              </div>
              <TransactionTable transactions={recent.slice(0, 8)} />
            </div>
          </>
        )}
      </div>
    </>
  );
}

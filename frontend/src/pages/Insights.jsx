import { useEffect, useState } from "react";
import { AlertTriangle, BarChart2, LayoutDashboard, PiggyBank, RefreshCw, Store, TrendingDown, Upload } from "lucide-react";
import {
  Bar,
  BarChart,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import Header from "../components/Header.jsx";
import MetricCard from "../components/MetricCard.jsx";
import EmptyState from "../components/EmptyState.jsx";
import { getInsights } from "../api/client.js";
import { chartColors, metricTone } from "../styles/theme.js";

// Same key used by Dashboard — single source of truth for income data
const INCOME_STORAGE_KEY = "smart_expense_income_entries";

function getTotalIncomeFromStorage() {
  try {
    const saved = localStorage.getItem(INCOME_STORAGE_KEY);
    const entries = saved ? JSON.parse(saved) : [];
    return entries.reduce((s, e) => s + (Number(e.amount) || 0), 0);
  } catch {
    return 0;
  }
}

function formatRp(value) {
  return `Rp ${Number(value || 0).toLocaleString("id-ID")}`;
}

function formatRpShort(value) {
  const n = Number(value || 0);
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}jt`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}rb`;
  return `${n}`;
}

function TooltipBox({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "#fff", border: "1px solid var(--color-border)", borderRadius: 10, padding: "10px 12px", boxShadow: "var(--shadow)", fontSize: 12 }}>
      <div style={{ fontWeight: 800, marginBottom: 4 }}>{label}</div>
      <div style={{ color: "var(--color-primary)", fontWeight: 800 }}>{formatRp(payload[0].value)}</div>
    </div>
  );
}

function bucketName(key) {
  const names = {
    kebutuhan: "Kebutuhan",
    keinginan: "Keinginan",
    tabungan: "Tabungan / Investasi",
  };
  return names[key] || key;
}

export default function Insights({ onNavigate }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [totalIncome, setTotalIncome] = useState(getTotalIncomeFromStorage);

  const load = async (monthlyIncome = 0) => {
    setLoading(true);
    setError("");
    try {
      setData(await getInsights(monthlyIncome));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Re-read income from localStorage on mount so Insights always reflects
    // whatever was entered in Dashboard — no separate input needed here.
    const income = getTotalIncomeFromStorage();
    setTotalIncome(income);
    load(income);
  }, []);

  const summary = data?.summary || {};
  const breakdown = data?.category_breakdown || [];
  const recommendations = data?.recommendations || [];
  const toReview = data?.transactions_to_review || [];
  const budget = data?.budget_comparison || {};
  const hasData = (summary.transaction_count || 0) > 0;
  const potentialSaving = totalIncome > 0
    ? Math.max(0, totalIncome - Number(summary.total_expense || 0))
    : 0;

  return (
    <>
      <Header
        title="Insight Penghematan"
        subtitle="Lihat pola pengeluaran dan saran hemat dari transaksi yang tersimpan."
        actions={
          <button
            className="btn btn-ghost btn-sm"
            onClick={() => {
              const income = getTotalIncomeFromStorage();
              setTotalIncome(income);
              load(income);
            }}
            disabled={loading}
          >
            <RefreshCw size={15} />
            Perbarui
          </button>
        }
      />

      <div className="page-body">
        {error && <div className="alert alert-error mb-4">{error}</div>}

        {loading ? (
          <div className="loading-bar">
            <span className="spinner" />
            Memuat insight penghematan...
          </div>
        ) : !hasData ? (
          <div className="card card-body">
            <EmptyState
              title="Belum ada data pengeluaran"
              description="Simpan beberapa transaksi untuk melihat insight penghematan."
              action={
                <button className="btn btn-primary" onClick={() => onNavigate("upload")}>
                  <Upload size={16} />
                  Upload & Proses
                </button>
              }
            />
          </div>
        ) : (
          <>
            {/* Income info banner — read-only, sourced from Dashboard */}
            {totalIncome > 0 ? (
              <div style={{
                display: "flex", alignItems: "center", justifyContent: "space-between",
                flexWrap: "wrap", gap: 12,
                background: "var(--color-success-soft)",
                border: "1px solid #b6f0df",
                borderRadius: "var(--radius)",
                padding: "12px 18px",
                marginBottom: 4,
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <PiggyBank size={16} color="var(--color-success)" />
                  <span style={{ fontSize: 14, color: "var(--color-success)", fontWeight: 600 }}>
                    Pemasukan dari Dashboard:{" "}
                    <strong>{formatRp(totalIncome)}</strong>
                  </span>
                </div>
                <button
                  className="btn btn-ghost btn-sm"
                  onClick={() => onNavigate("dashboard")}
                  style={{ fontSize: 13 }}
                >
                  <LayoutDashboard size={14} />
                  Ubah di Dashboard
                </button>
              </div>
            ) : (
              <div style={{
                display: "flex", alignItems: "center", justifyContent: "space-between",
                flexWrap: "wrap", gap: 12,
                background: "var(--color-primary-soft)",
                border: "1px solid #d0ceff",
                borderRadius: "var(--radius)",
                padding: "12px 18px",
                marginBottom: 4,
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <PiggyBank size={16} color="var(--color-primary)" />
                  <span style={{ fontSize: 14, color: "var(--color-primary)", fontWeight: 600 }}>
                    Tambahkan pemasukan di Dashboard agar budget dan potensi hemat dapat dihitung.
                  </span>
                </div>
                <button
                  className="btn btn-primary btn-sm"
                  onClick={() => onNavigate("dashboard")}
                  style={{ fontSize: 13 }}
                >
                  <LayoutDashboard size={14} />
                  Ke Dashboard
                </button>
              </div>
            )}

            <div className="metric-grid">
              <MetricCard
                label="Total Pengeluaran"
                value={formatRp(summary.total_expense)}
                icon={BarChart2}
                {...metricTone.primary}
              />
              <MetricCard
                label="Kategori Terbesar"
                value={summary.top_category_display || "-"}
                icon={TrendingDown}
                {...metricTone.warning}
              />
              <MetricCard
                label="Potensi Hemat"
                value={totalIncome > 0 ? formatRp(potentialSaving) : "-"}
                icon={PiggyBank}
                {...metricTone.success}
              />
              <MetricCard
                label="Toko Sering Muncul"
                value={summary.top_merchant || "-"}
                icon={Store}
                {...metricTone.violet}
              />
            </div>

            <div className="two-col section-gap">
              <div className="card card-body">
                <div className="card-title">Pengeluaran per Kategori</div>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={breakdown} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                    <XAxis
                      dataKey="category_display"
                      tick={{ fontSize: 11, fill: "var(--color-muted)" }}
                      tickLine={false}
                      axisLine={false}
                      interval={0}
                      tickFormatter={(value) => (value.length > 10 ? `${value.slice(0, 10)}...` : value)}
                    />
                    <YAxis
                      tick={{ fontSize: 11, fill: "var(--color-muted)" }}
                      tickLine={false}
                      axisLine={false}
                      width={42}
                      tickFormatter={formatRpShort}
                    />
                    <Tooltip content={<TooltipBox />} />
                    <Bar dataKey="total" radius={[8, 8, 0, 0]}>
                      {breakdown.map((_, index) => (
                        <Cell key={index} fill={chartColors[index % chartColors.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="card card-body">
                <div className="card-title">Proporsi Pengeluaran</div>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={breakdown}
                      dataKey="total"
                      nameKey="category_display"
                      cx="50%"
                      cy="45%"
                      innerRadius={52}
                      outerRadius={82}
                      paddingAngle={4}
                      label={({ percent }) => `${(percent * 100).toFixed(1)}%`}
                      labelLine={false}
                    >
                      {breakdown.map((_, index) => (
                        <Cell key={index} fill={chartColors[index % chartColors.length]} />
                      ))}
                    </Pie>
                    <Legend
                      iconType="circle"
                      iconSize={9}
                      formatter={(value) => <span style={{ fontSize: 12, color: "var(--color-muted)" }}>{value}</span>}
                    />
                    <Tooltip formatter={(value) => [formatRp(value), "Total"]} contentStyle={{ borderRadius: 10, fontSize: 12 }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>

            {budget.buckets && (
              <div className="card card-body section-gap">
                <div className="card-title">Perbandingan Budget Bulanan</div>
                <div className="card-subtitle">Kebutuhan, keinginan, dan tabungan dibandingkan dengan pendapatan bulanan.</div>
                {Object.entries(budget.buckets).map(([key, info], index) => {
                  const actual = Number(info?.actual_ratio || 0);
                  const ideal = Number(info?.ideal_ratio || 0);
                  const actualAmount = Number(info?.actual || 0);
                  // For tabungan (savings), actual > ideal is good, not over-budget
                  const over = key !== "tabungan" && ideal > 0 && actual > ideal;
                  const width = Math.min(100, ideal > 0 ? (actual / ideal) * 100 : 0);
                  return (
                    <div key={key} style={{ padding: "12px 0", borderBottom: index < Object.keys(budget.buckets).length - 1 ? "1px solid var(--color-border)" : "none" }}>
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <div style={{ fontWeight: 850 }}>{bucketName(key)}</div>
                          <div style={{ color: "var(--color-muted)", fontSize: 13 }}>
                            {formatRp(actualAmount)} dari target {Math.round(ideal * 100)}%
                          </div>
                        </div>
                        <div style={{ fontWeight: 850, color: over ? "var(--color-danger)" : "var(--color-success)" }}>
                          {Math.round(actual * 100)}%
                        </div>
                      </div>
                      <div style={{ height: 10, background: "#eef2f7", borderRadius: 999, overflow: "hidden", marginTop: 10 }}>
                        <div
                          style={{
                            width: `${width}%`,
                            height: "100%",
                            borderRadius: 999,
                            background: over ? "var(--color-danger)" : "var(--color-success)",
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            <div className="insight-grid">
              <div className="card card-body section-gap">
                <div className="card-title">Rekomendasi Hemat</div>
                {recommendations.length ? (
                  <div style={{ display: "grid", gap: 10 }}>
                    {recommendations.map((item, index) => (
                      <div key={index} className="alert alert-info">
                        <strong>{item.message}</strong>
                        {item.detail && <div style={{ marginTop: 4 }}>{item.detail}</div>}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ color: "var(--color-muted)" }}>Belum ada rekomendasi khusus.</div>
                )}
              </div>

              <div className="card card-body section-gap">
                <div className="card-title">Transaksi yang Perlu Dicek</div>
                <div className="card-subtitle">Transaksi yang nominalnya terlihat tidak biasa.</div>
                {toReview.length ? (
                  toReview.slice(0, 6).map((item, index) => (
                    <div
                      key={item.id || index}
                      className="flex items-center justify-between gap-3"
                      style={{
                        padding: "12px 0",
                        borderBottom: index < Math.min(toReview.length, 6) - 1 ? "1px solid var(--color-border)" : "none",
                      }}
                    >
                      <div className="flex items-center gap-3" style={{ minWidth: 0 }}>
                        <span className="badge badge-yellow">
                          <AlertTriangle size={14} />
                        </span>
                        <div style={{ minWidth: 0 }}>
                          <div style={{ fontWeight: 850, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                            {item.merchant || "Tidak diketahui"}
                          </div>
                          <div style={{ color: "var(--color-muted)", fontSize: 13 }}>
                            {item.category_display} - {item.date || "-"}
                          </div>
                          {item.anomaly_reason && (
                            <div style={{ fontSize: 12, color: "var(--color-warning, #d97706)", marginTop: 2 }}>
                              {item.anomaly_reason}
                            </div>
                          )}
                        </div>
                      </div>
                      <div style={{ fontWeight: 850, whiteSpace: "nowrap" }}>{formatRp(item.amount)}</div>
                    </div>
                  ))
                ) : (
                  <div style={{ color: "var(--color-muted)" }}>Tidak ada transaksi yang perlu dicek.</div>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </>
  );
}

import { useEffect, useState } from "react";
import { AlertTriangle, BarChart2, PiggyBank, RefreshCw, Store, TrendingDown, Upload } from "lucide-react";
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
  const [income, setIncome] = useState("");
  const [submitting, setSubmitting] = useState(false);

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
    load();
  }, []);

  const handleIncome = async (event) => {
    event.preventDefault();
    const parsed = Number(String(income || "").replace(/[^\d.]/g, "")) || 0;
    setSubmitting(true);
    await load(parsed);
    setSubmitting(false);
  };

  const summary = data?.summary || {};
  const breakdown = data?.category_breakdown || [];
  const recommendations = data?.recommendations || [];
  const toReview = data?.transactions_to_review || [];
  const budget = data?.budget_comparison || {};
  const hasData = (summary.transaction_count || 0) > 0;
  const potentialSaving = budget.monthly_income > 0
    ? Math.max(0, Number(budget.monthly_income) - Number(summary.total_expense || 0))
    : 0;

  return (
    <>
      <Header
        title="Insight Penghematan"
        subtitle="Lihat pola pengeluaran dan saran hemat dari transaksi yang tersimpan."
        actions={
          <button
            className="btn btn-ghost btn-sm"
            onClick={() => load(Number(String(income || "").replace(/[^\d.]/g, "")) || 0)}
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
            <div className="card card-body section-gap">
              <div className="card-title" style={{ fontSize: 18 }}>Pendapatan Bulanan</div>
              <form onSubmit={handleIncome} style={{ display: "flex", gap: 12, alignItems: "flex-end", flexWrap: "wrap" }}>
                <div className="form-group" style={{ flex: "1 1 260px", marginBottom: 0 }}>
                  <label className="form-label">Masukkan pendapatan bulanan (Rp)</label>
                  <input
                    className="form-input"
                    value={income}
                    onChange={(event) => setIncome(event.target.value)}
                    placeholder="Contoh: 5000000"
                    type="number"
                    min="0"
                  />
                </div>
                <button className="btn btn-primary" type="submit" disabled={submitting}>
                  {submitting ? (
                    <>
                      <span className="spinner" />
                      Menghitung...
                    </>
                  ) : (
                    "Hitung Budget"
                  )}
                </button>
              </form>
            </div>

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
                value={budget.monthly_income > 0 ? formatRp(potentialSaving) : "-"}
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
                      label={({ percentage }) => `${percentage}%`}
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
                  const over = ideal > 0 && actual > ideal;
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

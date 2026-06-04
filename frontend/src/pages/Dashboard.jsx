import { useEffect, useState } from "react";
import { BarChart2, RefreshCw, ShoppingBag, TrendingUp, Upload, Wallet } from "lucide-react";
import {
  Bar,
  BarChart,
  Cell,
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

function formatRpShort(value) {
  const n = Number(value || 0);
  if (n >= 1_000_000) return `Rp ${(n / 1_000_000).toFixed(1)}jt`;
  if (n >= 1_000) return `Rp ${(n / 1_000).toFixed(0)}rb`;
  return `Rp ${n.toLocaleString("id-ID")}`;
}

function formatRp(value) {
  return `Rp ${Number(value || 0).toLocaleString("id-ID")}`;
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div
      style={{
        background: "#fff",
        border: "1px solid var(--color-border)",
        borderRadius: 10,
        padding: "10px 12px",
        fontSize: 12,
        boxShadow: "var(--shadow)",
      }}
    >
      <div style={{ fontWeight: 800, marginBottom: 4 }}>{label}</div>
      <div style={{ color: "var(--color-primary)", fontWeight: 800 }}>
        {formatRp(payload[0].value)}
      </div>
    </div>
  );
}

export default function Dashboard({ onNavigate }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      setData(await getInsights());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const summary = data?.summary || {};
  const breakdown = data?.category_breakdown || [];
  const recent = data?.recent_transactions || [];
  const hasData = (summary.transaction_count || 0) > 0;

  return (
    <>
      <Header
        title="Dashboard"
        subtitle="Pantau pengeluaran dari struk dan screenshot pembayaran."
        actions={
          <button className="btn btn-ghost btn-sm" onClick={load} disabled={loading}>
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
            Memuat dashboard...
          </div>
        ) : !hasData ? (
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
        ) : (
          <>
            <div className="metric-grid">
              <MetricCard
                label="Total Pengeluaran"
                value={formatRp(summary.total_expense)}
                icon={Wallet}
                {...metricTone.primary}
              />
              <MetricCard
                label="Jumlah Transaksi"
                value={summary.transaction_count || 0}
                sub="transaksi tersimpan"
                icon={ShoppingBag}
                {...metricTone.success}
              />
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

            <div className="two-col section-gap">
              <div className="card card-body">
                <div className="card-title">Pengeluaran per Kategori</div>
                {breakdown.length ? (
                  <ResponsiveContainer width="100%" height={240}>
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
                        width={46}
                        tickFormatter={formatRpShort}
                      />
                      <Tooltip content={<ChartTooltip />} />
                      <Bar dataKey="total" radius={[8, 8, 0, 0]}>
                        {breakdown.map((_, index) => (
                          <Cell key={index} fill={chartColors[index % chartColors.length]} />
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
                      borderBottom: index < Math.min(breakdown.length, 5) - 1 ? "1px solid var(--color-border)" : "none",
                    }}
                  >
                    <div className="flex items-center gap-3">
                      <span
                        style={{
                          width: 10,
                          height: 10,
                          borderRadius: "50%",
                          background: chartColors[index % chartColors.length],
                        }}
                      />
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
                  <div style={{ color: "var(--color-muted)", fontSize: 14 }}>Transaksi terakhir yang berhasil disimpan.</div>
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

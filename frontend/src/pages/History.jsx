import { useEffect, useState } from "react";
import { RefreshCw, Search, ShoppingBag, Trash2, TrendingUp, Upload, Wallet } from "lucide-react";
import Header from "../components/Header.jsx";
import MetricCard from "../components/MetricCard.jsx";
import EmptyState from "../components/EmptyState.jsx";
import TransactionTable from "../components/TransactionTable.jsx";
import { deleteTransactions, getTransactions } from "../api/client.js";
import { categories, metricTone } from "../styles/theme.js";

const SOURCE_OPTIONS = [
  { value: "", label: "Semua Sumber" },
  { value: "receipt", label: "Foto Struk" },
  { value: "screenshot", label: "Screenshot Pembayaran" },
];

function formatRp(value) {
  return `Rp ${Number(value || 0).toLocaleString("id-ID")}`;
}

export default function History({ onNavigate }) {
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState("");
  const [source, setSource] = useState("");
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await getTransactions();
      setTransactions(res.transactions || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const filtered = transactions.filter((item) => {
    const query = search.trim().toLowerCase();
    if (query && !(item.merchant || "").toLowerCase().includes(query)) return false;
    if (category && item.category !== category) return false;
    if (source && item.source !== source) return false;
    return true;
  });

  const total = transactions.reduce((sum, item) => sum + (item.amount || 0), 0);
  const average = transactions.length ? total / transactions.length : 0;

  const handleDelete = async () => {
    setDeleting(true);
    setError("");
    try {
      await deleteTransactions();
      setTransactions([]);
      setConfirmDelete(false);
    } catch (err) {
      setError(err.message);
    } finally {
      setDeleting(false);
    }
  };

  return (
    <>
      <Header
        title="Riwayat Transaksi"
        subtitle="Semua transaksi yang sudah disimpan."
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
            Memuat riwayat transaksi...
          </div>
        ) : transactions.length === 0 ? (
          <div className="card card-body">
            <EmptyState
              title="Belum ada transaksi"
              description="Upload gambar transaksi pertama untuk mulai melihat riwayat."
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
                label="Total Transaksi"
                value={transactions.length}
                icon={ShoppingBag}
                {...metricTone.success}
              />
              <MetricCard
                label="Total Pengeluaran"
                value={formatRp(total)}
                icon={Wallet}
                {...metricTone.primary}
              />
              <MetricCard
                label="Rata-rata Transaksi"
                value={formatRp(average)}
                icon={TrendingUp}
                {...metricTone.violet}
              />
            </div>

            <div className="card card-body section-gap">
              <div style={{ display: "grid", gridTemplateColumns: "minmax(220px, 1fr) 220px 220px", gap: 12 }}>
                <div style={{ position: "relative" }}>
                  <Search
                    size={17}
                    style={{
                      position: "absolute",
                      left: 14,
                      top: "50%",
                      transform: "translateY(-50%)",
                      color: "var(--color-subtle)",
                    }}
                  />
                  <input
                    className="form-input"
                    style={{ paddingLeft: 42 }}
                    placeholder="Cari toko / penerima"
                    value={search}
                    onChange={(event) => setSearch(event.target.value)}
                  />
                </div>
                <select className="form-select" value={category} onChange={(event) => setCategory(event.target.value)}>
                  <option value="">Semua Kategori</option>
                  {categories.map((item) => (
                    <option key={item.value} value={item.value}>
                      {item.label}
                    </option>
                  ))}
                </select>
                <select className="form-select" value={source} onChange={(event) => setSource(event.target.value)}>
                  {SOURCE_OPTIONS.map((item) => (
                    <option key={item.value} value={item.value}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </div>
              <div style={{ marginTop: 14, color: "var(--color-muted)", fontSize: 14 }}>
                Menampilkan <strong>{filtered.length}</strong> dari <strong>{transactions.length}</strong> transaksi.
              </div>
            </div>

            <div className="card card-body section-gap">
              <div className="flex items-center justify-between" style={{ marginBottom: 16 }}>
                <div>
                  <div className="card-title" style={{ marginBottom: 4 }}>Riwayat Transaksi</div>
                  <div style={{ color: "var(--color-muted)", fontSize: 14 }}>Daftar transaksi yang sudah disimpan.</div>
                </div>
              </div>
              <TransactionTable transactions={filtered} />
            </div>

            <div className="flex justify-end">
              {!confirmDelete ? (
                <button className="btn btn-ghost btn-sm" style={{ color: "var(--color-danger)" }} onClick={() => setConfirmDelete(true)}>
                  <Trash2 size={15} />
                  Hapus Semua Data
                </button>
              ) : (
                <div className="alert alert-error flex items-center gap-3" style={{ flexWrap: "wrap" }}>
                  <span>Yakin ingin menghapus semua {transactions.length} transaksi?</span>
                  <button className="btn btn-danger btn-sm" onClick={handleDelete} disabled={deleting}>
                    {deleting ? "Menghapus..." : "Ya, Hapus"}
                  </button>
                  <button className="btn btn-ghost btn-sm" onClick={() => setConfirmDelete(false)}>
                    Batal
                  </button>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </>
  );
}

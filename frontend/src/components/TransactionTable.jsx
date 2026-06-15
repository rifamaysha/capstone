import { useState } from "react";
import { Trash2 } from "lucide-react";

const MERCHANT_MAX = 48;

const CATEGORIES = [
  { value: "makanan_minuman", label: "Makanan & Minuman" },
  { value: "transportasi",    label: "Transportasi" },
  { value: "belanja",         label: "Belanja & Retail" },
  { value: "hiburan",         label: "Hiburan & Wisata" },
  { value: "kesehatan",       label: "Kesehatan" },
  { value: "pendidikan",      label: "Pendidikan" },
  { value: "tagihan",         label: "Tagihan & Utilitas" },
  { value: "lainnya",         label: "Lainnya" },
];

const CAT_STYLE = {
  makanan_minuman: { bg: "var(--color-warning-soft)",  color: "var(--color-warning)" },
  transportasi:    { bg: "#eef6ff",                    color: "var(--chart-2)" },
  belanja:         { bg: "var(--color-violet-soft)",   color: "var(--color-violet)" },
  hiburan:         { bg: "var(--color-danger-soft)",   color: "var(--color-danger)" },
  kesehatan:       { bg: "var(--color-success-soft)",  color: "var(--color-success)" },
  pendidikan:      { bg: "#edf8ff",                    color: "#0369a1" },
  tagihan:         { bg: "#f1f4f8",                    color: "var(--color-muted)" },
  lainnya:         { bg: "#f1f4f8",                    color: "var(--color-muted)" },
};

function catStyle(cat) {
  return CAT_STYLE[cat] || CAT_STYLE.lainnya;
}

function catLabel(cat) {
  return CATEGORIES.find((c) => c.value === cat)?.label || cat;
}

function formatRp(v) {
  return `Rp ${Number(v || 0).toLocaleString("id-ID")}`;
}

function displayMerchant(merchant) {
  if (!merchant) return "-";
  if (merchant.length <= MERCHANT_MAX) return merchant;
  return `${merchant.slice(0, MERCHANT_MAX).trimEnd()}...`;
}

function displaySource(source) {
  if (source === "manual") return "Manual";
  if (source === "screenshot") return "Screenshot Pembayaran";
  if (source === "receipt") return "Foto Struk";
  return "Scan Transaksi";
}

function formatSavedAt(savedAt) {
  if (!savedAt) return "-";
  try {
    const d = new Date(savedAt);
    return d.toLocaleDateString("id-ID", {
      day: "numeric", month: "short", year: "numeric",
      hour: "2-digit", minute: "2-digit",
    });
  } catch {
    return savedAt;
  }
}

export default function TransactionTable({
  transactions,
  showSavedAt = false,
  onDeleteOne,
  onUpdateCategory,
}) {
  const [confirmId, setConfirmId]           = useState(null);
  const [deletingId, setDeletingId]         = useState(null);
  const [editCatId, setEditCatId]           = useState(null);
  const [savingCatId, setSavingCatId]       = useState(null);
  // Track optimistic category per id so the badge updates instantly
  const [localCats, setLocalCats]           = useState({});

  if (!transactions || transactions.length === 0) {
    return (
      <div style={{ padding: "28px", textAlign: "center", color: "var(--color-muted)", fontSize: "14px" }}>
        Tidak ada transaksi yang sesuai.
      </div>
    );
  }

  const handleDeleteClick   = (id) => setConfirmId(id);
  const handleDeleteCancel  = ()   => setConfirmId(null);
  const handleDeleteConfirm = async (id) => {
    setDeletingId(id);
    try { await onDeleteOne(id); } finally { setDeletingId(null); setConfirmId(null); }
  };

  const handleCategoryClick  = (id) => onUpdateCategory && setEditCatId(id);
  const handleCategoryCancel = ()   => setEditCatId(null);
  const handleCategoryChange = async (id, newCat) => {
    setSavingCatId(id);
    setLocalCats((prev) => ({ ...prev, [id]: newCat }));
    setEditCatId(null);
    try {
      await onUpdateCategory(id, newCat);
    } catch {
      // revert optimistic update on failure
      setLocalCats((prev) => { const n = { ...prev }; delete n[id]; return n; });
    } finally {
      setSavingCatId(null);
    }
  };

  return (
    <div className="table-wrapper">
      <table className="data-table">
        <thead>
          <tr>
            <th>Merchant</th>
            <th style={{ textAlign: "right" }}>Amount</th>
            <th>Kategori</th>
            <th>Sumber</th>
            <th>Tanggal Transaksi</th>
            {showSavedAt && <th>Disimpan</th>}
            {onDeleteOne && <th style={{ width: 80 }}></th>}
          </tr>
        </thead>
        <tbody>
          {transactions.map((t) => {
            const activeCat = localCats[t.id] || t.category;
            const cs        = catStyle(activeCat);
            const full      = t.merchant || "";
            const shown     = displayMerchant(full);
            const isConfirming  = confirmId  === t.id;
            const isDeleting    = deletingId === t.id;
            const isEditingCat  = editCatId  === t.id;
            const isSavingCat   = savingCatId === t.id;

            return (
              <tr key={t.id}>
                <td>
                  <span style={{ fontWeight: 750 }} title={full.length > MERCHANT_MAX ? full : undefined}>
                    {shown}
                  </span>
                </td>
                <td className="font-bold text-right" style={{ fontVariantNumeric: "tabular-nums", whiteSpace: "nowrap" }}>
                  {formatRp(t.amount)}
                </td>
                <td>
                  {isEditingCat ? (
                    <span style={{ display: "flex", gap: 4, alignItems: "center" }}>
                      <select
                        className="form-select"
                        style={{ fontSize: 12, padding: "2px 6px", height: 28 }}
                        defaultValue={activeCat}
                        autoFocus
                        onChange={(e) => handleCategoryChange(t.id, e.target.value)}
                        onBlur={handleCategoryCancel}
                      >
                        {CATEGORIES.map((c) => (
                          <option key={c.value} value={c.value}>{c.label}</option>
                        ))}
                      </select>
                    </span>
                  ) : (
                    <span
                      className="badge"
                      style={{
                        background: cs.bg,
                        color: cs.color,
                        cursor: onUpdateCategory ? "pointer" : "default",
                        opacity: isSavingCat ? 0.5 : 1,
                      }}
                      title={onUpdateCategory ? "Klik untuk ubah kategori" : undefined}
                      onClick={() => handleCategoryClick(t.id)}
                    >
                      {catLabel(activeCat)}
                    </span>
                  )}
                </td>
                <td style={{ color: "var(--color-muted)", fontSize: "13px" }}>
                  {displaySource(t.source)}
                </td>
                <td style={{ color: "var(--color-muted)", fontSize: "13px", whiteSpace: "nowrap" }}>
                  {t.date || "-"}
                </td>
                {showSavedAt && (
                  <td style={{ color: "var(--color-muted)", fontSize: "13px", whiteSpace: "nowrap" }}>
                    {formatSavedAt(t.saved_at)}
                  </td>
                )}
                {onDeleteOne && (
                  <td style={{ whiteSpace: "nowrap" }}>
                    {isConfirming ? (
                      <span style={{ display: "flex", gap: 6, alignItems: "center" }}>
                        <button
                          className="btn btn-danger btn-sm"
                          style={{ padding: "2px 10px", fontSize: 12 }}
                          onClick={() => handleDeleteConfirm(t.id)}
                          disabled={isDeleting}
                        >
                          {isDeleting ? "..." : "Hapus"}
                        </button>
                        <button
                          className="btn btn-ghost btn-sm"
                          style={{ padding: "2px 8px", fontSize: 12 }}
                          onClick={handleDeleteCancel}
                          disabled={isDeleting}
                        >
                          Batal
                        </button>
                      </span>
                    ) : (
                      <button
                        className="btn btn-ghost btn-sm"
                        style={{ color: "var(--color-danger)", padding: "4px 8px" }}
                        onClick={() => handleDeleteClick(t.id)}
                        title="Hapus transaksi ini"
                      >
                        <Trash2 size={14} />
                      </button>
                    )}
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

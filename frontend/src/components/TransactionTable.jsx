const MERCHANT_MAX = 48;

const CAT_STYLE = {
  makanan_minuman: { bg: "var(--color-warning-soft)", color: "var(--color-warning)" },
  transportasi: { bg: "#eef6ff", color: "var(--chart-2)" },
  belanja: { bg: "var(--color-violet-soft)", color: "var(--color-violet)" },
  hiburan: { bg: "var(--color-danger-soft)", color: "var(--color-danger)" },
  kesehatan: { bg: "var(--color-success-soft)", color: "var(--color-success)" },
  pendidikan: { bg: "#edf8ff", color: "#0369a1" },
  tagihan: { bg: "#f1f4f8", color: "var(--color-muted)" },
  lainnya: { bg: "#f1f4f8", color: "var(--color-muted)" },
};

function catStyle(cat) {
  return CAT_STYLE[cat] || CAT_STYLE.lainnya;
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

export default function TransactionTable({ transactions }) {
  if (!transactions || transactions.length === 0) {
    return (
      <div
        style={{
          padding: "28px",
          textAlign: "center",
          color: "var(--color-muted)",
          fontSize: "14px",
        }}
      >
        Tidak ada transaksi yang sesuai.
      </div>
    );
  }

  return (
    <div className="table-wrapper">
      <table className="data-table">
        <thead>
          <tr>
            <th>Merchant</th>
            <th style={{ textAlign: "right" }}>Amount</th>
            <th>Kategori</th>
            <th>Sumber</th>
            <th>Tanggal</th>
          </tr>
        </thead>
        <tbody>
          {transactions.map((t) => {
            const cs = catStyle(t.category);
            const full = t.merchant || "";
            const shown = displayMerchant(full);
            return (
              <tr key={t.id}>
                <td>
                  <span
                    style={{ fontWeight: 750 }}
                    title={full.length > MERCHANT_MAX ? full : undefined}
                  >
                    {shown}
                  </span>
                </td>
                <td
                  className="font-bold text-right"
                  style={{ fontVariantNumeric: "tabular-nums", whiteSpace: "nowrap" }}
                >
                  {formatRp(t.amount)}
                </td>
                <td>
                  <span className="badge" style={{ background: cs.bg, color: cs.color }}>
                    {t.category_display || t.category}
                  </span>
                </td>
                <td style={{ color: "var(--color-muted)", fontSize: "13px" }}>
                  {displaySource(t.source)}
                </td>
                <td style={{ color: "var(--color-muted)", fontSize: "13px", whiteSpace: "nowrap" }}>
                  {t.date || "-"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

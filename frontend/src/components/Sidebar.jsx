import {
  LayoutDashboard,
  ScanLine,
  ReceiptText,
  TrendingUp,
} from "lucide-react";

const NAV_ITEMS = [
  { key: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { key: "upload", label: "Upload & Proses", icon: ScanLine },
  { key: "history", label: "Riwayat Transaksi", icon: ReceiptText },
  { key: "insights", label: "Insight Penghematan", icon: TrendingUp },
];

export default function Sidebar({ activePage, onNavigate }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="sidebar-logo-wrap">
          <img src="/logo_icon.png" alt="Smart Personal Expense" />
        </div>
        <div className="sidebar-brand-text">
          <div className="sidebar-brand-name">Smart Personal Expense</div>
          <div className="sidebar-brand-sub">Personal finance assistant</div>
        </div>
      </div>

      <nav className="sidebar-nav">
        <div className="sidebar-section">Menu Utama</div>
        {NAV_ITEMS.map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            className={`sidebar-item${activePage === key ? " active" : ""}`}
            onClick={() => onNavigate(key)}
          >
            <Icon />
            <span>{label}</span>
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        Kelola pengeluaran lebih rapi
      </div>
    </aside>
  );
}

import { ReceiptText } from "lucide-react";

export default function EmptyState({ title, description, action, icon: Icon }) {
  const IconComp = Icon || ReceiptText;
  return (
    <div className="empty-state">
      <div className="empty-state-icon">
        <IconComp size={26} />
      </div>
      <div className="empty-state-title">{title}</div>
      <div className="empty-state-desc">{description}</div>
      {action && <div style={{ marginTop: 12 }}>{action}</div>}
    </div>
  );
}

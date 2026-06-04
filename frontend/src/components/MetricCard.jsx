export default function MetricCard({
  label,
  value,
  sub,
  icon: Icon,
  iconBg,
  iconColor,
  accent,
}) {
  return (
    <div className="metric-card" style={accent ? { "--metric-accent": accent } : {}}>
      {Icon && (
        <div
          className="metric-card-icon"
          style={{
            background: iconBg || "var(--color-primary-soft)",
            color: iconColor || "var(--color-primary)",
          }}
        >
          <Icon size={20} />
        </div>
      )}
      <div>
        <div className="metric-card-label">{label}</div>
        <div className="metric-card-value">{value}</div>
        {sub && <div className="metric-card-sub">{sub}</div>}
      </div>
    </div>
  );
}

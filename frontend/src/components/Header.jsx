export default function Header({ title, subtitle, actions }) {
  return (
    <div className="page-header">
      <div className="page-header-left">
        <h1 className="page-header-title">{title}</h1>
        {subtitle && <div className="page-header-sub">{subtitle}</div>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}

import { Component } from "react";

/**
 * Global error boundary — catches React render/lifecycle errors anywhere
 * in the app and shows a friendly fallback instead of a white screen.
 * Specifically useful during a live demo: a single bad data point in any
 * page won't crash the whole UI.
 */
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    // Log to console for dev debugging; in production you'd ship this somewhere.
    console.error("ErrorBoundary caught an error:", error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  handleReload = () => {
    window.location.reload();
  };

  render() {
    if (!this.state.hasError) return this.props.children;

    return (
      <div
        style={{
          minHeight: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: 40,
          background: "var(--color-background, #f7f8fc)",
        }}
      >
        <div
          style={{
            maxWidth: 520,
            width: "100%",
            background: "var(--color-card, #fff)",
            border: "1px solid var(--color-border, #e2e8f0)",
            borderRadius: 16,
            padding: 32,
            textAlign: "center",
            boxShadow: "var(--shadow, 0 10px 40px rgba(0,0,0,0.08))",
          }}
        >
          <div style={{ fontSize: 56, lineHeight: 1, marginBottom: 12 }}>😵</div>
          <h2 style={{ marginBottom: 8, fontWeight: 800 }}>
            Aplikasi mengalami kendala
          </h2>
          <p
            style={{
              color: "var(--color-muted, #64748b)",
              marginBottom: 20,
              fontSize: 14,
            }}
          >
            Ada error tak terduga. Coba klik tombol di bawah, atau reload halaman.
          </p>

          {this.state.error?.message && (
            <pre
              style={{
                background: "var(--color-danger-soft, #fff1f2)",
                color: "#b91c1c",
                padding: 12,
                borderRadius: 10,
                fontSize: 12,
                textAlign: "left",
                overflowX: "auto",
                marginBottom: 20,
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
              }}
            >
              {this.state.error.message}
            </pre>
          )}

          <div style={{ display: "flex", gap: 10, justifyContent: "center", flexWrap: "wrap" }}>
            <button className="btn btn-secondary btn-sm" onClick={this.handleReset}>
              Coba Lagi
            </button>
            <button className="btn btn-primary btn-sm" onClick={this.handleReload}>
              Reload Halaman
            </button>
          </div>
        </div>
      </div>
    );
  }
}

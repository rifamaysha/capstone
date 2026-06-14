import { useState } from "react";
import Sidebar from "./components/Sidebar.jsx";
import ErrorBoundary from "./components/ErrorBoundary.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import UploadProcess from "./pages/UploadProcess.jsx";
import History from "./pages/History.jsx";
import Insights from "./pages/Insights.jsx";

const PAGES = {
  dashboard: Dashboard,
  upload: UploadProcess,
  history: History,
  insights: Insights,
};

export default function App() {
  const [page, setPage] = useState("dashboard");

  const PageComponent = PAGES[page] || Dashboard;

  return (
    <ErrorBoundary>
      <div className="app-layout">
        <Sidebar activePage={page} onNavigate={setPage} />
        <div className="app-content">
          {/* Inner ErrorBoundary with `key` so error in one page doesn't
              persist when user navigates to another tab. */}
          <ErrorBoundary key={page}>
            <PageComponent onNavigate={setPage} />
          </ErrorBoundary>
        </div>
      </div>
    </ErrorBoundary>
  );
}

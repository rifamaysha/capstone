import { useState } from "react";
import Sidebar from "./components/Sidebar.jsx";
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
    <div className="app-layout">
      <Sidebar activePage={page} onNavigate={setPage} />
      <div className="app-content">
        <PageComponent onNavigate={setPage} />
      </div>
    </div>
  );
}

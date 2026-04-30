import { BrowserRouter as Router, Routes, Route, Navigate, NavLink } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import { Music, Settings, LayoutDashboard } from 'lucide-react';
import './App.css';

const Navigation = () => {
  return (
    <nav className="nav-bar">
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <Music color="#4facfe" />
        <span style={{ fontWeight: 'bold', letterSpacing: '1px' }}>GENRE.AI</span>
      </div>
      <div className="nav-links">
        <NavLink to="/dashboard" className={({ isActive }) => isActive ? 'active' : ''}>
          <LayoutDashboard size={18} /> Dashboard
        </NavLink>
        <NavLink to="/settings" className={({ isActive }) => isActive ? 'active' : ''}>
          <Settings size={18} /> Settings
        </NavLink>
      </div>
    </nav>
  );
};

function App() {
  return (
    <Router>
      <div id="root">
        <Navigation />
        <Routes>
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/settings" element={
            <div className="glass-card" style={{ maxWidth: '600px', margin: '2rem auto' }}>
              <h2>Under Construction</h2>
              <p>Settings and theme customization coming soon!</p>
            </div>
          } />
          <Route path="*" element={<Navigate to="/dashboard" />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;

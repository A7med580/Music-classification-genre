import { BrowserRouter as Router, Routes, Route, Navigate, Link, NavLink } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import AuthPage from './pages/AuthPage';
import Dashboard from './pages/Dashboard';
import { Music, LogOut, Settings, LayoutDashboard, History } from 'lucide-react';
import './App.css';

const ProtectedRoute = ({ children }) => {
  const { token, loading } = useAuth();
  if (loading) return <div>Loading...</div>;
  return token ? children : <Navigate to="/login" />;
};

const Navigation = () => {
  const { user, logout } = useAuth();
  if (!user) return null;

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
        <button onClick={logout} style={{ background: 'transparent', padding: '5px 10px', display: 'flex', alignItems: 'center', gap: '5px', fontSize: '0.9em', color: '#ff6b6b' }}>
          <LogOut size={16} /> Logout
        </button>
      </div>
    </nav>
  );
};

function App() {
  return (
    <Router>
      <AuthProvider>
        <div id="root">
          <Navigation />
          <Routes>
            <Route path="/login" element={<AuthPage isLogin={true} />} />
            <Route path="/signup" element={<AuthPage isLogin={false} />} />
            <Route path="/dashboard" element={
              <ProtectedRoute>
                <Dashboard />
              </ProtectedRoute>
            } />
            <Route path="/settings" element={
              <ProtectedRoute>
                <div className="glass-card">
                  <h2>Under Construction</h2>
                  <p>Settings and theme customization coming soon!</p>
                </div>
              </ProtectedRoute>
            } />
            <Route path="/" element={<Navigate to="/dashboard" />} />
          </Routes>
        </div>
      </AuthProvider>
    </Router>
  );
}

export default App;

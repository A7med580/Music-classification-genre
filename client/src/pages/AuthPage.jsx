import { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { useNavigate, Link } from 'react-router-dom';
import { Music, Lock, User, ArrowRight } from 'lucide-react';

export default function AuthPage({ isLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login, register } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      if (isLogin) {
        await login(username, password);
        navigate('/dashboard');
      } else {
        await register(username, password);
        navigate('/login');
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Authentication failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="glass-card" style={{ maxWidth: '400px' }}>
      <div style={{ marginBottom: '2rem' }}>
        <Music size={48} color="#4facfe" />
        <h2>{isLogin ? 'Welcome Back' : 'Create Account'}</h2>
        <p style={{ opacity: 0.6 }}>Experience music through AI</p>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="input-group">
          <User size={18} />
          <input
            type="text"
            placeholder="Username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          />
        </div>
        <div className="input-group">
          <Lock size={18} />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>

        {error && <p style={{ color: '#ff6b6b', fontSize: '0.9em' }}>{error}</p>}

        <button type="submit" disabled={loading} style={{ width: '100%', marginTop: '1rem' }}>
          {loading ? 'Processing...' : (isLogin ? 'Login' : 'Sign Up')}
          <ArrowRight size={18} style={{ marginLeft: '8px' }} />
        </button>
      </form>

      <p style={{ marginTop: '1.5rem', fontSize: '0.9em', opacity: 0.8 }}>
        {isLogin ? "Don't have an account? " : "Already have an account? "}
        <Link to={isLogin ? '/signup' : '/login'} style={{ color: '#4facfe', fontWeight: 'bold', textDecoration: 'none' }}>
          {isLogin ? 'Sign Up' : 'Login'}
        </Link>
      </p>
    </div>
  );
}

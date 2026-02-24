import { useState, useEffect } from 'react';
import { attemptLogin, getLockoutRemaining } from '../auth';

const LoginPage = ({ onLogin }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [lockoutSecs, setLockoutSecs] = useState(() => getLockoutRemaining());

  // Tick down the lockout countdown
  useEffect(() => {
    if (lockoutSecs <= 0) return;
    const timer = setInterval(() => {
      const remaining = getLockoutRemaining();
      setLockoutSecs(remaining);
      if (remaining <= 0) clearInterval(timer);
    }, 1000);
    return () => clearInterval(timer);
  }, [lockoutSecs > 0]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const result = await attemptLogin(email, password);
      if (result.success) {
        onLogin(result.email);
      } else {
        setError(result.error);
        if (result.locked) setLockoutSecs(getLockoutRemaining());
      }
    } catch (err) {
      setError('Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const isLocked = lockoutSecs > 0;

  return (
    <div style={styles.wrapper}>
      <div style={styles.card}>
        {/* Logo / Brand */}
        <div style={styles.brandRow}>
          <div style={styles.logoBox}>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="12" fill="#00a884" />
              <path d="M7 12.5l3.5 3.5 6.5-7" stroke="#fff" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
          <div>
            <div style={styles.brandName}>JPSearch</div>
            <div style={styles.brandSub}>by Talentin</div>
          </div>
        </div>

        <h2 style={styles.heading}>Welcome back</h2>
        <p style={styles.subheading}>Sign in with your work email to continue</p>

        <form onSubmit={handleSubmit} style={styles.form}>
          <div style={styles.fieldGroup}>
            <label style={styles.label}>Work Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@talentin.ai"
              required
              disabled={isLocked}
              style={{ ...styles.input, opacity: isLocked ? 0.6 : 1 }}
              autoFocus
            />
          </div>

          <div style={styles.fieldGroup}>
            <label style={styles.label}>Password</label>
            <div style={styles.passwordWrap}>
              <input
                type={showPassword ? 'text' : 'password'}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••••"
                required
                disabled={isLocked}
                style={{ ...styles.input, paddingRight: '2.8rem', opacity: isLocked ? 0.6 : 1 }}
              />
              <button
                type="button"
                onClick={() => setShowPassword((v) => !v)}
                style={styles.eyeBtn}
                tabIndex={-1}
              >
                {showPassword ? (
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#6b7280" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" />
                    <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" />
                    <line x1="1" y1="1" x2="23" y2="23" />
                  </svg>
                ) : (
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#6b7280" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                    <circle cx="12" cy="12" r="3" />
                  </svg>
                )}
              </button>
            </div>
          </div>

          {error && <div style={styles.errorBox}>{error}</div>}

          <button type="submit" disabled={loading || isLocked} style={{ ...styles.submitBtn, opacity: (loading || isLocked) ? 0.6 : 1, cursor: (loading || isLocked) ? 'not-allowed' : 'pointer' }}>
            {loading ? 'Signing in...' : isLocked ? `Locked — ${lockoutSecs}s` : 'Sign In'}
          </button>
        </form>

        <p style={styles.footer}>
          Access restricted to <strong>@talentin.ai</strong> and <strong>@avistatech.net</strong> accounts.
        </p>
      </div>
    </div>
  );
};

const styles = {
  wrapper: {
    minHeight: '100vh',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'linear-gradient(135deg, #f0fdf9 0%, #e0f2f1 40%, #f3f4f6 100%)',
    fontFamily: "'Inter', sans-serif",
    padding: '1rem',
  },
  card: {
    background: '#ffffff',
    borderRadius: '1.25rem',
    boxShadow: '0 20px 60px rgba(0,0,0,0.1)',
    padding: '2.5rem 2.25rem',
    width: '100%',
    maxWidth: '420px',
  },
  brandRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
    marginBottom: '2rem',
  },
  logoBox: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  brandName: {
    fontWeight: '700',
    fontSize: '1.1rem',
    color: '#111827',
    lineHeight: 1.2,
  },
  brandSub: {
    fontSize: '0.75rem',
    color: '#6b7280',
  },
  heading: {
    fontSize: '1.5rem',
    fontWeight: '700',
    color: '#111827',
    margin: '0 0 0.4rem 0',
  },
  subheading: {
    fontSize: '0.875rem',
    color: '#6b7280',
    marginBottom: '1.75rem',
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1.1rem',
  },
  fieldGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.35rem',
  },
  label: {
    fontSize: '0.8rem',
    fontWeight: '600',
    color: '#374151',
    letterSpacing: '0.02em',
  },
  input: {
    width: '100%',
    padding: '0.65rem 0.875rem',
    border: '1.5px solid #e5e7eb',
    borderRadius: '0.625rem',
    fontSize: '0.9rem',
    color: '#111827',
    outline: 'none',
    transition: 'border-color 0.15s',
    boxSizing: 'border-box',
    background: '#fafafa',
  },
  passwordWrap: {
    position: 'relative',
  },
  eyeBtn: {
    position: 'absolute',
    right: '0.75rem',
    top: '50%',
    transform: 'translateY(-50%)',
    background: 'none',
    border: 'none',
    cursor: 'pointer',
    padding: '0',
    display: 'flex',
    alignItems: 'center',
  },
  errorBox: {
    background: '#fef2f2',
    border: '1px solid #fecaca',
    color: '#dc2626',
    borderRadius: '0.5rem',
    padding: '0.6rem 0.875rem',
    fontSize: '0.825rem',
  },
  submitBtn: {
    background: '#00a884',
    color: '#fff',
    border: 'none',
    borderRadius: '0.625rem',
    padding: '0.75rem',
    fontSize: '0.95rem',
    fontWeight: '600',
    cursor: 'pointer',
    marginTop: '0.25rem',
    transition: 'background 0.15s',
  },
  footer: {
    marginTop: '1.5rem',
    fontSize: '0.75rem',
    color: '#9ca3af',
    textAlign: 'center',
    lineHeight: 1.6,
  },
};

export default LoginPage;

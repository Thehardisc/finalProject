import { useState } from 'react';
import axios from 'axios';
import LiveDemoBackground from './LiveDemoBackground';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';

const DEMO_SLOTS = [
    { slot: 0, name: 'Alice Chen',   initials: 'AC' },
    { slot: 1, name: 'Bob Kim',      initials: 'BK' },
    { slot: 2, name: 'Charlie Park', initials: 'CP' },
    { slot: 3, name: 'Diana Lee',    initials: 'DL' },
    { slot: 4, name: 'Eve Zhao',     initials: 'EZ' },
];

const INJECTED_CSS = `
@keyframes meshDrift1 {
  0%, 100% { transform: translate(0, 0) scale(1); }
  33%       { transform: translate(60px, -40px) scale(1.08); }
  66%       { transform: translate(-40px, 50px) scale(0.94); }
}
@keyframes meshDrift2 {
  0%, 100% { transform: translate(0, 0) scale(1); }
  33%       { transform: translate(-60px, 55px) scale(1.05); }
  66%       { transform: translate(50px, -35px) scale(0.96); }
}
@keyframes meshDrift3 {
  0%, 100% { transform: translate(0, 0) scale(1); }
  50%       { transform: translate(35px, 45px) scale(1.06); }
}
@keyframes meshDrift4 {
  0%, 100% { transform: translate(0, 0) scale(1); }
  40%       { transform: translate(-30px, -50px) scale(1.04); }
  80%       { transform: translate(45px, 30px) scale(0.97); }
}
@keyframes loginCardAppear {
  from { opacity: 0; transform: translateY(24px) scale(0.96); }
  to   { opacity: 1; transform: translateY(0)    scale(1); }
}
.il-login-input::placeholder { color: rgba(255,255,255,0.22); }
.il-login-input:focus {
  outline: none;
  border-color: rgba(139, 92, 246, 0.55) !important;
  box-shadow: 0 0 0 3px rgba(109, 40, 217, 0.18), 0 0 12px rgba(109, 40, 217, 0.12) !important;
  background: rgba(255, 255, 255, 0.09) !important;
}
.il-submit-btn:hover:not(:disabled) {
  box-shadow: 0 8px 44px rgba(109, 40, 217, 0.70), inset 0 1px 0 rgba(255,255,255,0.12) !important;
  transform: translateY(-1px);
}
.il-submit-btn:active:not(:disabled) { transform: translateY(0); }
.il-demo-btn:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.09) !important;
  border-color: rgba(255, 255, 255, 0.22) !important;
  transform: translateY(-2px);
}
.il-tab-btn:hover { color: rgba(255,255,255,0.80) !important; }
`;

export default function LoginModal({ onSuccess, onClose }) {
    const [tab, setTab] = useState('login');
    const [email, setEmail] = useState('');
    const [firstName, setFirstName] = useState('');
    const [lastName, setLastName] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const [demoLoading, setDemoLoading] = useState(null);

    const resetForm = () => {
        setEmail(''); setFirstName(''); setLastName('');
        setPassword(''); setConfirmPassword(''); setError('');
    };

    const validate = () => {
        if (!email.trim() || !/^[^@]+@[^@]+\.[^@]+$/.test(email)) {
            setError('A valid email address is required.'); return false;
        }
        if (tab === 'register') {
            if (!firstName.trim()) { setError('First name is required.'); return false; }
            if (!lastName.trim())  { setError('Last name is required.');  return false; }
        }
        if (!password) { setError('Password is required.'); return false; }
        if (password.length < 8) { setError('Password must be at least 8 characters.'); return false; }
        if (tab === 'register' && password !== confirmPassword) {
            setError('Passwords do not match.'); return false;
        }
        return true;
    };

    const handleDemoLogin = async (slot) => {
        setError('');
        setDemoLoading(slot);
        try {
            const res = await axios.post(`${API_BASE}/auth/demo-login/${slot}`);
            onSuccess(res.data);
        } catch (err) {
            const detail = err.response?.data?.detail;
            setError(typeof detail === 'string' ? detail : 'Demo login failed. Is the server running?');
        } finally {
            setDemoLoading(null);
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        if (!validate()) return;
        setLoading(true);
        try {
            const endpoint = tab === 'register' ? '/auth/register' : '/auth/login';
            const payload = tab === 'register'
                ? { email: email.trim(), first_name: firstName.trim(), last_name: lastName.trim(), password }
                : { email: email.trim(), password };
            const res = await axios.post(`${API_BASE}${endpoint}`, payload);
            onSuccess(res.data);
        } catch (err) {
            const detail = err.response?.data?.detail;
            if (typeof detail === 'string') {
                setError(detail);
            } else if (Array.isArray(detail)) {
                setError(detail.map(d => d.msg).join(' '));
            } else {
                setError('Something went wrong. Please try again.');
            }
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={s.overlay}>
            <style>{INJECTED_CSS}</style>

            <div style={s.orb1} />
            <div style={s.orb2} />
            <div style={s.orb3} />
            <div style={s.orb4} />

            <LiveDemoBackground />

            <div style={s.card}>
                {onClose && (
                    <button
                        onClick={onClose}
                        style={{ position: 'absolute', top: 18, right: 18, width: 32, height: 32, borderRadius: '50%', background: 'rgba(255,255,255,0.07)', border: '1px solid rgba(255,255,255,0.12)', color: 'rgba(255,255,255,0.50)', fontSize: '1rem', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'all 0.2s', fontFamily: 'inherit', zIndex: 2 }}
                        title="Back to landing page"
                    >
                        ×
                    </button>
                )}
                <div style={s.logoRow}>
                    <div style={s.logoIcon} />
                    <h1 style={s.logoText}>InnerLink</h1>
                </div>

                <p style={s.tagline}>Understand every message. Feel every word.</p>
                <p style={s.subTagline}>Real-time emotion intelligence for human connection.</p>

                <div style={s.tabRow}>
                    <button
                        id="auth-tab-login"
                        type="button"
                        className="il-tab-btn"
                        aria-selected={tab === 'login'}
                        style={{ ...s.tabBtn, ...(tab === 'login' ? s.tabActive : {}) }}
                        onClick={() => { setTab('login'); resetForm(); }}
                    >
                        Sign In
                    </button>
                    <button
                        id="auth-tab-register"
                        type="button"
                        className="il-tab-btn"
                        aria-selected={tab === 'register'}
                        style={{ ...s.tabBtn, ...(tab === 'register' ? s.tabActive : {}) }}
                        onClick={() => { setTab('register'); resetForm(); }}
                    >
                        Register
                    </button>
                </div>

                <form onSubmit={handleSubmit} style={s.form} noValidate>
                    <label htmlFor="auth-email" style={s.label}>Email Address</label>
                    <input
                        id="auth-email"
                        name="email"
                        type="email"
                        autoComplete="email"
                        autoFocus
                        value={email}
                        onChange={e => setEmail(e.target.value)}
                        placeholder="you@example.com"
                        className="il-login-input"
                        style={s.input}
                        disabled={loading}
                        aria-required="true"
                    />

                    {tab === 'register' && (
                        <>
                            <label htmlFor="auth-first-name" style={s.label}>First Name</label>
                            <input
                                id="auth-first-name"
                                name="given-name"
                                type="text"
                                autoComplete="given-name"
                                value={firstName}
                                onChange={e => setFirstName(e.target.value)}
                                placeholder="Jane"
                                className="il-login-input"
                                style={s.input}
                                disabled={loading}
                                aria-required="true"
                            />
                            <label htmlFor="auth-last-name" style={s.label}>Last Name</label>
                            <input
                                id="auth-last-name"
                                name="family-name"
                                type="text"
                                autoComplete="family-name"
                                value={lastName}
                                onChange={e => setLastName(e.target.value)}
                                placeholder="Doe"
                                className="il-login-input"
                                style={s.input}
                                disabled={loading}
                                aria-required="true"
                            />
                        </>
                    )}

                    <label htmlFor="auth-password" style={s.label}>Password</label>
                    <input
                        id="auth-password"
                        name="password"
                        type="password"
                        autoComplete={tab === 'register' ? 'new-password' : 'current-password'}
                        value={password}
                        onChange={e => setPassword(e.target.value)}
                        placeholder="Min. 8 characters"
                        className="il-login-input"
                        style={s.input}
                        disabled={loading}
                        aria-required="true"
                    />

                    {tab === 'register' && (
                        <>
                            <label htmlFor="auth-confirm-password" style={s.label}>Confirm Password</label>
                            <input
                                id="auth-confirm-password"
                                name="confirm-password"
                                type="password"
                                autoComplete="new-password"
                                value={confirmPassword}
                                onChange={e => setConfirmPassword(e.target.value)}
                                placeholder="Repeat password"
                                className="il-login-input"
                                style={s.input}
                                disabled={loading}
                                aria-required="true"
                            />
                        </>
                    )}

                    {error && (
                        <div style={s.errorBox} role="alert">
                            <span>⚠ </span>{error}
                        </div>
                    )}

                    <button
                        id="auth-submit"
                        type="submit"
                        className="il-submit-btn"
                        style={s.submitBtn}
                        disabled={loading}
                        aria-busy={loading}
                    >
                        {loading
                            ? <span style={s.spinner} />
                            : (tab === 'register' ? 'Create Account' : 'Sign In')}
                    </button>
                </form>

                <p style={s.switchHint}>
                    {tab === 'login'
                        ? <>No account?{' '}<span role="button" tabIndex="0" style={s.switchLink} onClick={() => { setTab('register'); resetForm(); }}>Register here</span></>
                        : <>Have an account?{' '}<span role="button" tabIndex="0" style={s.switchLink} onClick={() => { setTab('login'); resetForm(); }}>Sign in</span></>
                    }
                </p>

                <div style={s.demoSection}>
                    <div style={s.demoDivider}>
                        <span style={s.demoDividerLine} />
                        <span style={s.demoDividerText}>or try a demo account</span>
                        <span style={s.demoDividerLine} />
                    </div>
                    <div style={s.demoGrid}>
                        {DEMO_SLOTS.map(({ slot, name, initials }) => (
                            <button
                                key={slot}
                                type="button"
                                className="il-demo-btn"
                                onClick={() => handleDemoLogin(slot)}
                                disabled={demoLoading !== null || loading}
                                style={{
                                    ...s.demoBtn,
                                    opacity: (demoLoading !== null || loading) ? 0.50 : 1,
                                }}
                                title={`Log in as ${name}`}
                            >
                                <div style={s.demoAvatar}>
                                    {demoLoading === slot
                                        ? <span style={{ ...s.spinner, width: 14, height: 14, borderWidth: 2 }} />
                                        : initials}
                                </div>
                                <span style={s.demoBtnName}>{name.split(' ')[0]}</span>
                            </button>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}

const s = {
    overlay: {
        position: 'fixed', inset: 0,
        background: '#05060F',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        zIndex: 1000, overflow: 'hidden',
    },
    orb1: {
        position: 'absolute',
        width: 720, height: 720, borderRadius: '50%',
        top: '-220px', left: '-230px',
        background: 'radial-gradient(circle, rgba(79, 40, 210, 0.52) 0%, transparent 70%)',
        filter: 'blur(100px)',
        animation: 'meshDrift1 20s ease-in-out infinite',
    },
    orb2: {
        position: 'absolute',
        width: 660, height: 660, borderRadius: '50%',
        bottom: '-190px', right: '-210px',
        background: 'radial-gradient(circle, rgba(46, 16, 130, 0.62) 0%, transparent 70%)',
        filter: 'blur(90px)',
        animation: 'meshDrift2 25s ease-in-out infinite',
    },
    orb3: {
        position: 'absolute',
        width: 440, height: 440, borderRadius: '50%',
        top: '8%', right: '4%',
        background: 'radial-gradient(circle, rgba(180, 68, 48, 0.14) 0%, transparent 70%)',
        filter: 'blur(70px)',
        animation: 'meshDrift3 18s ease-in-out infinite',
    },
    orb4: {
        position: 'absolute',
        width: 510, height: 510, borderRadius: '50%',
        bottom: '4%', left: '8%',
        background: 'radial-gradient(circle, rgba(90, 20, 150, 0.32) 0%, transparent 70%)',
        filter: 'blur(80px)',
        animation: 'meshDrift4 22s ease-in-out infinite',
    },
    card: {
        position: 'relative', zIndex: 1,
        background: 'rgba(255, 255, 255, 0.055)',
        backdropFilter: 'blur(32px)',
        WebkitBackdropFilter: 'blur(32px)',
        border: '1px solid rgba(255, 255, 255, 0.11)',
        borderRadius: '28px',
        padding: '48px 44px',
        width: '100%', maxWidth: '420px',
        boxShadow: '0 32px 80px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,255,255,0.08)',
        animation: 'loginCardAppear 0.55s cubic-bezier(0.34, 1.2, 0.64, 1) both',
    },
    logoRow: { display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '10px' },
    logoIcon: {
        width: '36px', height: '36px', borderRadius: '10px',
        background: 'linear-gradient(135deg, #7c3aed 0%, #4338ca 100%)',
        boxShadow: '0 4px 16px rgba(109, 40, 217, 0.50)',
        flexShrink: 0,
    },
    logoText: {
        margin: 0, fontSize: '1.8rem', fontWeight: 800,
        color: '#fff', letterSpacing: '-0.5px',
    },
    tagline: {
        margin: '0 0 4px 0',
        fontSize: '1.05rem', fontWeight: 600,
        color: 'rgba(255,255,255,0.82)',
        letterSpacing: '-0.2px',
    },
    subTagline: {
        margin: '0 0 28px 0',
        fontSize: '0.82rem',
        color: 'rgba(255,255,255,0.36)',
        letterSpacing: '0.01em',
    },
    tabRow: {
        display: 'flex',
        background: 'rgba(255, 255, 255, 0.06)',
        borderRadius: '10px', padding: '4px', marginBottom: '24px',
    },
    tabBtn: {
        flex: 1, padding: '9px', border: 'none', borderRadius: '8px',
        cursor: 'pointer', background: 'transparent',
        color: 'rgba(255,255,255,0.38)', fontSize: '0.9rem', fontWeight: 500,
        transition: 'all 0.2s', fontFamily: 'inherit',
    },
    tabActive: {
        background: 'rgba(255, 255, 255, 0.11)',
        color: '#fff',
        boxShadow: '0 2px 8px rgba(0,0,0,0.28)',
    },
    form: { display: 'flex', flexDirection: 'column', gap: '6px' },
    label: {
        fontSize: '0.70rem', fontWeight: 700,
        color: 'rgba(255,255,255,0.36)',
        letterSpacing: '0.09em', textTransform: 'uppercase', marginTop: '12px',
    },
    input: {
        padding: '13px 14px',
        background: 'rgba(255, 255, 255, 0.07)',
        border: '1px solid rgba(255, 255, 255, 0.10)',
        borderRadius: '11px',
        color: '#fff',
        fontSize: '0.95rem',
        outline: 'none',
        transition: 'border-color 0.2s, box-shadow 0.2s, background 0.2s',
        fontFamily: 'inherit',
    },
    errorBox: {
        marginTop: '10px', padding: '10px 14px',
        background: 'rgba(220, 38, 38, 0.12)',
        border: '1px solid rgba(220, 38, 38, 0.25)',
        borderRadius: '8px',
        color: 'rgba(252, 165, 165, 0.88)',
        fontSize: '0.84rem',
    },
    submitBtn: {
        marginTop: '22px', padding: '14px',
        background: 'linear-gradient(135deg, #4c1d95 0%, #6d28d9 60%, #4338ca 100%)',
        border: '1px solid rgba(139, 92, 246, 0.35)',
        borderRadius: '12px', color: '#fff',
        fontWeight: 700, fontSize: '1rem', cursor: 'pointer',
        transition: 'box-shadow 0.2s, transform 0.15s',
        display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '48px',
        boxShadow: '0 4px 24px rgba(109, 40, 217, 0.48), inset 0 1px 0 rgba(255,255,255,0.08)',
        fontFamily: 'inherit',
    },
    spinner: {
        width: '20px', height: '20px', borderRadius: '50%',
        border: '2px solid rgba(255,255,255,0.25)',
        borderTopColor: '#fff',
        animation: 'spin 0.7s linear infinite',
        display: 'inline-block',
    },
    switchHint: {
        marginTop: '20px', textAlign: 'center',
        fontSize: '0.84rem', color: 'rgba(255,255,255,0.32)',
    },
    switchLink: {
        color: 'rgba(167, 139, 250, 0.88)',
        cursor: 'pointer', textDecoration: 'underline',
    },
    demoSection: { marginTop: '24px' },
    demoDivider: {
        display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '14px',
    },
    demoDividerLine: {
        flex: 1, height: '1px', background: 'rgba(255,255,255,0.09)',
    },
    demoDividerText: {
        fontSize: '0.70rem', color: 'rgba(255,255,255,0.26)',
        whiteSpace: 'nowrap', fontWeight: 500, letterSpacing: '0.04em',
    },
    demoGrid: {
        display: 'flex', gap: '8px', justifyContent: 'space-between',
    },
    demoBtn: {
        flex: 1,
        display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px',
        background: 'rgba(255, 255, 255, 0.05)',
        border: '1px solid rgba(255, 255, 255, 0.09)',
        borderRadius: '14px', padding: '10px 4px',
        cursor: 'pointer',
        transition: 'background 0.2s, border-color 0.2s, transform 0.15s',
    },
    demoAvatar: {
        width: 34, height: 34, borderRadius: '50%',
        background: 'rgba(255, 255, 255, 0.12)',
        border: '1px solid rgba(255, 255, 255, 0.18)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: '0.68rem', fontWeight: 700, color: 'rgba(255,255,255,0.82)',
        flexShrink: 0, letterSpacing: '0.03em',
    },
    demoBtnName: {
        fontSize: '0.68rem', fontWeight: 600,
        color: 'rgba(255,255,255,0.42)',
        letterSpacing: '0.02em',
    },
};

import React, { useState } from 'react';
import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';

/**
 * LoginModal
 * Props:
 *   onSuccess(userData) — called with { user_id, display_name, email, role } on success
 */
const DEMO_SLOTS = [
    { slot: 0, name: 'Alice Chen',    initials: 'AC', color: '#7c3aed' },
    { slot: 1, name: 'Bob Kim',       initials: 'BK', color: '#0077ff' },
    { slot: 2, name: 'Charlie Park',  initials: 'CP', color: '#059669' },
    { slot: 3, name: 'Diana Lee',     initials: 'DL', color: '#dc2626' },
    { slot: 4, name: 'Eve Zhao',      initials: 'EZ', color: '#d97706' },
];

export default function LoginModal({ onSuccess }) {
    const [tab, setTab] = useState('login'); // 'login' | 'register'
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
            if (!lastName.trim()) { setError('Last name is required.'); return false; }
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
        <div style={styles.overlay}>
            {/* Animated background orbs */}
            <div style={styles.orb1} />
            <div style={styles.orb2} />

            <div style={styles.card}>
                {/* Logo */}
                <div style={styles.logoRow}>
                    <div style={styles.logoIcon} />
                    <h1 style={styles.logoText}>InnerLink</h1>
                </div>
                <p style={styles.tagline}>Emotion-aware private messaging</p>

                {/* Tab switcher */}
                <div style={styles.tabRow}>
                    <button
                        id="auth-tab-login"
                        aria-selected={tab === 'login'}
                        style={{ ...styles.tabBtn, ...(tab === 'login' ? styles.tabActive : {}) }}
                        onClick={() => { setTab('login'); resetForm(); }}
                    >
                        Sign In
                    </button>
                    <button
                        id="auth-tab-register"
                        aria-selected={tab === 'register'}
                        style={{ ...styles.tabBtn, ...(tab === 'register' ? styles.tabActive : {}) }}
                        onClick={() => { setTab('register'); resetForm(); }}
                    >
                        Register
                    </button>
                </div>

                <form onSubmit={handleSubmit} style={styles.form} noValidate>
                    <label htmlFor="auth-email" style={styles.label}>Email Address</label>
                    <input
                        id="auth-email"
                        name="email"
                        type="email"
                        autoComplete="email"
                        autoFocus
                        value={email}
                        onChange={e => setEmail(e.target.value)}
                        placeholder="you@example.com"
                        style={styles.input}
                        disabled={loading}
                        aria-required="true"
                    />

                    {tab === 'register' && (
                        <>
                            <label htmlFor="auth-first-name" style={styles.label}>First Name</label>
                            <input
                                id="auth-first-name"
                                name="given-name"
                                type="text"
                                autoComplete="given-name"
                                value={firstName}
                                onChange={e => setFirstName(e.target.value)}
                                placeholder="Jane"
                                style={styles.input}
                                disabled={loading}
                                aria-required="true"
                            />

                            <label htmlFor="auth-last-name" style={styles.label}>Last Name</label>
                            <input
                                id="auth-last-name"
                                name="family-name"
                                type="text"
                                autoComplete="family-name"
                                value={lastName}
                                onChange={e => setLastName(e.target.value)}
                                placeholder="Doe"
                                style={styles.input}
                                disabled={loading}
                                aria-required="true"
                            />


                        </>
                    )}

                    <label htmlFor="auth-password" style={styles.label}>Password</label>
                    <input
                        id="auth-password"
                        name="password"
                        type="password"
                        autoComplete={tab === 'register' ? 'new-password' : 'current-password'}
                        value={password}
                        onChange={e => setPassword(e.target.value)}
                        placeholder="Min. 8 characters"
                        style={styles.input}
                        disabled={loading}
                        aria-required="true"
                    />

                    {tab === 'register' && (
                        <>
                            <label htmlFor="auth-confirm-password" style={styles.label}>Confirm Password</label>
                            <input
                                id="auth-confirm-password"
                                name="confirm-password"
                                type="password"
                                autoComplete="new-password"
                                value={confirmPassword}
                                onChange={e => setConfirmPassword(e.target.value)}
                                placeholder="Repeat password"
                                style={styles.input}
                                disabled={loading}
                                aria-required="true"
                            />
                        </>
                    )}

                    {error && (
                        <div style={styles.errorBox} role="alert">
                            <span>⚠ </span>{error}
                        </div>
                    )}

                    <button id="auth-submit" type="submit" style={styles.submitBtn} disabled={loading} aria-busy={loading}>
                        {loading
                            ? <span style={styles.spinner} />
                            : (tab === 'register' ? 'Create Account' : 'Sign In')}
                    </button>
                </form>

                <p style={styles.switchHint}>
                    {tab === 'login'
                        ? <>No account? <span role="button" tabIndex="0" style={styles.switchLink} onClick={() => { setTab('register'); resetForm(); }}>Register here</span></>
                        : <>Have an account? <span role="button" tabIndex="0" style={styles.switchLink} onClick={() => { setTab('login'); resetForm(); }}>Sign in</span></>
                    }
                </p>

                {/* Demo users */}
                <div style={styles.demoSection}>
                    <div style={styles.demoDivider}>
                        <span style={styles.demoDividerLine} />
                        <span style={styles.demoDividerText}>or try a demo account</span>
                        <span style={styles.demoDividerLine} />
                    </div>
                    <div style={styles.demoGrid}>
                        {DEMO_SLOTS.map(({ slot, name, initials, color }) => (
                            <button
                                key={slot}
                                onClick={() => handleDemoLogin(slot)}
                                disabled={demoLoading !== null || loading}
                                style={{
                                    ...styles.demoBtn,
                                    opacity: (demoLoading !== null || loading) ? 0.6 : 1,
                                }}
                                title={`Log in as ${name}`}
                            >
                                <div style={{
                                    width: 36, height: 36, borderRadius: '50%',
                                    background: color,
                                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                                    fontSize: '0.72rem', fontWeight: 700, color: '#fff',
                                    flexShrink: 0,
                                    position: 'relative',
                                }}>
                                    {demoLoading === slot
                                        ? <span style={{ ...styles.spinner, width: 14, height: 14, borderWidth: 2 }} />
                                        : initials}
                                </div>
                                <span style={styles.demoBtnName}>{name.split(' ')[0]}</span>
                            </button>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}

// ── Inline styles — iOS 26 Zero-Dark (white canvas) ──────────────────────
const styles = {
    overlay: {
        position: 'fixed', inset: 0,
        background: 'radial-gradient(ellipse at 40% 30%, #eef3ff 0%, #ffffff 60%, #fff5f0 100%)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        zIndex: 1000, overflow: 'hidden',
    },
    orb1: {
        position: 'absolute', width: 500, height: 500,
        borderRadius: '50%', top: '-120px', left: '-130px',
        background: 'radial-gradient(circle, rgba(112,0,255,0.09) 0%, transparent 70%)',
        filter: 'blur(60px)',
    },
    orb2: {
        position: 'absolute', width: 460, height: 460,
        borderRadius: '50%', bottom: '-80px', right: '-100px',
        background: 'radial-gradient(circle, rgba(0,119,255,0.09) 0%, transparent 70%)',
        filter: 'blur(60px)',
    },
    card: {
        position: 'relative', zIndex: 1,
        background: 'rgba(255,255,255,0.82)',
        backdropFilter: 'blur(28px) saturate(200%)',
        WebkitBackdropFilter: 'blur(28px) saturate(200%)',
        border: '1px solid rgba(255,255,255,0.92)',
        borderRadius: '28px',
        padding: '48px 44px',
        width: '100%', maxWidth: '420px',
        boxShadow: '0 24px 80px rgba(0,0,0,0.10), inset 0 1px 1px rgba(255,255,255,1)',
    },
    logoRow: { display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '6px' },
    logoIcon: {
        width: '36px', height: '36px', borderRadius: '10px',
        background: 'linear-gradient(135deg, #0077ff 0%, #7000ff 100%)',
        boxShadow: '0 4px 16px rgba(0,119,255,0.35)',
    },
    logoText: { margin: 0, fontSize: '1.8rem', fontWeight: 700, color: '#1c1c2e', letterSpacing: '-0.5px' },
    tagline: { margin: '0 0 28px 0', fontSize: '0.85rem', color: '#6b7280' },
    tabRow: {
        display: 'flex', background: 'rgba(0,0,0,0.04)',
        borderRadius: '10px', padding: '4px', marginBottom: '28px',
    },
    tabBtn: {
        flex: 1, padding: '9px', border: 'none', borderRadius: '8px',
        cursor: 'pointer', background: 'transparent',
        color: '#6b7280', fontSize: '0.9rem', fontWeight: 500,
        transition: 'all 0.2s',
    },
    tabActive: {
        background: 'rgba(0,119,255,0.10)',
        color: '#0077ff',
        boxShadow: '0 2px 8px rgba(0,119,255,0.12)',
    },
    form: { display: 'flex', flexDirection: 'column', gap: '6px' },
    label: {
        fontSize: '0.78rem', fontWeight: 600, color: '#6b7280',
        letterSpacing: '0.05em', textTransform: 'uppercase', marginTop: '10px',
    },
    input: {
        padding: '12px 14px',
        background: 'rgba(255,255,255,0.85)',
        border: '1px solid rgba(0,0,0,0.09)',
        borderRadius: '10px', color: '#1c1c2e',
        fontSize: '0.95rem', outline: 'none',
        transition: 'border-color 0.2s, box-shadow 0.2s',
        boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.04)',
    },
    errorBox: {
        marginTop: '10px', padding: '10px 14px',
        background: 'rgba(255,34,68,0.07)', border: '1px solid rgba(255,34,68,0.25)',
        borderRadius: '8px', color: '#cc0022', fontSize: '0.85rem',
    },
    submitBtn: {
        marginTop: '22px', padding: '14px',
        background: 'linear-gradient(135deg, #0077ff 0%, #7000ff 100%)',
        border: 'none', borderRadius: '12px', color: '#fff',
        fontWeight: 700, fontSize: '1rem', cursor: 'pointer',
        transition: 'opacity 0.2s, transform 0.1s',
        display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '48px',
        boxShadow: '0 6px 20px rgba(0,119,255,0.35)',
    },
    spinner: {
        width: '20px', height: '20px', borderRadius: '50%',
        border: '2px solid rgba(255,255,255,0.35)',
        borderTopColor: '#fff',
        animation: 'spin 0.7s linear infinite',
        display: 'inline-block',
    },
    switchHint: { marginTop: '20px', textAlign: 'center', fontSize: '0.85rem', color: '#6b7280' },
    switchLink: { color: '#0077ff', cursor: 'pointer', textDecoration: 'underline' },
    demoSection: { marginTop: '24px' },
    demoDivider: {
        display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px',
    },
    demoDividerLine: {
        flex: 1, height: '1px', background: 'rgba(0,0,0,0.08)',
    },
    demoDividerText: {
        fontSize: '0.75rem', color: '#9ca3af', whiteSpace: 'nowrap', fontWeight: 500,
    },
    demoGrid: {
        display: 'flex', gap: '8px', justifyContent: 'space-between',
    },
    demoBtn: {
        flex: 1,
        display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px',
        background: 'rgba(0,0,0,0.03)', border: '1px solid rgba(0,0,0,0.06)',
        borderRadius: '12px', padding: '10px 4px',
        cursor: 'pointer', transition: 'background 0.15s, transform 0.1s',
    },
    demoBtnName: {
        fontSize: '0.72rem', fontWeight: 600, color: '#374151',
    },
};

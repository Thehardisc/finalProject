import React, { useState } from 'react';
import axios from 'axios';

const API_BASE = 'http://localhost:8001';

/**
 * LoginModal
 * Props:
 *   onSuccess(userData) — called with { token, user_id, username, role } on success
 */
export default function LoginModal({ onSuccess }) {
    const [tab, setTab] = useState('login'); // 'login' | 'register'
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const resetForm = () => { setUsername(''); setPassword(''); setConfirmPassword(''); setError(''); };

    const validate = () => {
        if (!username.trim()) { setError('Username is required.'); return false; }
        if (username.length < 3) { setError('Username must be at least 3 characters.'); return false; }
        if (!/^[a-zA-Z0-9_.-]+$/.test(username)) {
            setError('Username may only contain letters, numbers, _, -, and .'); return false;
        }
        if (!password) { setError('Password is required.'); return false; }
        if (password.length < 8) { setError('Password must be at least 8 characters.'); return false; }
        if (tab === 'register' && password !== confirmPassword) {
            setError('Passwords do not match.'); return false;
        }
        return true;
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        if (!validate()) return;
        setLoading(true);
        try {
            const endpoint = tab === 'register' ? '/auth/register' : '/auth/login';
            const res = await axios.post(`${API_BASE}${endpoint}`, { username: username.trim(), password });
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
                        style={{ ...styles.tabBtn, ...(tab === 'login' ? styles.tabActive : {}) }}
                        onClick={() => { setTab('login'); resetForm(); }}
                    >
                        Sign In
                    </button>
                    <button
                        id="auth-tab-register"
                        style={{ ...styles.tabBtn, ...(tab === 'register' ? styles.tabActive : {}) }}
                        onClick={() => { setTab('register'); resetForm(); }}
                    >
                        Register
                    </button>
                </div>

                <form onSubmit={handleSubmit} style={styles.form} noValidate>
                    <label style={styles.label}>Username</label>
                    <input
                        id="auth-username"
                        type="text"
                        autoComplete="username"
                        autoFocus
                        value={username}
                        onChange={e => setUsername(e.target.value)}
                        placeholder="your_username"
                        style={styles.input}
                        disabled={loading}
                    />

                    <label style={styles.label}>Password</label>
                    <input
                        id="auth-password"
                        type="password"
                        autoComplete={tab === 'register' ? 'new-password' : 'current-password'}
                        value={password}
                        onChange={e => setPassword(e.target.value)}
                        placeholder="Min. 8 characters"
                        style={styles.input}
                        disabled={loading}
                    />

                    {tab === 'register' && (
                        <>
                            <label style={styles.label}>Confirm Password</label>
                            <input
                                id="auth-confirm-password"
                                type="password"
                                autoComplete="new-password"
                                value={confirmPassword}
                                onChange={e => setConfirmPassword(e.target.value)}
                                placeholder="Repeat password"
                                style={styles.input}
                                disabled={loading}
                            />
                        </>
                    )}

                    {error && (
                        <div style={styles.errorBox}>
                            <span>⚠ </span>{error}
                        </div>
                    )}

                    <button id="auth-submit" type="submit" style={styles.submitBtn} disabled={loading}>
                        {loading
                            ? <span style={styles.spinner} />
                            : (tab === 'register' ? 'Create Account' : 'Sign In')}
                    </button>
                </form>

                <p style={styles.switchHint}>
                    {tab === 'login'
                        ? <>No account? <span style={styles.switchLink} onClick={() => { setTab('register'); resetForm(); }}>Register here</span></>
                        : <>Have an account? <span style={styles.switchLink} onClick={() => { setTab('login'); resetForm(); }}>Sign in</span></>
                    }
                </p>
            </div>
        </div>
    );
}

// ── Inline styles (consistent with glassmorphism theme) ──────────────────────
const styles = {
    overlay: {
        position: 'fixed', inset: 0,
        background: 'radial-gradient(ellipse at 60% 30%, #0a0f1e 0%, #060a14 100%)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        zIndex: 1000, overflow: 'hidden',
    },
    orb1: {
        position: 'absolute', width: 420, height: 420,
        borderRadius: '50%', top: '-80px', left: '-100px',
        background: 'radial-gradient(circle, rgba(0,180,255,0.18) 0%, transparent 70%)',
        animation: 'none', filter: 'blur(40px)',
    },
    orb2: {
        position: 'absolute', width: 380, height: 380,
        borderRadius: '50%', bottom: '-60px', right: '-80px',
        background: 'radial-gradient(circle, rgba(120,0,255,0.18) 0%, transparent 70%)',
        filter: 'blur(40px)',
    },
    card: {
        position: 'relative', zIndex: 1,
        background: 'rgba(255,255,255,0.04)',
        backdropFilter: 'blur(24px)',
        border: '1px solid rgba(255,255,255,0.1)',
        borderRadius: '24px',
        padding: '48px 44px',
        width: '100%', maxWidth: '420px',
        boxShadow: '0 8px 64px rgba(0,0,0,0.6)',
    },
    logoRow: { display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '6px' },
    logoIcon: {
        width: '36px', height: '36px', borderRadius: '10px',
        background: 'linear-gradient(135deg, #00b4ff 0%, #7000ff 100%)',
        boxShadow: '0 0 20px rgba(0,180,255,0.5)',
    },
    logoText: { margin: 0, fontSize: '1.8rem', fontWeight: 700, color: '#fff', letterSpacing: '-0.5px' },
    tagline: { margin: '0 0 28px 0', fontSize: '0.85rem', color: 'rgba(255,255,255,0.4)' },
    tabRow: {
        display: 'flex', background: 'rgba(255,255,255,0.05)',
        borderRadius: '10px', padding: '4px', marginBottom: '28px',
    },
    tabBtn: {
        flex: 1, padding: '9px', border: 'none', borderRadius: '8px',
        cursor: 'pointer', background: 'transparent',
        color: 'rgba(255,255,255,0.5)', fontSize: '0.9rem', fontWeight: 500,
        transition: 'all 0.2s',
    },
    tabActive: {
        background: 'rgba(0,180,255,0.2)',
        color: '#00b4ff',
        boxShadow: '0 0 12px rgba(0,180,255,0.2)',
    },
    form: { display: 'flex', flexDirection: 'column', gap: '6px' },
    label: { fontSize: '0.78rem', fontWeight: 600, color: 'rgba(255,255,255,0.5)', letterSpacing: '0.05em', textTransform: 'uppercase', marginTop: '10px' },
    input: {
        padding: '12px 14px',
        background: 'rgba(255,255,255,0.07)',
        border: '1px solid rgba(255,255,255,0.12)',
        borderRadius: '10px', color: '#fff',
        fontSize: '0.95rem', outline: 'none',
        transition: 'border-color 0.2s',
    },
    errorBox: {
        marginTop: '10px', padding: '10px 14px',
        background: 'rgba(255,60,60,0.12)', border: '1px solid rgba(255,60,60,0.3)',
        borderRadius: '8px', color: '#ff7070', fontSize: '0.85rem',
    },
    submitBtn: {
        marginTop: '22px', padding: '14px',
        background: 'linear-gradient(135deg, #00b4ff 0%, #7000ff 100%)',
        border: 'none', borderRadius: '12px', color: '#fff',
        fontWeight: 700, fontSize: '1rem', cursor: 'pointer',
        transition: 'opacity 0.2s, transform 0.1s',
        display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '48px',
    },
    spinner: {
        width: '20px', height: '20px', borderRadius: '50%',
        border: '2px solid rgba(255,255,255,0.3)',
        borderTopColor: '#fff',
        animation: 'spin 0.7s linear infinite',
        display: 'inline-block',
    },
    switchHint: { marginTop: '20px', textAlign: 'center', fontSize: '0.85rem', color: 'rgba(255,255,255,0.4)' },
    switchLink: { color: '#00b4ff', cursor: 'pointer', textDecoration: 'underline' },
};

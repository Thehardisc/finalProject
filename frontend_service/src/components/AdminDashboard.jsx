import { useState, useEffect, useCallback, useMemo } from 'react';
import { adminAPI } from '../api/client';
import { EmotionPalette } from './EmotionPalette';

const VIOLET  = '139,92,246';
const EMERALD = '52,211,153';
const RED     = '239,68,68';

// Every account wears one of the app's emotion-palette hues (stable per name).
const AVATAR_EMOTIONS = [
  'joy', 'admiration', 'curiosity', 'love', 'optimism', 'pride', 'caring',
  'excitement', 'gratitude', 'relief', 'realization', 'surprise', 'desire',
  'amusement', 'approval', 'confusion', 'nervousness', 'sadness', 'fear',
];
const avatarRgb = (seed = '') => {
  let h = 0;
  for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) >>> 0;
  return EmotionPalette[AVATAR_EMOTIONS[h % AVATAR_EMOTIONS.length]] || EmotionPalette.neutral;
};

const fmtDate = (ts) => ts ? new Date(ts * 1000).toLocaleDateString() : '—';
const fmtExact = (ts) => ts ? new Date(ts * 1000).toLocaleString() : '';
const relTime = (ts) => {
  if (!ts) return 'never';
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60)        return 'just now';
  if (s < 3600)      return `${Math.floor(s / 60)}m ago`;
  if (s < 86400)     return `${Math.floor(s / 3600)}h ago`;
  if (s < 86400 * 7) return `${Math.floor(s / 86400)}d ago`;
  return fmtDate(ts);
};

const MICRO = {
  fontSize: '0.60rem', fontWeight: 700, color: 'rgba(var(--ig-ink-rgb),0.38)',
  textTransform: 'uppercase', letterSpacing: '.10em',
};

function SignalChip({ dot, hollow, value, label }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
      <span style={{
        width: 7, height: 7, borderRadius: '50%', flexShrink: 0,
        background: hollow ? 'transparent' : `rgb(${dot})`,
        border: hollow ? '1.5px solid rgba(var(--ig-ink-rgb),0.30)' : 'none',
        boxShadow: hollow ? 'none' : `0 0 6px rgba(${dot},0.45)`,
      }} />
      <span style={{ fontSize: '0.82rem', fontWeight: 800, color: 'var(--ig-txt)', fontVariantNumeric: 'tabular-nums' }}>{value}</span>
      <span style={{ fontSize: '0.68rem', color: 'var(--ig-txt3)' }}>{label}</span>
    </span>
  );
}

function SkeletonRow() {
  return (
    <tr>
      <td colSpan={6} style={{ padding: '11px 16px' }}>
        <div className="adm-skeleton" style={{ height: 34, borderRadius: 9 }} />
      </td>
    </tr>
  );
}

export default function AdminDashboard({ currentUser }) {
  const [users, setUsers]               = useState([]);
  const [loading, setLoading]           = useState(true);
  const [actionError, setActionError]   = useState('');
  const [confirmModal, setConfirmModal] = useState(null);
  const [actionLoading, setActionLoading] = useState('');
  const [query, setQuery]               = useState('');

  const fetchUsers = useCallback(async () => {
    setLoading(true);
    try {
      const res = await adminAPI.listUsers();
      setUsers(res.data);
      setActionError('');
    } catch (err) {
      setActionError(err.apiMessage || 'Couldn’t load accounts.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchUsers(); }, [fetchUsers]);

  const updateUser = async (userId, payload) => {
    setActionLoading(userId);
    setActionError('');
    try {
      await adminAPI.updateUser(userId, payload);
      await fetchUsers();
    } catch (err) {
      setActionError(err.apiMessage || 'Update failed.');
    } finally {
      setActionLoading('');
      setConfirmModal(null);
    }
  };

  const deleteUser = async (userId) => {
    setActionLoading(userId);
    setActionError('');
    try {
      await adminAPI.deleteUser(userId);
      setUsers(prev => prev.filter(u => u.user_id !== userId));
    } catch (err) {
      setActionError(err.apiMessage || 'Delete failed.');
    } finally {
      setActionLoading('');
      setConfirmModal(null);
    }
  };

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return users;
    return users.filter(u =>
      (u.display_name || '').toLowerCase().includes(q) ||
      (u.email || '').toLowerCase().includes(q));
  }, [users, query]);

  const totalUsers  = users.length;
  const activeUsers = users.filter(u => u.is_active).length;
  const adminCount  = users.filter(u => u.role === 'admin').length;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <style>{`
        @keyframes admSpin    { to { transform: rotate(360deg); } }
        @keyframes admShimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }
        @keyframes admPop     { from { opacity: 0; transform: scale(.95) translateY(6px); } to { opacity: 1; transform: none; } }
        .adm-skeleton {
          background: linear-gradient(90deg, rgba(var(--ig-ink-rgb),0.05) 25%, rgba(var(--ig-ink-rgb),0.10) 50%, rgba(var(--ig-ink-rgb),0.05) 75%);
          background-size: 200% 100%;
          animation: admShimmer 1.4s linear infinite;
        }
        .adm-row { transition: background .12s; }
        .adm-row:hover { background: rgba(var(--ig-ink-rgb),0.03); }
        .adm-acts { opacity: .55; transition: opacity .15s; }
        .adm-row:hover .adm-acts, .adm-acts:focus-within { opacity: 1; }
        .adm-btn {
          padding: 5px 11px; border-radius: 8px; cursor: pointer;
          font-size: 0.74rem; font-weight: 600; white-space: nowrap;
          background: transparent; color: var(--ig-txt2);
          border: 1px solid rgba(var(--ig-ink-rgb),0.14);
          transition: all .13s;
        }
        .adm-btn:hover:not(:disabled)  { border-color: rgba(${VIOLET},0.55); color: rgb(${VIOLET}); background: rgba(${VIOLET},0.07); }
        .adm-btn.danger:hover:not(:disabled) { border-color: rgba(${RED},0.55); color: rgb(${RED}); background: rgba(${RED},0.07); }
        .adm-btn:disabled { opacity: .45; cursor: default; }
        .adm-btn:focus-visible, .adm-input:focus-visible, #admin-refresh-btn:focus-visible {
          outline: 2px solid rgba(${VIOLET},0.60); outline-offset: 1px;
        }
        @media (prefers-reduced-motion: reduce) {
          .adm-skeleton { animation: none; }
          #admin-refresh-btn svg { animation: none !important; }
        }
      `}</style>

      {/* Console bar — counts live with the roster, not as billboards */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 18, flexWrap: 'wrap',
        padding: '12px 16px',
        background: 'var(--ig-surf)',
        border: '1px solid var(--ig-bord)',
        borderRadius: 14,
      }}>
        <span style={MICRO}>Account roster</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
          <SignalChip dot={VIOLET}  value={totalUsers}  label={totalUsers === 1 ? 'account' : 'accounts'} />
          <SignalChip dot={EMERALD} value={activeUsers} label="active" />
          <SignalChip dot={VIOLET} hollow value={adminCount} label={adminCount === 1 ? 'admin' : 'admins'} />
        </div>
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
          <input
            className="adm-input"
            type="search"
            placeholder="Search name or email…"
            value={query}
            onChange={e => setQuery(e.target.value)}
            style={{
              width: 210, padding: '7px 12px',
              background: 'var(--ig-surf2)', border: '1px solid var(--ig-bord)',
              borderRadius: 10, fontSize: '0.8rem', color: 'var(--ig-txt)',
              outline: 'none', fontFamily: 'inherit',
            }}
          />
          <button
            id="admin-refresh-btn"
            onClick={fetchUsers}
            disabled={loading}
            title="Refresh roster"
            style={{
              width: 32, height: 32, borderRadius: 10, cursor: loading ? 'default' : 'pointer',
              background: 'var(--ig-surf2)', border: '1px solid var(--ig-bord)',
              color: 'var(--ig-txt2)', display: 'flex', alignItems: 'center', justifyContent: 'center',
              flexShrink: 0,
            }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4"
              strokeLinecap="round" strokeLinejoin="round"
              style={{ animation: loading ? 'admSpin .9s linear infinite' : 'none' }}>
              <path d="M21 12a9 9 0 1 1-2.64-6.36" />
              <path d="M21 3v6h-6" />
            </svg>
          </button>
        </div>
      </div>

      {actionError && (
        <div style={{
          padding: '10px 14px',
          background: `rgba(${RED},0.07)`, border: `1px solid rgba(${RED},0.22)`,
          borderRadius: 10, color: `rgb(${RED})`, fontSize: '0.82rem',
          display: 'flex', alignItems: 'center', gap: 8,
        }}>
          <span style={{ fontWeight: 700 }}>!</span> {actionError}
        </div>
      )}

      {/* Roster */}
      <div style={{
        background: 'var(--ig-surf)', border: '1px solid var(--ig-bord)',
        borderRadius: 14, overflow: 'hidden',
      }}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', minWidth: 760, borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                {['Account', 'Role', 'Status', 'Joined', 'Last seen', ''].map((h, i) => (
                  <th key={i} style={{
                    ...MICRO,
                    padding: '11px 16px', textAlign: 'left',
                    borderBottom: '1px solid var(--ig-bord)',
                    background: 'var(--ig-surf3)',
                  }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loading && users.length === 0 && [0, 1, 2, 3].map(i => <SkeletonRow key={i} />)}

              {shown.map(user => {
                const isSelf = user.user_id === currentUser.user_id;
                const busy   = actionLoading === user.user_id;
                const aRgb   = avatarRgb(user.display_name || user.email);
                const isAdmin = user.role === 'admin';
                return (
                  <tr key={user.user_id} className="adm-row"
                    style={{ borderBottom: '1px solid rgba(var(--ig-ink-rgb),0.05)', opacity: busy ? 0.5 : 1 }}>

                    <td style={{ padding: '10px 16px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                        <div style={{
                          width: 30, height: 30, borderRadius: '50%', flexShrink: 0,
                          background: `rgba(${aRgb},0.14)`, border: `1px solid rgba(${aRgb},0.28)`,
                          color: `rgb(${aRgb})`, fontSize: '0.78rem', fontWeight: 800,
                          display: 'flex', alignItems: 'center', justifyContent: 'center',
                        }}>
                          {(user.display_name || user.email || '?').charAt(0).toUpperCase()}
                        </div>
                        <div style={{ minWidth: 0 }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--ig-txt)' }}>{user.display_name}</span>
                            {isSelf && (
                              <span style={{
                                fontSize: '0.60rem', fontWeight: 700, padding: '1px 7px', borderRadius: 99,
                                background: `rgba(${VIOLET},0.10)`, color: `rgb(${VIOLET})`,
                                border: `1px solid rgba(${VIOLET},0.25)`,
                              }}>you</span>
                            )}
                          </div>
                          <div style={{ fontSize: '0.72rem', color: 'var(--ig-txt3)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{user.email}</div>
                        </div>
                      </div>
                    </td>

                    <td style={{ padding: '10px 16px' }}>
                      <span style={{
                        fontSize: '0.70rem', fontWeight: 700, padding: '3px 10px', borderRadius: 99,
                        background: isAdmin ? `rgba(${VIOLET},0.10)` : 'rgba(var(--ig-ink-rgb),0.06)',
                        color: isAdmin ? `rgb(${VIOLET})` : 'var(--ig-txt2)',
                        border: isAdmin ? `1px solid rgba(${VIOLET},0.28)` : '1px solid transparent',
                      }}>
                        {isAdmin ? 'Admin' : 'Member'}
                      </span>
                    </td>

                    <td style={{ padding: '10px 16px' }}>
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 7 }}>
                        <span style={{
                          width: 7, height: 7, borderRadius: '50%',
                          background: user.is_active ? `rgb(${EMERALD})` : 'transparent',
                          border: user.is_active ? 'none' : '1.5px solid rgba(var(--ig-ink-rgb),0.30)',
                          boxShadow: user.is_active ? `0 0 6px rgba(${EMERALD},0.45)` : 'none',
                        }} />
                        <span style={{ fontSize: '0.78rem', color: user.is_active ? 'var(--ig-txt2)' : 'var(--ig-txt3)' }}>
                          {user.is_active ? 'Active' : 'Deactivated'}
                        </span>
                      </span>
                    </td>

                    <td style={{ padding: '10px 16px', fontSize: '0.78rem', color: 'var(--ig-txt3)', fontVariantNumeric: 'tabular-nums' }}>
                      {fmtDate(user.created_at)}
                    </td>

                    <td title={fmtExact(user.last_login)}
                      style={{ padding: '10px 16px', fontSize: '0.78rem', color: 'var(--ig-txt3)', fontVariantNumeric: 'tabular-nums' }}>
                      {relTime(user.last_login)}
                    </td>

                    <td style={{ padding: '10px 16px', textAlign: 'right' }}>
                      {isSelf ? (
                        <span style={{ fontSize: '0.72rem', color: 'rgba(var(--ig-ink-rgb),0.25)' }}>—</span>
                      ) : (
                        <div className="adm-acts" style={{ display: 'inline-flex', gap: 6 }}>
                          <button
                            id={`admin-role-btn-${user.user_id}`}
                            className="adm-btn" disabled={busy}
                            onClick={() => setConfirmModal({ type: 'role', user })}
                          >
                            {isAdmin ? 'Demote' : 'Promote'}
                          </button>
                          <button
                            id={`admin-active-btn-${user.user_id}`}
                            className="adm-btn" disabled={busy}
                            onClick={() => updateUser(user.user_id, { is_active: !user.is_active })}
                          >
                            {user.is_active ? 'Deactivate' : 'Activate'}
                          </button>
                          <button
                            id={`admin-delete-btn-${user.user_id}`}
                            className="adm-btn danger" disabled={busy}
                            onClick={() => setConfirmModal({ type: 'delete', user })}
                          >
                            Delete
                          </button>
                        </div>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {!loading && shown.length === 0 && (
          <div style={{ textAlign: 'center', padding: '42px 20px' }}>
            <div style={{ fontSize: '0.86rem', fontWeight: 600, color: 'var(--ig-txt2)', marginBottom: 4 }}>
              {query ? `No accounts match “${query}”` : 'No accounts yet'}
            </div>
            <div style={{ fontSize: '0.74rem', color: 'var(--ig-txt3)' }}>
              {query ? 'Try a different name or email.' : 'New registrations appear here automatically.'}
            </div>
          </div>
        )}
      </div>

      {confirmModal && (
        <div
          onClick={() => setConfirmModal(null)}
          style={{
            position: 'fixed', inset: 0, zIndex: 2000,
            background: 'rgba(0,0,0,0.45)', backdropFilter: 'blur(4px)', WebkitBackdropFilter: 'blur(4px)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
          <div
            onClick={e => e.stopPropagation()}
            style={{
              background: 'var(--ig-surf)', border: '1px solid var(--ig-bord)',
              borderRadius: 20, padding: '26px 28px', maxWidth: 400, width: '90%',
              boxShadow: '0 12px 40px rgba(0,0,0,.18)',
              animation: 'admPop .20s cubic-bezier(.34,1.2,.64,1) both',
            }}>
            {confirmModal.type === 'delete' ? (
              <>
                <h3 style={{ margin: '0 0 10px', fontSize: '1rem', fontWeight: 800, color: `rgb(${RED})` }}>Delete account?</h3>
                <p style={{ margin: 0, fontSize: '0.86rem', color: 'var(--ig-txt2)', lineHeight: 1.55 }}>
                  Permanently delete <strong style={{ color: 'var(--ig-txt)' }}>{confirmModal.user.display_name}</strong> ({confirmModal.user.email})
                  and their access. This can’t be undone.
                </p>
                <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end', marginTop: 22 }}>
                  <button onClick={() => setConfirmModal(null)} style={{
                    padding: '9px 18px', background: 'var(--ig-surf2)', color: 'var(--ig-txt2)',
                    border: 'none', borderRadius: 10, cursor: 'pointer', fontWeight: 600, fontSize: '0.84rem',
                  }}>Cancel</button>
                  <button id="confirm-delete-btn" onClick={() => deleteUser(confirmModal.user.user_id)} style={{
                    padding: '9px 18px', background: `rgb(${RED})`, color: '#fff',
                    border: 'none', borderRadius: 10, cursor: 'pointer', fontWeight: 700, fontSize: '0.84rem',
                  }}>Delete account</button>
                </div>
              </>
            ) : (
              <>
                <h3 style={{ margin: '0 0 10px', fontSize: '1rem', fontWeight: 800, color: `rgb(${VIOLET})` }}>
                  {confirmModal.user.role === 'admin' ? 'Demote to member?' : 'Promote to admin?'}
                </h3>
                <p style={{ margin: 0, fontSize: '0.86rem', color: 'var(--ig-txt2)', lineHeight: 1.55 }}>
                  {confirmModal.user.role === 'admin'
                    ? <>Remove admin access from <strong style={{ color: 'var(--ig-txt)' }}>{confirmModal.user.display_name}</strong>. They keep their account and conversations.</>
                    : <>Grant <strong style={{ color: 'var(--ig-txt)' }}>{confirmModal.user.display_name}</strong> admin access. Admins can manage every account.</>}
                </p>
                <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end', marginTop: 22 }}>
                  <button onClick={() => setConfirmModal(null)} style={{
                    padding: '9px 18px', background: 'var(--ig-surf2)', color: 'var(--ig-txt2)',
                    border: 'none', borderRadius: 10, cursor: 'pointer', fontWeight: 600, fontSize: '0.84rem',
                  }}>Cancel</button>
                  <button id="confirm-role-btn"
                    onClick={() => updateUser(confirmModal.user.user_id, { role: confirmModal.user.role === 'admin' ? 'user' : 'admin' })}
                    style={{
                      padding: '9px 18px', background: 'linear-gradient(135deg,#5b21b6,#6d28d9)', color: '#fff',
                      border: 'none', borderRadius: 10, cursor: 'pointer', fontWeight: 700, fontSize: '0.84rem',
                      boxShadow: '0 3px 12px rgba(109,40,217,.30)',
                    }}>
                    {confirmModal.user.role === 'admin' ? 'Demote' : 'Promote'}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

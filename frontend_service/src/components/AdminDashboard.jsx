import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

const API_BASE = 'http://localhost:8001';

/**
 * AdminDashboard
 * Props:
 *   token        — JWT token for Authorization header
 *   currentUser  — { user_id, display_name, email, role }
 */
export default function AdminDashboard({ token, currentUser }) {
    const [users, setUsers] = useState([]);
    const [loading, setLoading] = useState(true);
    const [actionError, setActionError] = useState('');
    const [confirmModal, setConfirmModal] = useState(null); // { type, user }
    const [actionLoading, setActionLoading] = useState('');

    const headers = { Authorization: `Bearer ${token}` };

    const fetchUsers = useCallback(async () => {
        setLoading(true);
        try {
            const res = await axios.get(`${API_BASE}/admin/users`, { headers });
            setUsers(res.data);
        } catch (err) {
            setActionError(err.response?.data?.detail || 'Failed to load users.');
        } finally {
            setLoading(false);
        }
    }, [token]);

    useEffect(() => { fetchUsers(); }, [fetchUsers]);

    const updateUser = async (userId, payload) => {
        setActionLoading(userId);
        setActionError('');
        try {
            await axios.patch(`${API_BASE}/admin/users/${userId}`, payload, { headers });
            await fetchUsers();
        } catch (err) {
            setActionError(err.response?.data?.detail || 'Update failed.');
        } finally {
            setActionLoading('');
            setConfirmModal(null);
        }
    };

    const deleteUser = async (userId) => {
        setActionLoading(userId);
        setActionError('');
        try {
            await axios.delete(`${API_BASE}/admin/users/${userId}`, { headers });
            setUsers(prev => prev.filter(u => u.user_id !== userId));
        } catch (err) {
            setActionError(err.response?.data?.detail || 'Delete failed.');
        } finally {
            setActionLoading('');
            setConfirmModal(null);
        }
    };

    const totalUsers = users.length;
    const activeUsers = users.filter(u => u.is_active).length;
    const adminCount = users.filter(u => u.role === 'admin').length;

    const fmtDate = (ts) => ts ? new Date(ts * 1000).toLocaleDateString() : '—';
    const fmtTime = (ts) => ts ? new Date(ts * 1000).toLocaleString() : 'Never';

    return (
        <div style={styles.container}>
            <div style={styles.header}>
                <div>
                    <h2 style={styles.title}>👑 Admin Dashboard</h2>
                    <p style={styles.subtitle}>Manage users, roles, and account status</p>
                </div>
                <button id="admin-refresh-btn" onClick={fetchUsers} style={styles.refreshBtn} disabled={loading}>
                    {loading ? '⟳ Loading...' : '⟳ Refresh'}
                </button>
            </div>

            {/* Stats cards */}
            <div style={styles.statsRow}>
                <StatCard label="Total Users" value={totalUsers} icon="👥" color="#00b4ff" />
                <StatCard label="Active" value={activeUsers} icon="✅" color="#00e676" />
                <StatCard label="Admins" value={adminCount} icon="👑" color="#ffd700" />
            </div>

            {actionError && (
                <div style={styles.errorBox}>⚠ {actionError}</div>
            )}

            {/* Users table */}
            <div style={styles.tableWrapper}>
                <table style={styles.table}>
                    <thead>
                        <tr>
                            {['User', 'Role', 'Status', 'Joined', 'Last Login', 'Actions'].map(h => (
                                <th key={h} style={styles.th}>{h}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {users.map(user => {
                            const isSelf = user.user_id === currentUser.user_id;
                            const busy = actionLoading === user.user_id;
                            return (
                                <tr key={user.user_id} style={{ ...styles.tr, opacity: busy ? 0.5 : 1 }}>
                                    <td style={styles.td}>
                                        <span style={styles.usernameCell}>
                                            <span style={{ display: 'flex', flexDirection: 'column' }}>
                                                <span>{user.display_name}</span>
                                                <span style={{ fontSize: '0.75rem', color: 'rgba(255,255,255,0.4)', fontWeight: 'normal' }}>{user.email}</span>
                                            </span>
                                            {isSelf && <span style={styles.youBadge}>you</span>}
                                        </span>
                                    </td>
                                    <td style={styles.td}>
                                        <span style={{
                                            ...styles.roleBadge,
                                            background: user.role === 'admin' ? 'rgba(255,215,0,0.15)' : 'rgba(0,180,255,0.1)',
                                            color: user.role === 'admin' ? '#ffd700' : '#00b4ff',
                                            border: `1px solid ${user.role === 'admin' ? 'rgba(255,215,0,0.3)' : 'rgba(0,180,255,0.2)'}`,
                                        }}>
                                            {user.role === 'admin' ? '👑 Admin' : '👤 User'}
                                        </span>
                                    </td>
                                    <td style={styles.td}>
                                        <span style={{
                                            ...styles.statusDot,
                                            background: user.is_active ? '#00e676' : '#ff5252',
                                        }} />
                                        {user.is_active ? 'Active' : 'Inactive'}
                                    </td>
                                    <td style={{ ...styles.td, color: 'rgba(255,255,255,0.45)', fontSize: '0.8rem' }}>
                                        {fmtDate(user.created_at)}
                                    </td>
                                    <td style={{ ...styles.td, color: 'rgba(255,255,255,0.45)', fontSize: '0.8rem' }}>
                                        {fmtTime(user.last_login)}
                                    </td>
                                    <td style={styles.td}>
                                        {isSelf ? (
                                            <span style={{ color: 'rgba(255,255,255,0.25)', fontSize: '0.78rem' }}>— (you)</span>
                                        ) : (
                                            <div style={styles.actionBtns}>
                                                {/* Toggle role */}
                                                <button
                                                    id={`admin-role-btn-${user.user_id}`}
                                                    style={{ ...styles.actionBtn, ...styles.roleBtn }}
                                                    disabled={busy}
                                                    onClick={() => setConfirmModal({ type: 'role', user })}
                                                    title={user.role === 'admin' ? 'Demote to User' : 'Promote to Admin'}
                                                >
                                                    {user.role === 'admin' ? '↓ Demote' : '↑ Promote'}
                                                </button>

                                                {/* Toggle active */}
                                                <button
                                                    id={`admin-active-btn-${user.user_id}`}
                                                    style={{ ...styles.actionBtn, ...styles.toggleBtn }}
                                                    disabled={busy}
                                                    onClick={() => updateUser(user.user_id, { is_active: !user.is_active })}
                                                >
                                                    {user.is_active ? '🔒 Deactivate' : '🔓 Activate'}
                                                </button>

                                                {/* Delete */}
                                                <button
                                                    id={`admin-delete-btn-${user.user_id}`}
                                                    style={{ ...styles.actionBtn, ...styles.deleteBtn }}
                                                    disabled={busy}
                                                    onClick={() => setConfirmModal({ type: 'delete', user })}
                                                >
                                                    🗑 Delete
                                                </button>
                                            </div>
                                        )}
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
                {!loading && users.length === 0 && (
                    <p style={{ textAlign: 'center', color: 'rgba(255,255,255,0.3)', padding: '40px' }}>No users found.</p>
                )}
            </div>

            {/* Confirm modal */}
            {confirmModal && (
                <div style={styles.confirmOverlay}>
                    <div style={styles.confirmCard}>
                        {confirmModal.type === 'delete' ? (
                            <>
                                <h3 style={{ color: '#ff5252', marginTop: 0 }}>⚠ Delete User</h3>
                                <p>Permanently delete <strong style={{ color: '#fff' }}>{confirmModal.user.display_name}</strong> ({confirmModal.user.email})?<br />
                                    This cannot be undone.</p>
                                <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end', marginTop: '20px' }}>
                                    <button style={styles.cancelBtn} onClick={() => setConfirmModal(null)}>Cancel</button>
                                    <button id="confirm-delete-btn" style={styles.confirmDeleteBtn} onClick={() => deleteUser(confirmModal.user.user_id)}>
                                        Delete
                                    </button>
                                </div>
                            </>
                        ) : (
                            <>
                                <h3 style={{ color: '#ffd700', marginTop: 0 }}>
                                    {confirmModal.user.role === 'admin' ? '↓ Demote to User' : '↑ Promote to Admin'}
                                </h3>
                                <p>
                                    {confirmModal.user.role === 'admin'
                                        ? <>Remove admin privileges from <strong style={{ color: '#fff' }}>{confirmModal.user.display_name}</strong>?</>
                                        : <>Grant admin privileges to <strong style={{ color: '#fff' }}>{confirmModal.user.display_name}</strong>? They will be able to manage all users.</>
                                    }
                                </p>
                                <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end', marginTop: '20px' }}>
                                    <button style={styles.cancelBtn} onClick={() => setConfirmModal(null)}>Cancel</button>
                                    <button id="confirm-role-btn" style={styles.confirmRoleBtn} onClick={() => updateUser(confirmModal.user.user_id, { role: confirmModal.user.role === 'admin' ? 'user' : 'admin' })}>
                                        Confirm
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

function StatCard({ label, value, icon, color }) {
    return (
        <div style={{ ...statCardStyle.card, borderColor: `${color}30` }}>
            <div style={{ fontSize: '1.6rem' }}>{icon}</div>
            <div style={{ ...statCardStyle.value, color }}>{value}</div>
            <div style={statCardStyle.label}>{label}</div>
        </div>
    );
}

const statCardStyle = {
    card: {
        flex: 1, minWidth: '130px',
        background: 'rgba(255,255,255,0.04)',
        border: '1px solid rgba(255,255,255,0.1)',
        borderRadius: '16px', padding: '20px',
        textAlign: 'center',
        display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px',
    },
    value: { fontSize: '2rem', fontWeight: 700 },
    label: { fontSize: '0.78rem', color: 'rgba(255,255,255,0.4)', textTransform: 'uppercase', letterSpacing: '0.06em' },
};

const styles = {
    container: { display: 'flex', flexDirection: 'column', gap: '24px', height: '100%', overflow: 'auto', paddingBottom: '40px' },
    header: { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' },
    title: { margin: 0, fontSize: '1.6rem', fontWeight: 700, color: '#fff' },
    subtitle: { margin: '6px 0 0', color: 'rgba(255,255,255,0.4)', fontSize: '0.85rem' },
    refreshBtn: {
        padding: '8px 18px', background: 'rgba(255,255,255,0.08)',
        border: '1px solid rgba(255,255,255,0.12)', borderRadius: '8px',
        color: '#fff', cursor: 'pointer', fontSize: '0.85rem',
    },
    statsRow: { display: 'flex', gap: '16px', flexWrap: 'wrap' },
    errorBox: {
        padding: '12px 16px', background: 'rgba(255,60,60,0.1)',
        border: '1px solid rgba(255,60,60,0.25)', borderRadius: '10px',
        color: '#ff7070', fontSize: '0.88rem',
    },
    tableWrapper: {
        background: 'rgba(255,255,255,0.03)',
        border: '1px solid rgba(255,255,255,0.08)',
        borderRadius: '16px', overflow: 'hidden',
    },
    table: { width: '100%', borderCollapse: 'collapse' },
    th: {
        padding: '14px 16px', textAlign: 'left',
        background: 'rgba(255,255,255,0.04)',
        color: 'rgba(255,255,255,0.5)', fontSize: '0.75rem',
        fontWeight: 600, letterSpacing: '0.07em', textTransform: 'uppercase',
        borderBottom: '1px solid rgba(255,255,255,0.06)',
    },
    tr: { borderBottom: '1px solid rgba(255,255,255,0.04)', transition: 'background 0.15s' },
    td: { padding: '13px 16px', color: 'rgba(255,255,255,0.85)', fontSize: '0.88rem', verticalAlign: 'middle' },
    usernameCell: { display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 500 },
    youBadge: {
        fontSize: '0.68rem', padding: '1px 7px',
        background: 'rgba(0,180,255,0.15)', color: '#00b4ff',
        border: '1px solid rgba(0,180,255,0.25)', borderRadius: '20px',
    },
    roleBadge: { fontSize: '0.78rem', padding: '3px 10px', borderRadius: '20px', fontWeight: 600 },
    statusDot: { display: 'inline-block', width: '7px', height: '7px', borderRadius: '50%', marginRight: '6px' },
    actionBtns: { display: 'flex', gap: '6px', flexWrap: 'wrap' },
    actionBtn: {
        padding: '5px 10px', border: 'none', borderRadius: '6px',
        cursor: 'pointer', fontSize: '0.78rem', fontWeight: 500,
        transition: 'opacity 0.15s, transform 0.1s', whiteSpace: 'nowrap',
    },
    roleBtn: { background: 'rgba(255,215,0,0.12)', color: '#ffd700', border: '1px solid rgba(255,215,0,0.2)' },
    toggleBtn: { background: 'rgba(0,230,118,0.1)', color: '#00e676', border: '1px solid rgba(0,230,118,0.2)' },
    deleteBtn: { background: 'rgba(255,82,82,0.1)', color: '#ff5252', border: '1px solid rgba(255,82,82,0.2)' },
    confirmOverlay: {
        position: 'fixed', inset: 0, zIndex: 2000,
        background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(4px)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
    },
    confirmCard: {
        background: 'rgba(15,20,35,0.97)', border: '1px solid rgba(255,255,255,0.12)',
        borderRadius: '20px', padding: '32px 36px', maxWidth: '400px', width: '90%',
        color: 'rgba(255,255,255,0.75)',
    },
    cancelBtn: {
        padding: '10px 20px', background: 'rgba(255,255,255,0.08)',
        border: '1px solid rgba(255,255,255,0.12)', borderRadius: '8px',
        color: '#fff', cursor: 'pointer',
    },
    confirmDeleteBtn: {
        padding: '10px 20px', background: 'rgba(255,82,82,0.85)',
        border: 'none', borderRadius: '8px', color: '#fff', cursor: 'pointer', fontWeight: 600,
    },
    confirmRoleBtn: {
        padding: '10px 20px', background: 'rgba(255,215,0,0.85)',
        border: 'none', borderRadius: '8px', color: '#000', cursor: 'pointer', fontWeight: 600,
    },
};

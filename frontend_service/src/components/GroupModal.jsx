import React, { useState } from 'react';

/**
 * GroupModal — Create a new group conversation.
 * Props:
 *   globalUsers       — list of { user_id, display_name }
 *   currentUserId     — exclude self from picker
 *   onConfirm(name, memberIds) — called when the user submits
 *   onClose           — called to dismiss
 */
const GroupModal = ({ globalUsers, currentUserId, onConfirm, onClose }) => {
    const [groupName,    setGroupName]    = useState('');
    const [selectedIds,  setSelectedIds]  = useState([]);
    const [error,        setError]        = useState('');
    const [loading,      setLoading]      = useState(false);

    const eligibleUsers = globalUsers.filter(u => u.user_id !== currentUserId);

    const toggleUser = (userId) => {
        setSelectedIds(prev =>
            prev.includes(userId)
                ? prev.filter(id => id !== userId)
                : [...prev, userId]
        );
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        if (!groupName.trim()) { setError('Group name is required.'); return; }
        if (selectedIds.length === 0) { setError('Select at least one other member.'); return; }
        setLoading(true);
        try {
            await onConfirm(groupName.trim(), selectedIds);
        } catch {
            setError('Failed to create group. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-box glass" onClick={e => e.stopPropagation()}>
                <div className="modal-header">
                    <h3>New Group Chat</h3>
                    <button className="modal-close" onClick={onClose}>✕</button>
                </div>

                <form onSubmit={handleSubmit} className="modal-body">
                    <div className="modal-field">
                        <label>Group Name</label>
                        <input
                            className="glass modal-input"
                            type="text"
                            placeholder="e.g. Team Chat"
                            maxLength={100}
                            value={groupName}
                            onChange={e => setGroupName(e.target.value)}
                            autoFocus
                        />
                    </div>

                    <div className="modal-field">
                        <label>Add Members ({selectedIds.length} selected)</label>
                        <div className="member-picker">
                            {eligibleUsers.length === 0 ? (
                                <p style={{ color: '#64748b', padding: '8px 0' }}>No other users available.</p>
                            ) : (
                                eligibleUsers.map(u => (
                                    <label key={u.user_id} className="member-row">
                                        <input
                                            type="checkbox"
                                            checked={selectedIds.includes(u.user_id)}
                                            onChange={() => toggleUser(u.user_id)}
                                        />
                                        <span className="member-avatar">
                                            {u.display_name.charAt(0).toUpperCase()}
                                        </span>
                                        <span className="member-name">{u.display_name}</span>
                                    </label>
                                ))
                            )}
                        </div>
                    </div>

                    {error && <p className="modal-error">{error}</p>}

                    <div className="modal-actions">
                        <button type="button" className="btn-ghost" onClick={onClose}>Cancel</button>
                        <button type="submit" className="btn-primary" disabled={loading}>
                            {loading ? 'Creating...' : 'Create Group'}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
};

export default GroupModal;

/**
 * src/api/client.js — Centralized axios API client.
 *
 * Single source of truth for:
 *   - Base URL (reads from env or falls back to localhost)
 *   - Credentials (cookies sent automatically)
 *   - 401 → auto-redirect to login
 *   - Consistent error shape for all consumers
 */
import axios from 'axios';

export const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';
export const WS_BASE  = API_BASE.replace(/^http/, 'ws');

const client = axios.create({
    baseURL:         API_BASE,
    withCredentials: true,
    timeout:         15_000,
    headers: {
        'Content-Type': 'application/json',
    },
});

// ── Response interceptor ────────────────────────────────────────────────────
client.interceptors.response.use(
    (response) => response,
    (error) => {
        const status = error.response?.status;

        // Auto-redirect to login on session expiry
        if (status === 401) {
            // Avoid infinite loop if the error is ON the /auth/me route itself
            const url = error.config?.url || '';
            if (!url.includes('/auth/me') && !url.includes('/auth/login')) {
                window.location.href = '/';
            }
        }

        // Normalise error shape so callers can always do: e.apiMessage
        const apiMessage =
            error.response?.data?.detail ||
            error.response?.data?.message ||
            error.message ||
            'Unknown error';

        return Promise.reject(Object.assign(error, { apiMessage }));
    }
);

export default client;

// ── Typed API surface ───────────────────────────────────────────────────────
// Auth
export const authAPI = {
    me:       ()         => client.get('/auth/me'),
    login:    (body)     => client.post('/auth/login',    body),
    register: (body)     => client.post('/auth/register', body),
    logout:   ()         => client.post('/auth/logout'),
};

// Users & conversations
export const usersAPI = {
    list:              (currentUserId) => client.get('/users', { params: { current_user_id: currentUserId } }),
    createConversation:(body)          => client.post('/conversations', body),
    myConversations:   (userId)        => client.get(`/conversations/${userId}`),
    conversationState: (convId)        => client.get(`/conversation/${convId}/state`),
    messages:          (convId, limit) => client.get(`/conversation/${convId}/messages`, { params: { limit } }),
};

// Health & analytics
export const systemAPI = {
    status:      () => client.get('/health/status', { validateStatus: s => s === 200 || s === 503 }),
    calibration: () => client.get('/analytics/calibration'),
};

// Feedback
export const feedbackAPI = {
    post: (messageId, label) => client.post(`/message/${messageId}/feedback`, { label }),
};

// Admin
export const adminAPI = {
    listUsers:   ()               => client.get('/admin/users'),
    updateUser:  (userId, body)   => client.patch(`/admin/users/${userId}`, body),
    deleteUser:  (userId)         => client.delete(`/admin/users/${userId}`),
};

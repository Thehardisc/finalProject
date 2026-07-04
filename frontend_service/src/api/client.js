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

client.interceptors.response.use(
    (response) => response,
    (error) => {
        const status = error.response?.status;

        if (status === 401) {
            const url = error.config?.url || '';
            if (!url.includes('/auth/me') && !url.includes('/auth/login')) {
                window.location.href = '/';
            }
        }

        const apiMessage =
            error.response?.data?.detail ||
            error.response?.data?.message ||
            error.message ||
            'Unknown error';

        return Promise.reject(Object.assign(error, { apiMessage }));
    }
);

export default client;

export const authAPI = {
    me:       ()         => client.get('/auth/me'),
    login:    (body)     => client.post('/auth/login',    body),
    register: (body)     => client.post('/auth/register', body),
    logout:   ()         => client.post('/auth/logout'),
};

export const usersAPI = {
    list:              (currentUserId) => client.get('/users', { params: { current_user_id: currentUserId } }),
    createConversation:(body)          => client.post('/conversations', body),
    createGroup:       (name, memberIds) => client.post('/conversations/group', { name, member_ids: memberIds }),
    addMember:         (convId, userId)  => client.post(`/conversations/${convId}/members`, { user_id: userId }),
    removeMember:      (convId, userId)  => client.delete(`/conversations/${convId}/members/${userId}`),
    myConversations:   (userId)        => client.get(`/conversations/${userId}`),
    conversationState: (convId)        => client.get(`/conversation/${convId}/state`),
    messages:          (convId, limit) => client.get(`/conversation/${convId}/messages`, { params: { limit } }),
};

export const systemAPI = {
    status:      () => client.get('/health/status', { validateStatus: s => s === 200 || s === 503 }),
    calibration: () => client.get('/analytics/calibration'),
};

export const feedbackAPI = {
    post:   (messageId, label) => client.post(`/message/${messageId}/feedback`, { label }),
    delete: (messageId)        => client.delete(`/message/${messageId}`),
};

export const adminAPI = {
    listUsers:        ()                     => client.get('/admin/users'),
    updateUser:       (userId, body)         => client.patch(`/admin/users/${userId}`, body),
    deleteUser:       (userId)               => client.delete(`/admin/users/${userId}`),
    recentAnalyses:   (limit = 50, convId)   => client.get('/admin/recent-analyses', {
        params: { limit, ...(convId ? { conversation_id: convId } : {}) },
    }),
    pipelineDetail:   (messageId)            => client.get(`/admin/pipeline/${messageId}`),
    startAiDemo:      (topic, numMessages)  => client.post('/admin/ai-demo', { topic, num_messages: numMessages }),
    aiDemoStatus:     ()                     => client.get('/admin/ai-demo/status'),
    stopAiDemo:       ()                     => client.post('/admin/ai-demo/stop'),
};

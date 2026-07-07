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
    verifyEmotion:    (messageId, emotion, verified = true) =>
        client.post(`/admin/verify/${messageId}`, { emotion, verified }),
    startAiDemo:      (topic, numMessages)  => client.post('/admin/ai-demo', { topic, num_messages: numMessages }),
    aiDemoStatus:     ()                     => client.get('/admin/ai-demo/status'),
    stopAiDemo:       ()                     => client.post('/admin/ai-demo/stop'),
};

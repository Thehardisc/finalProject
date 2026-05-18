import { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE } from '../constants/emotions';

/**
 * useConversations
 * Manages the list of conversations and global user list.
 * Polls every 5 seconds to pick up new chats.
 */
export function useConversations(currentUser) {
    const [conversations, setConversations] = useState([]);
    const [globalUsers,   setGlobalUsers]   = useState([]);

    useEffect(() => {
        if (!currentUser) return;
        const fetchChats = async () => {
            try {
                const [chatsRes, usersRes] = await Promise.all([
                    axios.get(`${API_BASE}/conversations/${currentUser.user_id}`),
                    axios.get(`${API_BASE}/users?current_user_id=${currentUser.user_id}`)
                ]);
                setConversations(chatsRes.data || []);
                setGlobalUsers(usersRes.data   || []);
            } catch (e) {
                console.error('Failed to load initial panel data');
            }
        };
        fetchChats();
        const intl = setInterval(fetchChats, 5000);
        return () => clearInterval(intl);
    }, [currentUser]);

    const handleCreateChat = async (targetId) => {
        try {
            const res = await axios.post(`${API_BASE}/conversations`, {
                user_id:        currentUser.user_id,
                target_user_id: targetId
            });
            return res.data.conversation_id;
        } catch (e) {
            console.error('Failed to make chat', e);
            return null;
        }
    };

    return { conversations, globalUsers, handleCreateChat };
}

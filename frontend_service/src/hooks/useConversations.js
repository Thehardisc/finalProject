import { useState, useEffect, useCallback } from 'react';
import { usersAPI } from '../api/client';

/**
 * useConversations
 * Manages the list of conversations (direct + group) and global user list.
 * Polls every 5 seconds to pick up new chats.
 */
export function useConversations(currentUser) {
    const [conversations, setConversations] = useState([]);
    const [globalUsers,   setGlobalUsers]   = useState([]);

    const fetchChats = useCallback(async () => {
        if (!currentUser) return;
        try {
            const [chatsRes, usersRes] = await Promise.all([
                usersAPI.myConversations(currentUser.user_id),
                usersAPI.list(currentUser.user_id)
            ]);
            const all = chatsRes.data || [];
            // Filter out any self-conversations (defensive guard for legacy data)
            const filtered = all.filter(c =>
                c.type === 'group' || c.other_user_id !== currentUser.user_id
            );
            setConversations(filtered);
            setGlobalUsers(usersRes.data || []);
        } catch (e) {
            console.error('Failed to load panel data');
        }
    }, [currentUser]);

    useEffect(() => {
        if (!currentUser) return;
        fetchChats();
        const intl = setInterval(fetchChats, 5000);
        return () => clearInterval(intl);
    }, [currentUser, fetchChats]);

    const handleCreateChat = async (targetId) => {
        try {
            const res = await usersAPI.createConversation({
                user_id:        currentUser.user_id,
                target_user_id: targetId
            });
            return res.data.conversation_id;
        } catch (e) {
            throw new Error(e.apiMessage || 'Failed to create chat.');
        }
    };

    const handleCreateGroup = async (name, memberIds) => {
        try {
            const res = await usersAPI.createGroup(name, memberIds);
            return res.data.conversation_id;
        } catch (e) {
            throw new Error(e.apiMessage || 'Failed to create group.');
        }
    };

    const handleAddMember = async (convId, userId) => {
        await usersAPI.addMember(convId, userId);  // throws on error
        await fetchChats();                         // immediate refresh
    };

    const handleRemoveMember = async (convId, userId) => {
        await usersAPI.removeMember(convId, userId); // throws on error
        await fetchChats();                          // immediate refresh
    };

    return {
        conversations, globalUsers, fetchChats,
        handleCreateChat, handleCreateGroup,
        handleAddMember, handleRemoveMember
    };
}

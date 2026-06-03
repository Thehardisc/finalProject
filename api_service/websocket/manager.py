"""
api_service/websocket/manager.py — WebSocket connection registry.
"""
from fastapi import WebSocket
from shared.utils.logger import get_logger

logger = get_logger("api_service")


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)
        logger.info(f"Client {user_id} connected. Active: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket, user_id: str):
        if user_id in self.active_connections:
            try:
                self.active_connections[user_id].remove(websocket)
                if not self.active_connections[user_id]:
                    del self.active_connections[user_id]
            except ValueError:
                pass
        logger.info(f"Client {user_id} disconnected. Active: {len(self.active_connections)}")

    async def broadcast_to_user(self, user_id: str, message: dict):
        if user_id not in self.active_connections:
            return
        dead = []
        for connection in list(self.active_connections[user_id]):
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to {user_id}: {e}")
                dead.append(connection)
        for conn in dead:
            self.disconnect(conn, user_id)
        if dead:
            logger.info(f"Removed {len(dead)} stale connection(s) for user {user_id}.")


# Shared singleton used across the api_service package
manager = ConnectionManager()

from pydantic import BaseModel
from typing import Optional, Dict

class MessageEvent(BaseModel):
    conversation_id: str
    user_id: str
    text: str
    timestamp: float
    message_id: str
    metadata: Optional[Dict] = {}

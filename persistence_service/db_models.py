from sqlalchemy import Column, Integer, String, Float, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
import uuid
import time

def gen_uuid():
    return str(uuid.uuid4())

Base = declarative_base()

class Message(Base):
    __tablename__ = 'messages'
    message_id = Column(String, primary_key=True)
    conversation_id = Column(String, index=True)
    user_id = Column(String)
    text = Column(Text)
    timestamp = Column(Float, index=True)
    
class EmotionAnalysis(Base):
    __tablename__ = 'emotion_analysis'
    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(String, index=True)
    emotions_json = Column(Text)
    reasoning_json = Column(Text, nullable=True)
    pipeline_log_json = Column(Text, nullable=True)
    ground_truth_emotion = Column(String, nullable=True)
    is_verified = Column(Boolean, default=False)

class ConversationState(Base):
    __tablename__ = 'conversation_states'
    conversation_id = Column(String, primary_key=True)
    state_json = Column(Text)
    escalation_score = Column(Float, default=0.0)
    last_updated = Column(Float)

class User(Base):
    __tablename__ = 'users'
    user_id = Column(String, primary_key=True, default=gen_uuid)
    username = Column(String, unique=True, index=True, nullable=True)
    email = Column(String, unique=True, index=True, nullable=True)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    display_name = Column(String, nullable=True)
    password_hash = Column(String, nullable=True)
    role = Column(String, default="user")
    is_active = Column(Boolean, default=True)
    created_at = Column(Float, default=time.time)
    last_login = Column(Float, nullable=True)

class Conversation(Base):
    __tablename__ = 'conversations'
    conversation_id = Column(String, primary_key=True, default=gen_uuid)
    type = Column(String, default="direct")
    created_at = Column(Float, default=time.time)

class ConversationParticipant(Base):
    __tablename__ = 'conversation_participants'
    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String, index=True)
    user_id = Column(String, index=True)
    joined_at = Column(Float, default=time.time)

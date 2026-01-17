from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Message(Base):
    __tablename__ = 'messages'
    message_id = Column(String, primary_key=True)
    conversation_id = Column(String, index=True)
    user_id = Column(String)
    text = Column(Text)
    timestamp = Column(Float)
    
class EmotionAnalysis(Base):
    __tablename__ = 'emotion_analysis'
    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(String, index=True)
    emotions_json = Column(Text)
    reasoning_json = Column(Text, nullable=True) # For Explainability
    pipeline_log_json = Column(Text, nullable=True) # For Debugging

class ConversationState(Base):
    __tablename__ = 'conversation_states'
    conversation_id = Column(String, primary_key=True)
    state_json = Column(Text)
    last_updated = Column(Float)

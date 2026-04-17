from sqlalchemy import Column, Integer, String, Float, Text, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base

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
    reasoning_json = Column(Text, nullable=True) # For Explainability
    pipeline_log_json = Column(Text, nullable=True) # For Debugging
    ground_truth_emotion = Column(String, nullable=True)
    is_verified = Column(Boolean, default=False)

class ConversationState(Base):
    __tablename__ = 'conversation_states'
    conversation_id = Column(String, primary_key=True)
    state_json = Column(Text)
    escalation_score = Column(Float, default=0.0)
    last_updated = Column(Float)

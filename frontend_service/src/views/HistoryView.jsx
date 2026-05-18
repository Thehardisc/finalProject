import React from 'react';
import { EMOTION_COLORS } from '../constants/emotions';

/**
 * HistoryView — Scrollable archive of all messages with inline feedback selectors.
 */
const HistoryView = ({ messages, handleFeedback, handleHistoryClick }) => (
    <div className="history-view">
        <h3>Conversation Archive</h3>
        <div className="history-list">
            {messages.map(msg => (
                <div
                    key={msg.id}
                    className="history-item"
                    onClick={() => handleHistoryClick(msg)}
                    style={{ cursor: 'pointer' }}
                >
                    <div className="history-item-header">
                        <span>{msg.senderName}</span>
                        <span>{new Date(msg.id).toLocaleTimeString()}</span>
                    </div>
                    <div className="history-item-text">{msg.text}</div>
                    <div className="history-footer">
                        {msg.analysis && (
                            <div
                                className="history-item-emotion"
                                style={{
                                    background: EMOTION_COLORS[msg.analysis.data.final_dominant_emotion?.toLowerCase()] || '#888'
                                }}
                            >
                                {msg.analysis.data.final_dominant_emotion}
                            </div>
                        )}
                        <select
                            className="history-feedback-select"
                            onClick={e => e.stopPropagation()}
                            onChange={e => handleFeedback(msg.id, e.target.value)}
                            value={msg.feedbackLabel || ''}
                        >
                            <option value="" disabled>✎</option>
                            {Object.keys(EMOTION_COLORS).map(emo => (
                                <option key={emo} value={emo}>{emo}</option>
                            ))}
                        </select>
                    </div>
                </div>
            ))}
        </div>
    </div>
);

export default HistoryView;

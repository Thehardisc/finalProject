import React from 'react';
import { EMOTIONAL_WORDS, EMOTION_COLORS } from '../constants/emotions';

/**
 * EmotionalTextOverlay
 * Highlights emotionally charged words with their corresponding emotion colour.
 */
const EmotionalTextOverlay = ({ text }) => {
    if (!text) return null;
    const words = text.split(/\s+/);
    return (
        <div className="text-overlay" style={{ fontSize: '1.2rem', lineHeight: '1.6', marginBottom: '20px' }}>
            {words.map((word, i) => {
                const clean = word.toLowerCase().replace(/[^a-z]/g, '');
                const emo   = EMOTIONAL_WORDS[clean];
                if (emo) {
                    const style = {
                        color:       EMOTION_COLORS[emo] || 'white',
                        textShadow:  `0 0 10px ${EMOTION_COLORS[emo]}`,
                        fontWeight:  'bold',
                        marginRight: '5px',
                        cursor:      'help'
                    };
                    return <span key={i} style={style} title={`${emo} impact detected`}>{word} </span>;
                }
                return <span key={i} style={{ marginRight: '5px', color: '#e2e8f0' }}>{word} </span>;
            })}
        </div>
    );
};

export default EmotionalTextOverlay;

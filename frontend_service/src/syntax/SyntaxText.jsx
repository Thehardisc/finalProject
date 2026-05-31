import React, { useMemo, useState } from 'react';
import { classify, ROLES, roleColor } from './highlighter';

// <SyntaxText text="..." theme="prism" dark={false} />
export default function SyntaxText({ text, theme = 'prism', dark = false }) {
  const tokens = useMemo(() => classify(text), [text]);
  const [hoveredIdx, setHoveredIdx] = useState(null);

  if (!tokens || tokens.length === 0) return null;

  return (
    <div style={{ display: 'inline', position: 'relative' }}>
      {tokens.map((t, idx) => {
        const r = t.role;
        if (r === 'default' || !t.word) {
          return <span key={idx} style={{ color: roleColor(theme, 'default', dark).color }}>{t.text}{t.ws}</span>;
        }

        const isHovered = hoveredIdx === idx;
        const confPct = Math.round((t.conf || 0.7) * 100);
        const { color, tint } = roleColor(theme, r, dark);

        return (
          <span
            key={idx}
            onMouseEnter={() => setHoveredIdx(idx)}
            onMouseLeave={() => setHoveredIdx(null)}
            style={{
              position: 'relative',
              display: 'inline-block',
              cursor: 'default',
              padding: '0 2px',
              margin: '0 -2px',
              borderRadius: '4px',
              color: color,
              backgroundColor: isHovered ? tint : 'transparent',
              fontWeight: r !== 'default' ? 600 : 400,
              textDecoration: isHovered ? `underline dashed ${color} 1.5px` : 'none',
              textUnderlineOffset: '3px',
              transition: 'background-color 0.15s, text-decoration 0.15s',
            }}
          >
            {t.text}
            
            {/* Tooltip */}
            {isHovered && (
              <div
                style={{
                  position: 'absolute',
                  bottom: '100%',
                  left: '50%',
                  transform: 'translateX(-50%) translateY(-6px)',
                  backgroundColor: dark ? '#1a1a2e' : '#ffffff',
                  border: `1.5px solid ${color}`,
                  borderRadius: '10px',
                  padding: '8px 12px',
                  zIndex: 100,
                  boxShadow: '0 8px 24px rgba(0,0,0,0.18)',
                  minWidth: '150px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '4px',
                  pointerEvents: 'none',
                  animation: 'syntaxTooltipIn 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275) both',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.70rem', fontWeight: 700, color, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                    {ROLES[r]?.label || r}
                  </span>
                  <span style={{ fontSize: '0.65rem', fontWeight: 600, color: dark ? '#888' : '#aaa' }}>
                    {confPct}%
                  </span>
                </div>
                
                <div style={{ fontSize: '0.75rem', color: dark ? '#ccc' : '#444', lineHeight: 1.3 }}>
                  {ROLES[r]?.note}
                </div>

                {t.emotion && (
                  <div style={{ 
                    marginTop: '4px', 
                    paddingTop: '4px', 
                    borderTop: `1px solid ${dark ? '#333' : '#eee'}`,
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px'
                  }}>
                    <span style={{ fontSize: '0.65rem', color: dark ? '#888' : '#aaa' }}>Emotion:</span>
                    <span style={{ fontSize: '0.75rem', fontWeight: 600, color: color, textTransform: 'capitalize' }}>
                      {t.emotion}
                    </span>
                  </div>
                )}
              </div>
            )}
            
            <span style={{ color: roleColor(theme, 'default', dark).color }}>{t.ws}</span>
          </span>
        );
      })}
      
      {/* Inject animation keyframes safely (only once) */}
      <style>{`
        @keyframes syntaxTooltipIn {
          0% { opacity: 0; transform: translateX(-50%) translateY(4px) scale(0.95); }
          100% { opacity: 1; transform: translateX(-50%) translateY(-6px) scale(1); }
        }
      `}</style>
    </div>
  );
}

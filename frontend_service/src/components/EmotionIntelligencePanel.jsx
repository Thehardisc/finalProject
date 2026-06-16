import React, { useEffect, useState } from 'react';
import { EmotionPalette } from './EmotionPalette';
import EmotionArcChart from './EmotionArcChart';

export default function EmotionIntelligencePanel({ lastAnalysis, processing, partialModels = new Set(), messages = [] }) {
  const data = lastAnalysis?.data;
  const [animKey, setAnimKey] = useState(0);
  const [prevDom, setPrevDom] = useState(null);

  const dom = data?.final_dominant_emotion?.toLowerCase() || 'neutral';
  const rgb = EmotionPalette[dom] || EmotionPalette.neutral;
  const confidence = data?.meta_confidence ?? null;
  const valence = data?.final_valence ?? 0;
  const valencePct = Math.round(((valence + 1) / 2) * 100);

  useEffect(() => {
    if (dom !== prevDom) { setPrevDom(dom); setAnimKey(k => k + 1); }
  }, [dom]);

  const topEmos = [...(data?.bert_emotions || [])].sort((a, b) => b.score - a.score).slice(0, 6);
  const valenceHistory = messages
    .filter(m => m.analysis?.data?.final_valence != null)
    .map(m => m.analysis.data.final_valence)
    .slice(-24);
  const traj   = data?.lstm_trajectory;
  const snap   = data?.context_snapshot;
  const sarcasm = data?.sarcasm_score ?? data?.llm_sarcasm_score ?? 0;
  const shift  = data?.context_shift;
  const logicMap = data?.logic_map || {};

  const BORDER = 'rgba(255,255,255,0.07)';
  const MUTED  = 'var(--text-muted, #4b5563)';
  const TEXT   = 'var(--text-primary, #ededf5)';
  const SECT   = { fontSize: '0.58rem', fontWeight: 700, color: '#4b5563', textTransform: 'uppercase', letterSpacing: '.10em', marginBottom: 10 };

  const PIPELINE_STAGES = [
    { key: 'vader',          label: 'VADER'   },
    { key: 'basic_bert',     label: 'BERT'    },
    { key: 'go_emotions',    label: 'GoEmo'   },
    { key: 'context_engine', label: 'Context' },
    { key: 'meta',           label: 'Meta'    },
  ];

  if (!data && !processing) {
    return (
      <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 16, background: 'var(--bg-void, #03030a)' }}>
        <div style={{
          width: 56, height: 56, borderRadius: '50%',
          background: 'rgba(99,102,241,0.08)',
          border: '1px solid rgba(99,102,241,0.14)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '1.6rem', opacity: 0.5,
        }}>🧠</div>
        <div style={{ fontSize: '0.76rem', fontWeight: 600, textAlign: 'center', lineHeight: 1.7, color: '#374151' }}>
          Send a message to<br />activate emotion analysis
        </div>
      </div>
    );
  }

  if (processing && !data) {
    return (
      <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 20, background: 'var(--bg-void, #03030a)' }}>
        <div style={{ position: 'relative', width: 60, height: 60 }}>
          <div style={{ position: 'absolute', inset: 0, borderRadius: '50%', border: `2px solid rgba(${rgb},0.2)`, animation: 'eiSpin 2s linear infinite' }} />
          <div style={{ position: 'absolute', inset: 8, borderRadius: '50%', background: `radial-gradient(circle, rgba(${rgb},0.4), transparent)` }} />
        </div>
        <div style={{ fontSize: '0.76rem', fontWeight: 600, color: '#6b7280', letterSpacing: '0.05em', textTransform: 'uppercase' }}>Analyzing…</div>
        <div style={{ display: 'flex', gap: 5 }}>
          {PIPELINE_STAGES.map(s => (
            <div key={s.key} style={{
              padding: '3px 8px', borderRadius: 99, fontSize: '0.60rem', fontWeight: 700,
              background: partialModels.has(s.key) ? `rgba(${rgb},0.16)` : 'rgba(255,255,255,0.04)',
              color: partialModels.has(s.key) ? `rgb(${rgb})` : '#4b5563',
              border: `1px solid ${partialModels.has(s.key) ? `rgba(${rgb},0.30)` : 'transparent'}`,
              transition: 'all 0.3s ease',
              boxShadow: partialModels.has(s.key) ? `0 0 8px rgba(${rgb},0.25)` : 'none',
            }}>{s.label}</div>
          ))}
        </div>
      </div>
    );
  }

  // top-3 LSTM predictions
  const lstmTop3 = traj?.predicted_next
    ? Object.entries(traj.predicted_next).sort((a, b) => b[1] - a[1]).slice(0, 3)
    : null;

  return (
    <div style={{ height: '100%', overflowY: 'auto', background: 'var(--bg-void, #03030a)', display: 'flex', flexDirection: 'column' }}>

      {/* ── Hero ── */}
      <div key={animKey} style={{
        padding: '24px 18px 18px',
        background: `radial-gradient(ellipse at 50% -10%, rgba(${rgb},0.26) 0%, transparent 65%)`,
        borderBottom: `1px solid ${BORDER}`,
        animation: 'eiFadeIn 0.42s ease both',
        flexShrink: 0,
      }}>
        <div style={{ fontSize: '0.57rem', fontWeight: 700, color: '#374151', textTransform: 'uppercase', letterSpacing: '.13em', marginBottom: 16 }}>
          {processing ? 'Updating…' : 'Current Emotion'}
        </div>

        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
          <div>
            <div style={{
              fontSize: 'var(--text-2xl, 1.65rem)',
              fontWeight: 'var(--weight-black, 900)',
              textTransform: 'capitalize',
              lineHeight: 1.05,
              background: `linear-gradient(135deg, rgb(${rgb}) 0%, rgba(${rgb},0.55) 100%)`,
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
              letterSpacing: '-0.02em',
            }}>{dom}</div>
            {confidence != null && (
              <div style={{ fontSize: '0.68rem', color: '#374151', marginTop: 6, fontWeight: 500 }}>
                {(confidence * 100).toFixed(0)}% confidence · live
              </div>
            )}
          </div>

          <div style={{ position: 'relative', width: 52, height: 52, flexShrink: 0 }}>
            <div style={{ position: 'absolute', inset: -6, borderRadius: '50%', background: `rgba(${rgb},0.08)`, animation: 'eiPulse 2.6s ease-in-out infinite' }} />
            <div style={{
              position: 'absolute', inset: 0, borderRadius: '50%',
              background: `radial-gradient(circle at 35% 35%, rgba(${rgb},0.92), rgba(${rgb},0.32))`,
              boxShadow: `0 0 20px rgba(${rgb},0.55), 0 0 44px rgba(${rgb},0.16)`,
            }} />
          </div>
        </div>

        {shift?.significance === 'High' && (
          <div style={{ marginTop: 14, padding: '7px 10px', borderRadius: 8, background: 'rgba(239,68,68,0.09)', border: '1px solid rgba(239,68,68,0.20)', fontSize: '0.70rem', color: '#f87171', display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: '0.85rem' }}>⚡</span>
            <span>Mood shift: <strong style={{ textTransform: 'capitalize' }}>{shift.from}</strong> → <strong style={{ textTransform: 'capitalize' }}>{shift.to}</strong></span>
          </div>
        )}

        {sarcasm > 0.38 && (
          <div style={{ marginTop: 8, padding: '6px 10px', borderRadius: 8, background: 'rgba(245,158,11,0.09)', border: '1px solid rgba(245,158,11,0.20)', fontSize: '0.70rem', color: '#fbbf24', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span>Sarcasm detected</span>
            <strong>{Math.round(sarcasm * 100)}%</strong>
          </div>
        )}
      </div>

      {/* ── Valence + Arc ── */}
      <div style={{ padding: '16px 18px', borderBottom: `1px solid ${BORDER}`, flexShrink: 0 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
          <span style={SECT}>Valence</span>
          <span style={{ fontSize: '0.74rem', fontWeight: 700, color: valence > 0.15 ? '#4ade80' : valence < -0.15 ? '#f87171' : '#9ca3b0' }}>
            {valence > 0.15 ? 'Positive' : valence < -0.15 ? 'Negative' : 'Neutral'} {valence >= 0 ? '+' : ''}{valence.toFixed(2)}
          </span>
        </div>
        <div style={{ position: 'relative', height: 8, borderRadius: 4, overflow: 'hidden', background: 'rgba(255,255,255,0.05)' }}>
          <div style={{ position: 'absolute', inset: 0, left: '50%', borderRadius: '0 4px 4px 0', background: 'linear-gradient(90deg, transparent, rgba(74,222,128,0.20))' }} />
          <div style={{ position: 'absolute', inset: 0, right: '50%', borderRadius: '4px 0 0 4px', background: 'linear-gradient(90deg, rgba(248,113,113,0.20), transparent)' }} />
          <div style={{
            position: 'absolute', top: '50%', transform: 'translate(-50%, -50%)',
            left: `${valencePct}%`,
            width: 14, height: 14, borderRadius: '50%',
            background: `rgb(${rgb})`,
            boxShadow: `0 0 10px rgba(${rgb},0.9), 0 0 20px rgba(${rgb},0.4)`,
            transition: 'left 0.7s cubic-bezier(0.34,1.2,0.64,1)',
            zIndex: 2,
          }} />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
          <span style={{ fontSize: '0.54rem', color: '#374151' }}>Negative</span>
          <span style={{ fontSize: '0.54rem', color: '#374151' }}>Positive</span>
        </div>

        {valenceHistory.length >= 3 && (
          <div style={{ marginTop: 14 }}>
            <div style={{ ...SECT, marginBottom: 8 }}>
              Conversation Arc · {valenceHistory.length} messages
            </div>
            <EmotionArcChart values={valenceHistory} color={rgb} height={68} />
          </div>
        )}
      </div>

      {/* ── Emotion spectrum ── */}
      {topEmos.length > 0 && (
        <div style={{ padding: '14px 18px', borderBottom: `1px solid ${BORDER}`, flexShrink: 0 }}>
          <div style={SECT}>Emotion Spectrum</div>
          {topEmos.map(({ label, score }) => {
            const eRgb = EmotionPalette[label.toLowerCase()] || EmotionPalette.default;
            return (
              <div key={label} style={{ marginBottom: 8 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                  <span style={{ fontSize: '0.74rem', color: '#d1d5db', textTransform: 'capitalize', fontWeight: 500 }}>{label}</span>
                  <span style={{ fontSize: '0.68rem', fontWeight: 700, color: `rgb(${eRgb})` }}>{(score * 100).toFixed(0)}%</span>
                </div>
                <div style={{ height: 4, borderRadius: 2, background: 'rgba(255,255,255,0.05)' }}>
                  <div style={{
                    height: '100%', width: `${score * 100}%`,
                    background: `linear-gradient(90deg, rgba(${eRgb},0.9), rgba(${eRgb},0.40))`,
                    borderRadius: 2, boxShadow: `0 0 5px rgba(${eRgb},0.30)`,
                    transition: 'width 0.7s cubic-bezier(0.34,1.2,0.64,1)',
                  }} />
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* ── LSTM Trajectory — top 3 ── */}
      {traj?.model_available && lstmTop3?.length > 0 && (
        <div style={{ padding: '14px 18px', borderBottom: `1px solid ${BORDER}`, flexShrink: 0 }}>
          <div style={SECT}>LSTM — Next Predicted</div>
          <div style={{ padding: '12px', borderRadius: 12, background: 'rgba(139,92,246,0.07)', border: '1px solid rgba(139,92,246,0.16)' }}>
            {/* primary prediction */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
              <div style={{
                width: 36, height: 36, borderRadius: 10,
                background: `rgba(${EmotionPalette[lstmTop3[0][0].toLowerCase()] || '139,92,246'},0.18)`,
                border: `1px solid rgba(${EmotionPalette[lstmTop3[0][0].toLowerCase()] || '139,92,246'},0.30)`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '1.0rem',
              }}>→</div>
              <div>
                <div style={{ fontSize: '0.92rem', fontWeight: 800, color: '#c4b5fd', textTransform: 'capitalize' }}>
                  {lstmTop3[0][0]}
                </div>
                <div style={{ fontSize: '0.62rem', color: '#6b7280', marginTop: 1 }}>
                  {(lstmTop3[0][1] * 100).toFixed(0)}% probability
                </div>
              </div>
            </div>

            {/* top 3 bars */}
            {lstmTop3.map(([emo, score]) => {
              const eRgb = EmotionPalette[emo.toLowerCase()] || '139,92,246';
              return (
                <div key={emo} style={{ marginBottom: 7 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 3 }}>
                    <span style={{ fontSize: '0.70rem', color: '#9ca3b0', textTransform: 'capitalize', fontWeight: 500 }}>{emo}</span>
                    <span style={{ fontSize: '0.66rem', fontWeight: 700, color: `rgb(${eRgb})` }}>{(score * 100).toFixed(0)}%</span>
                  </div>
                  <div style={{ height: 4, borderRadius: 2, background: 'rgba(255,255,255,0.05)' }}>
                    <div style={{
                      height: '100%', width: `${score * 100}%`,
                      background: `linear-gradient(90deg, rgba(${eRgb},0.85), rgba(${eRgb},0.35))`,
                      borderRadius: 2,
                      transition: 'width 0.6s cubic-bezier(0.34,1.2,0.64,1)',
                    }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Context Engine ── */}
      {snap && (
        <div style={{ padding: '14px 18px', borderBottom: `1px solid ${BORDER}`, flexShrink: 0 }}>
          <div style={SECT}>Context Engine</div>
          <div style={{ padding: '12px', borderRadius: 12, background: 'rgba(6,182,212,0.05)', border: '1px solid rgba(6,182,212,0.12)' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: snap.cdm_current_state ? 8 : 0 }}>
              {[
                { label: 'Volatility',   val: `${Math.round((snap.volatility || 0) * 100)}%`,      color: (snap.volatility || 0) > 0.6 ? '#f87171' : '#d1d5db' },
                { label: 'Topic Fit',    val: `${Math.round((snap.topic_resonance || 0) * 100)}%`,  color: '#d1d5db' },
                { label: 'Episodic',     val: snap.ce_available ? 'Active' : 'Off',                  color: snap.ce_available ? '#4ade80' : '#4b5563' },
                { label: 'Abruptness',   val: snap.cdm_entry_abruptness != null ? `${Math.round(snap.cdm_entry_abruptness * 100)}%` : '—', color: (snap.cdm_entry_abruptness || 0) > 0.5 ? '#f87171' : '#d1d5db' },
              ].map(({ label, val, color }) => (
                <div key={label} style={{ padding: '6px 8px', borderRadius: 7, background: 'rgba(255,255,255,0.03)' }}>
                  <div style={{ fontSize: '0.54rem', color: '#374151', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.06em' }}>{label}</div>
                  <div style={{ fontSize: '0.74rem', fontWeight: 700, color, marginTop: 3 }}>{val}</div>
                </div>
              ))}
            </div>
            {snap.cdm_current_state && (
              <div style={{
                padding: '5px 10px', borderRadius: 99, fontSize: '0.68rem', fontWeight: 700,
                background: 'rgba(139,92,246,0.12)', color: '#a78bfa',
                border: '1px solid rgba(139,92,246,0.22)',
                display: 'inline-block',
              }}>
                CDM: {snap.cdm_current_state}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Model contributions ── */}
      {Object.keys(logicMap).length > 0 && (
        <div style={{ padding: '14px 18px', borderBottom: `1px solid ${BORDER}`, flexShrink: 0 }}>
          <div style={SECT}>Model Contributions</div>
          {Object.entries(logicMap).map(([model, v]) => (
            <div key={model} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
              <span style={{ fontSize: '0.64rem', color: '#4b5563', width: 68, flexShrink: 0 }}>{model}</span>
              <div style={{ flex: 1, height: 4, borderRadius: 2, background: 'rgba(255,255,255,0.05)', overflow: 'hidden' }}>
                <div style={{
                  height: '100%', width: `${Math.min(Math.abs(v) * 100, 100)}%`,
                  background: v >= 0 ? 'linear-gradient(90deg,#3b82f6,#8b5cf6)' : '#ef4444',
                  borderRadius: 2, transition: 'width 0.5s ease',
                }} />
              </div>
              <span style={{ fontSize: '0.66rem', fontWeight: 700, color: v >= 0 ? '#60a5fa' : '#f87171', width: 34, textAlign: 'right', flexShrink: 0 }}>
                {v >= 0 ? '+' : ''}{(v * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      )}

      {/* ── Pipeline badges ── */}
      <div style={{ padding: '12px 18px 18px', flexShrink: 0 }}>
        <div style={SECT}>ML Pipeline</div>
        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
          {PIPELINE_STAGES.map(s => {
            const active = partialModels.has(s.key) || (data && !processing);
            return (
              <div key={s.key} style={{
                padding: '2px 9px', borderRadius: 99, fontSize: '0.60rem', fontWeight: 700,
                background: active ? `rgba(${rgb},0.12)` : 'rgba(255,255,255,0.04)',
                color: active ? `rgb(${rgb})` : '#4b5563',
                border: `1px solid ${active ? `rgba(${rgb},0.26)` : 'transparent'}`,
                transition: 'all 0.35s ease',
                boxShadow: active ? `0 0 6px rgba(${rgb},0.22)` : 'none',
              }}>{s.label}</div>
            );
          })}
        </div>
      </div>

      <style>{`
        @keyframes eiPulse {
          0%, 100% { transform: scale(1); opacity: 0.6; }
          50%       { transform: scale(1.22); opacity: 0.25; }
        }
        @keyframes eiFadeIn {
          from { opacity: 0; transform: translateY(-5px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes eiSpin {
          from { transform: rotate(0deg); }
          to   { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

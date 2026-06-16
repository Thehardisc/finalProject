import React, { useState } from 'react';
import axios from 'axios';
import { EmotionPalette } from './EmotionPalette';

const API_BASE = 'http://localhost:8001';

const EMOTION_LABELS = [
  'admiration','amusement','anger','annoyance','approval','caring',
  'confusion','curiosity','desire','disappointment','disapproval',
  'disgust','embarrassment','excitement','fear','gratitude','grief',
  'joy','love','nervousness','optimism','pride','realization',
  'relief','remorse','sadness','surprise','neutral',
];

// Russell circumplex approximations
const AROUSAL = {
  anger:.88, excitement:.95, joy:.76, surprise:.86, fear:.80,
  annoyance:.65, admiration:.58, amusement:.55, confusion:.48,
  curiosity:.64, desire:.70, disgust:.62, disapproval:.50,
  embarrassment:.55, approval:.44, caring:.40, gratitude:.34,
  love:.50, optimism:.60, pride:.65, realization:.45, relief:.30,
  sadness:.24, disappointment:.33, grief:.20, remorse:.28,
  nervousness:.76, neutral:.35,
};
const DOMINANCE = {
  anger:.85, excitement:.70, joy:.65, surprise:.40, fear:.14,
  annoyance:.72, admiration:.28, amusement:.60, confusion:.24,
  curiosity:.54, desire:.55, disgust:.72, disapproval:.66,
  embarrassment:.18, approval:.60, caring:.55, gratitude:.44,
  love:.54, optimism:.65, pride:.88, realization:.60, relief:.56,
  sadness:.18, disappointment:.24, grief:.14, remorse:.24,
  nervousness:.18, neutral:.50,
};

function computeVAD(bertEmotions, valence) {
  if (!bertEmotions?.length) return { v: ((valence ?? 0) + 1) / 2, a: .5, d: .5 };
  let totalW = 0, ar = 0, dom = 0;
  bertEmotions.forEach(({ label, score }) => {
    const k = label.toLowerCase();
    ar  += (AROUSAL[k]    ?? .40) * score;
    dom += (DOMINANCE[k]  ?? .50) * score;
    totalW += score;
  });
  return {
    v: Math.max(0, Math.min(1, ((valence ?? 0) + 1) / 2)),
    a: totalW ? ar / totalW : .5,
    d: totalW ? dom / totalW : .5,
  };
}

function Gauge({ label, value, lo, hi, color }) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
        <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#d1d5db' }}>{label}</span>
        <span style={{ fontSize: '0.70rem', color: '#4b5563' }}>{pct}%</span>
      </div>
      <div style={{ position: 'relative', height: 5, borderRadius: 3, background: 'rgba(255,255,255,0.07)' }}>
        <div style={{
          position: 'absolute', top: 0, left: 0, height: '100%',
          width: `${pct}%`,
          background: `linear-gradient(90deg, rgba(${color},0.9), rgba(${color},0.5))`,
          borderRadius: 3,
          boxShadow: `0 0 6px rgba(${color},0.4)`,
          transition: 'width .55s cubic-bezier(.34,1.2,.64,1)',
        }} />
        <div style={{
          position: 'absolute', top: '50%', left: `${pct}%`,
          transform: 'translate(-50%,-50%)',
          width: 11, height: 11, borderRadius: '50%',
          background: `rgb(${color})`,
          boxShadow: `0 0 8px rgba(${color},0.8)`,
          transition: 'left .55s cubic-bezier(.34,1.2,.64,1)',
        }} />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 3 }}>
        <span style={{ fontSize: '0.60rem', color: '#374151' }}>{lo}</span>
        <span style={{ fontSize: '0.60rem', color: '#374151' }}>{hi}</span>
      </div>
    </div>
  );
}

function generateAutoInsight(data) {
  if (!data) return null;
  const parts = [];
  const sarcasm = data.sarcasm_score ?? data.llm_sarcasm_score ?? 0;
  const shift   = data.context_shift;
  const traj    = data.lstm_trajectory;
  const snap    = data.context_snapshot;
  const v       = data.final_valence ?? 0;
  const emo     = data.final_dominant_emotion;

  if (sarcasm > 0.40) parts.push(`Sarcasm likely (${Math.round(sarcasm * 100)}%) — positive surface, negative intent.`);
  if (shift?.significance === 'High') parts.push(`Mood shift: ${shift.from} → ${shift.to}.`);
  if (traj?.top_predicted && traj.top_predicted !== emo) parts.push(`Trajectory predicts ${traj.top_predicted} next.`);
  const vol = snap?.volatility ?? 0;
  if (vol > 0.65) parts.push(`High volatility (${Math.round(vol * 100)}%).`);
  else if (v > 0.4) parts.push(`Positive tone (valence ${v.toFixed(2)}).`);
  else if (v < -0.3) parts.push(`Negative tone (valence ${v.toFixed(2)}).`);
  if (snap?.cdm_current_state) parts.push(`State: ${snap.cdm_current_state}.`);

  return parts.length ? parts.join(' ') : null;
}

export default function AnalysisDrawer({ msg, onClose, onFeedbackSent, dark = false }) {
  const [selected, setSelected] = useState('');
  const [sent, setSent]         = useState(false);
  const [loading, setLoading]   = useState(false);

  const data = msg?.analysis?.data;
  if (!data) return (
    <div style={{ padding: 24, color: '#9ca3af', fontSize: '0.85rem', textAlign: 'center' }}>
      No analysis available for this message yet.
      <button onClick={onClose} style={{ display: 'block', margin: '16px auto 0', background: 'none', border: 'none', color: '#0077ff', cursor: 'pointer', fontWeight: 600 }}>← Back</button>
    </div>
  );

  const emotions    = data.bert_emotions || [];
  const topEmos     = [...emotions].sort((a, b) => b.score - a.score).slice(0, 8);
  const totalScore  = topEmos.reduce((s, e) => s + e.score, 0) || 1;
  const vad         = computeVAD(emotions, data.final_valence);
  const domRgb      = EmotionPalette[data.final_dominant_emotion?.toLowerCase()] || EmotionPalette.neutral;
  const logicMap    = data.logic_map || {};
  const sarcasmScore = data.sarcasm_score ?? data.llm_sarcasm_score ?? 0;
  const displayInsight = msg?.analysis?.ai_insight || generateAutoInsight(data);

  const submitFeedback = async () => {
    if (!selected || !data.id) return;
    setLoading(true);
    try {
      await axios.post(`${API_BASE}/message/${data.id}/feedback`, { label: selected });
      setSent(true);
      onFeedbackSent?.(data.id, selected);
    } catch {}
    finally { setLoading(false); }
  };

  const BG    = '#080810';
  const BORDER = 'rgba(255,255,255,0.07)';
  const MUTED  = '#4b5563';
  const TEXT   = '#d1d5db';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflowY: 'auto', background: BG }}>

      {/* ── Header (sticky) ── */}
      <div style={{
        padding: '14px 14px 10px', borderBottom: `1px solid ${BORDER}`,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        position: 'sticky', top: 0, background: BG, zIndex: 5,
      }}>
        <div>
          <div style={{ fontWeight: 700, fontSize: '0.82rem', color: TEXT }}>Message Analysis</div>
          <div style={{ fontSize: '0.67rem', color: MUTED, marginTop: 2 }}>
            {String(data.id).startsWith('demo') ? 'demo sample' : `msg:${String(data.id).substring(0, 8)}…`}
          </div>
        </div>
        <button onClick={onClose} style={{
          background: 'rgba(255,255,255,0.06)', border: 'none', borderRadius: '50%',
          width: 26, height: 26, cursor: 'pointer',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: MUTED, fontSize: '0.85rem', flexShrink: 0,
        }}>✕</button>
      </div>

      <div style={{ padding: '14px' }}>

        {/* ── Dominant + confidence ── */}
        <div style={{
          background: `radial-gradient(ellipse at top, rgba(${domRgb},.16) 0%, rgba(${domRgb},.04) 80%)`,
          border: `1px solid rgba(${domRgb},.22)`,
          borderRadius: 12, padding: '13px', marginBottom: 16,
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 4 }}>
            <div>
              <div style={{ fontSize: '0.63rem', color: MUTED, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '.07em', marginBottom: 4 }}>Dominant</div>
              <div style={{ fontSize: '1.08rem', fontWeight: 800, textTransform: 'capitalize',
                background: `linear-gradient(135deg, rgb(${domRgb}), rgba(${domRgb},0.6))`,
                WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
              }}>
                {data.final_dominant_emotion || 'Neutral'}
              </div>
            </div>
            {data.meta_confidence != null && (
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: '0.63rem', color: MUTED, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '.07em', marginBottom: 4 }}>Confidence</div>
                <div style={{ fontSize: '1.08rem', fontWeight: 800, color: `rgb(${domRgb})` }}>
                  {(data.meta_confidence * 100).toFixed(0)}%
                </div>
              </div>
            )}
          </div>

          {data.context_shift && (
            <div style={{
              marginTop: 8, padding: '5px 9px', background: 'rgba(220,38,38,.08)',
              borderRadius: 6, fontSize: '0.71rem', color: '#dc2626',
              display: 'flex', alignItems: 'center', gap: 5,
            }}>
              ⚡ Mood shift: <strong>{data.context_shift.from}</strong> → <strong>{data.context_shift.to}</strong> · {data.context_shift.significance}
            </div>
          )}

          {displayInsight && (
            <div style={{
              marginTop: 8, padding: '8px 10px',
              background: dark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,.03)',
              borderRadius: 6, fontSize: '0.72rem',
              color: dark ? '#c0c0c0' : '#374151', lineHeight: 1.55,
              fontStyle: 'italic',
            }}>
              "{displayInsight}"
            </div>
          )}
        </div>

        {/* ── Sarcasm ── */}
        {sarcasmScore > 0.05 && (
          <div style={{ marginBottom: 16 }}>
            <div style={sectionTitle}>Sarcasm Detector</div>
            <div style={{
              background: sarcasmScore > 0.4 ? 'rgba(245,158,11,0.08)' : (dark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.03)'),
              border: sarcasmScore > 0.4 ? '1px solid rgba(245,158,11,0.25)' : (dark ? '1px solid rgba(255,255,255,0.07)' : '1px solid rgba(0,0,0,0.07)'),
              borderRadius: 10, padding: '10px 12px',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <span style={{ fontSize: '0.78rem', fontWeight: 600, color: sarcasmScore > 0.4 ? '#d97706' : (dark ? '#9ca3af' : '#6b7280') }}>
                  {sarcasmScore > 0.6 ? 'High' : sarcasmScore > 0.4 ? 'Moderate' : 'Low'} likelihood
                </span>
                <span style={{ fontSize: '0.82rem', fontWeight: 800, color: sarcasmScore > 0.4 ? '#d97706' : '#9ca3af' }}>
                  {Math.round(sarcasmScore * 100)}%
                </span>
              </div>
              <div style={{ height: 5, borderRadius: 3, background: dark ? 'rgba(255,255,255,.07)' : 'rgba(0,0,0,.08)' }}>
                <div style={{
                  height: '100%', width: `${sarcasmScore * 100}%`,
                  background: sarcasmScore > 0.4 ? 'linear-gradient(90deg,#f59e0b,#d97706)' : '#9ca3af',
                  borderRadius: 3, transition: 'width .6s cubic-bezier(.34,1.2,.64,1)',
                }} />
              </div>
            </div>
          </div>
        )}

        {/* ── Emotion breakdown ── */}
        <div style={{ marginBottom: 16 }}>
          <div style={sectionTitle}>Emotion Weights</div>
          {topEmos.map(({ label, score }) => {
            const rgb = EmotionPalette[label.toLowerCase()] || EmotionPalette.default;
            const pct = (score / totalScore) * 100;
            return (
              <div key={label} style={{ marginBottom: 9 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ fontSize: '0.76rem', fontWeight: 500, color: TEXT, textTransform: 'capitalize' }}>{label}</span>
                  <span style={{ fontSize: '0.72rem', fontWeight: 700, color: `rgb(${rgb})` }}>{pct.toFixed(0)}%</span>
                </div>
                <div style={{ height: 4, borderRadius: 2, background: 'rgba(255,255,255,0.06)' }}>
                  <div style={{
                    height: '100%', width: `${pct}%`,
                    background: `linear-gradient(90deg, rgba(${rgb},0.9), rgba(${rgb},0.45))`,
                    borderRadius: 2,
                    boxShadow: `0 0 5px rgba(${rgb},0.3)`,
                    transition: 'width .6s cubic-bezier(.34,1.2,.64,1)',
                  }} />
                </div>
              </div>
            );
          })}
        </div>

        {/* ── VAD Gauges ── */}
        <div style={{ marginBottom: 16 }}>
          <div style={sectionTitle}>VAD Dimensions</div>
          <Gauge label="Valence"   value={vad.v} lo="Negative"   hi="Positive"  color="52,199,89"  />
          <Gauge label="Arousal"   value={vad.a} lo="Calm"       hi="Activated" color="255,126,0"  />
          <Gauge label="Dominance" value={vad.d} lo="Submissive" hi="Dominant"  color="162,67,220" />
        </div>

        {/* ── Model contributions ── */}
        {Object.keys(logicMap).length > 0 && (
          <div style={{ marginBottom: 16 }}>
            <div style={sectionTitle}>Model Contributions</div>
            {Object.entries(logicMap).map(([model, v]) => (
              <div key={model} style={{ marginBottom: 8 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                  <span style={{ fontSize: '0.75rem', color: '#6b7280' }}>{model}</span>
                  <span style={{ fontSize: '0.72rem', fontWeight: 600, color: v >= 0 ? '#0077ff' : '#ef4444' }}>
                    {v >= 0 ? '+' : ''}{(v * 100).toFixed(0)}%
                  </span>
                </div>
                <div style={{ height: 4, borderRadius: 2, background: 'rgba(0,0,0,.07)', overflow: 'hidden' }}>
                  <div style={{
                    height: '100%',
                    width: `${Math.min(Math.abs(v) * 100, 100)}%`,
                    background: v >= 0 ? 'linear-gradient(90deg,#0077ff,#7000ff)' : '#ef4444',
                    borderRadius: 2, transition: 'width .5s ease',
                  }} />
                </div>
              </div>
            ))}
          </div>
        )}

        {/* ── Context Engine snapshot ── */}
        {data.context_snapshot && (
          <div style={{ marginBottom: 16 }}>
            <div style={sectionTitle}>Context Engine</div>
            <div style={{
              background: 'rgba(0,119,255,0.05)',
              border: '1px solid rgba(0,119,255,0.15)',
              borderRadius: 12, padding: '12px 14px',
            }}>
              {/* Valence arc */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
                <div style={{
                  padding: '3px 9px', borderRadius: 99, fontSize: '0.71rem', fontWeight: 700,
                  background: dark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)', color: dark ? '#c0c0c0' : '#374151', textTransform: 'capitalize',
                }}>
                  {data.context_snapshot.prev_emotion || 'none'}
                </div>
                <div style={{ fontSize: '0.75rem', color: '#9ca3af' }}>→</div>
                <div style={{
                  padding: '3px 9px', borderRadius: 99, fontSize: '0.71rem', fontWeight: 700,
                  background: `rgba(${domRgb},0.15)`, color: `rgb(${domRgb})`, textTransform: 'capitalize',
                }}>
                  {data.final_dominant_emotion}
                </div>
              </div>

              {/* Stats grid */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                {[
                  { label: 'Valence', value: data.context_snapshot.cur_valence?.toFixed(3), color: (data.context_snapshot.cur_valence ?? 0) >= 0 ? '#16a34a' : '#dc2626' },
                  { label: 'Topic Resonance', value: `${((data.context_snapshot.topic_resonance || 0) * 100).toFixed(0)}%`, color: '#7c3aed' },
                  { label: 'Volatility', value: `${((data.context_snapshot.volatility || 0) * 100).toFixed(0)}%`, color: '#d97706' },
                  { label: 'Episodic Memory', value: data.context_snapshot.ce_available ? 'active' : 'off', color: data.context_snapshot.ce_available ? '#16a34a' : '#9ca3af' },
                ].map(({ label, value, color }) => (
                  <div key={label} style={{ padding: '7px 10px', background: dark ? 'rgba(255,255,255,0.04)' : 'rgba(255,255,255,0.7)', borderRadius: 8 }}>
                    <div style={{ fontSize: '0.63rem', color: '#9ca3af', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '.06em', marginBottom: 3 }}>{label}</div>
                    <div style={{ fontSize: '0.82rem', fontWeight: 700, color }}>{value ?? '—'}</div>
                  </div>
                ))}
              </div>

              {/* CDM state + abruptness */}
              {data.context_snapshot.cdm_current_state && (
                <div style={{ marginTop: 8, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  <div style={{ padding: '3px 9px', borderRadius: 99, background: dark ? 'rgba(112,0,255,0.15)' : 'rgba(112,0,255,0.08)', border: '1px solid rgba(112,0,255,0.20)', fontSize: '0.68rem', fontWeight: 700, color: '#7000ff' }}>
                    CDM: {data.context_snapshot.cdm_current_state}
                  </div>
                  {data.context_snapshot.cdm_entry_abruptness > 0.5 && (
                    <div style={{ padding: '3px 9px', borderRadius: 99, background: 'rgba(220,38,38,0.08)', border: '1px solid rgba(220,38,38,0.18)', fontSize: '0.68rem', fontWeight: 700, color: '#dc2626' }}>
                      Abrupt entry {Math.round(data.context_snapshot.cdm_entry_abruptness * 100)}%
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── LSTM Trajectory / State Machine ── */}
        {data.lstm_trajectory?.model_available && (
          <div style={{ marginBottom: 16 }}>
            <div style={sectionTitle}>Trajectory — Next Predicted</div>
            <div style={{
              background: 'rgba(112,0,255,0.05)',
              border: '1px solid rgba(112,0,255,0.18)',
              borderRadius: 12, padding: '12px 14px',
            }}>
              <div style={{ marginBottom: 10, display: 'flex', alignItems: 'center', gap: 8 }}>
                <div style={{
                  padding: '4px 12px', borderRadius: 99, fontSize: '0.78rem', fontWeight: 800,
                  background: `rgba(${EmotionPalette[data.lstm_trajectory.top_predicted?.toLowerCase()] || '112,0,255'},0.15)`,
                  color: `rgb(${EmotionPalette[data.lstm_trajectory.top_predicted?.toLowerCase()] || '112,0,255'})`,
                  textTransform: 'capitalize',
                }}>
                  → {data.lstm_trajectory.top_predicted}
                </div>
                <span style={{ fontSize: '0.70rem', color: '#9ca3af' }}>most likely next emotion</span>
              </div>

              {Object.entries(data.lstm_trajectory.predicted_next || {}).map(([emo, score]) => {
                const rgb = EmotionPalette[emo.toLowerCase()] || '112,0,255';
                return (
                  <div key={emo} style={{ marginBottom: 7 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                      <span style={{ fontSize: '0.74rem', color: '#374151', textTransform: 'capitalize' }}>{emo}</span>
                      <span style={{ fontSize: '0.70rem', fontWeight: 700, color: `rgb(${rgb})` }}>{(score * 100).toFixed(0)}%</span>
                    </div>
                    <div style={{ height: 5, borderRadius: 3, background: 'rgba(0,0,0,0.07)' }}>
                      <div style={{
                        height: '100%', width: `${score * 100}%`,
                        background: `linear-gradient(90deg, rgba(${rgb},0.8), rgba(${rgb},0.4))`,
                        borderRadius: 3, transition: 'width 0.6s cubic-bezier(0.34,1.2,0.64,1)',
                      }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* ── Human-in-the-loop ── */}
        <div style={{
          background: 'rgba(255,255,255,0.03)', borderRadius: 12, padding: '13px',
          border: '1px solid rgba(255,255,255,0.07)',
        }}>
          <div style={sectionTitle}>Correct the Model</div>
          {sent ? (
            <div style={{
              padding: '10px', background: 'rgba(52,199,89,.08)',
              border: '1px solid rgba(52,199,89,.25)', borderRadius: 8,
              fontSize: '0.82rem', color: '#15803d', textAlign: 'center',
            }}>
              ✓ Correction submitted — model will improve
            </div>
          ) : (
            <>
              <select
                value={selected}
                onChange={e => setSelected(e.target.value)}
                style={{
                  width: '100%', padding: '9px 12px',
                  background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.10)',
                  borderRadius: 8, fontSize: '0.84rem', color: '#d1d5db',
                  outline: 'none', marginBottom: 9, cursor: 'pointer',
                  appearance: 'none',
                }}
              >
                <option value="">Select correct emotion…</option>
                {EMOTION_LABELS.map(e => (
                  <option key={e} value={e}>{e[0].toUpperCase() + e.slice(1)}</option>
                ))}
              </select>
              <button
                onClick={submitFeedback}
                disabled={!selected || loading}
                style={{
                  width: '100%', padding: '10px',
                  background: selected ? 'linear-gradient(135deg,#0077ff,#7000ff)' : '#e5e7eb',
                  border: 'none', borderRadius: 8,
                  color: selected ? '#fff' : '#9ca3af',
                  fontWeight: 700, fontSize: '0.84rem',
                  cursor: selected ? 'pointer' : 'default',
                  transition: 'background .2s',
                }}
              >
                {loading ? 'Submitting…' : 'Submit Correction'}
              </button>
            </>
          )}
        </div>

      </div>
    </div>
  );
}

const sectionTitle = {
  fontSize: '0.60rem', fontWeight: 700, color: '#4b5563',
  textTransform: 'uppercase', letterSpacing: '.10em', marginBottom: 10,
};

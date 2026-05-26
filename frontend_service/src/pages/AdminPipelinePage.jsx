/**
 * AdminPipelinePage — Step-by-step ML pipeline inspector for admins.
 *
 * Shows every stage of the analysis pipeline for any message:
 *   Input → VADER → BERT → GoEmotions → Context Engine → Meta-Learner → Context Impact
 */
import React, { useState, useEffect, useCallback } from 'react';
import { adminAPI } from '../api/client';
import { EmotionPalette } from '../components/EmotionPalette';

// ── Helpers ───────────────────────────────────────────────────────────────────

const rgb = (emo) => EmotionPalette[emo?.toLowerCase()] || EmotionPalette.neutral || '150,150,150';

const CDM_STATES = [
  { id: 0, short: 'NEUTRAL',   color: '107,114,128' },
  { id: 1, short: 'ASCENDING', color: '34,197,94'   },
  { id: 2, short: 'CONFLICT',  color: '239,68,68'   },
  { id: 3, short: 'ANCHOR',    color: '59,130,246'  },
  { id: 4, short: 'CONVERGE',  color: '20,184,166'  },
  { id: 5, short: 'PEAK',      color: '168,85,247'  },
  { id: 6, short: 'DIVERG',    color: '249,115,22'  },
];

const fmt = (ts) => ts ? new Date(ts * 1000).toLocaleString() : '—';
const pct = (v)  => v != null ? `${(v * 100).toFixed(1)}%` : '—';

const STAGE_ORDER = ['vader', 'bert', 'goemotions'];

const STAGE_META = {
  vader:      { label: 'VADER',      icon: '📊', color: '255,180,0',  desc: 'Lexicon-based sentiment' },
  bert:       { label: 'BERT',       icon: '🤖', color: '0,119,255',  desc: 'Ekman 7-emotion classifier' },
  goemotions: { label: 'GoEmotions', icon: '🧠', color: '162,67,220', desc: '28-class emotion model' },
};

const MODEL_LABEL_MAP = {
  VADER:      'vader',
  BERT:       'bert',
  GoEmotions: 'goemotions',
  Context:    'context',
};

// ── Sub-components ────────────────────────────────────────────────────────────

function Bar({ label, value, color, maxVal = 1, highlight = false }) {
  const w = Math.min((value / (maxVal || 1)) * 100, 100);
  return (
    <div style={{ marginBottom: 7 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
        <span style={{
          fontSize: '0.74rem', color: highlight ? `rgb(${color})` : 'rgba(255,255,255,0.7)',
          fontWeight: highlight ? 700 : 400, textTransform: 'capitalize',
        }}>{label}</span>
        <span style={{ fontSize: '0.72rem', color: `rgb(${color})`, fontWeight: 600 }}>
          {(value * 100).toFixed(1)}%
        </span>
      </div>
      <div style={{ height: 5, borderRadius: 3, background: 'rgba(255,255,255,0.07)' }}>
        <div style={{
          height: '100%', width: `${w}%`, borderRadius: 3,
          background: `linear-gradient(90deg, rgb(${color}), rgba(${color},0.5))`,
          transition: 'width 0.6s cubic-bezier(.34,1.2,.64,1)',
        }} />
      </div>
    </div>
  );
}

function StageCard({ stageKey, data }) {
  const meta    = STAGE_META[stageKey];
  const entries = Object.entries(data).sort((a, b) => b[1] - a[1]);
  const top     = entries.slice(0, 8);
  const maxVal  = top[0]?.[1] || 1;
  const isEmpty = entries.length === 0;

  return (
    <div style={S.stageCard}>
      <div style={S.stageHeader}>
        <span style={{ fontSize: '1.1rem' }}>{meta.icon}</span>
        <div>
          <div style={{ ...S.stageTitle, color: `rgb(${meta.color})` }}>{meta.label}</div>
          <div style={S.stageDesc}>{meta.desc}</div>
        </div>
        <div style={{
          marginLeft: 'auto', fontSize: '0.68rem', padding: '2px 8px',
          background: `rgba(${meta.color},0.12)`, color: `rgb(${meta.color})`,
          border: `1px solid rgba(${meta.color},0.25)`, borderRadius: 20,
        }}>
          {entries.length} scores
        </div>
      </div>

      {isEmpty ? (
        <div style={S.emptyMsg}>No data for this stage</div>
      ) : (
        <div style={{ marginTop: 12 }}>
          {top.map(([label, val]) => (
            <Bar
              key={label} label={label} value={val}
              color={rgb(label)} maxVal={maxVal}
              highlight={label === top[0][0]}
            />
          ))}
          {entries.length > 8 && (
            <div style={{ fontSize: '0.68rem', color: 'rgba(255,255,255,0.3)', marginTop: 6 }}>
              +{entries.length - 8} more emotions…
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function VaderCard({ data }) {
  if (!data || Object.keys(data).length === 0)
    return <StageCard stageKey="vader" data={{}} />;

  const compound  = data.vader_compound ?? data.compound ?? 0;
  const neg       = data.vader_neg ?? data.neg ?? 0;
  const neu       = data.vader_neu ?? data.neu ?? 0;
  const pos       = data.vader_pos ?? data.pos ?? 0;
  const sentiment = compound >= 0.05 ? 'Positive' : compound <= -0.05 ? 'Negative' : 'Neutral';
  const sentColor = compound >= 0.05 ? '52,199,89' : compound <= -0.05 ? '255,82,82' : '150,150,150';

  return (
    <div style={S.stageCard}>
      <div style={S.stageHeader}>
        <span style={{ fontSize: '1.1rem' }}>📊</span>
        <div>
          <div style={{ ...S.stageTitle, color: 'rgb(255,180,0)' }}>VADER</div>
          <div style={S.stageDesc}>Lexicon-based sentiment</div>
        </div>
        <div style={{
          marginLeft: 'auto', padding: '2px 10px', borderRadius: 20, fontSize: '0.72rem',
          background: `rgba(${sentColor},0.12)`, color: `rgb(${sentColor})`,
          border: `1px solid rgba(${sentColor},0.25)`, fontWeight: 700,
        }}>
          {sentiment}
        </div>
      </div>
      <div style={{ marginTop: 14 }}>
        <Bar label="Positive"  value={pos}            color="52,199,89"    />
        <Bar label="Neutral"   value={neu}            color="150,150,150"  />
        <Bar label="Negative"  value={neg}            color="255,82,82"    />
        <div style={{ marginTop: 14, padding: '10px 14px', background: 'rgba(255,255,255,0.04)', borderRadius: 10 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.75rem', color: 'rgba(255,255,255,0.5)' }}>Compound Score</span>
            <span style={{ fontSize: '1.1rem', fontWeight: 800, color: `rgb(${sentColor})` }}>
              {compound.toFixed(4)}
            </span>
          </div>
          <div style={{ marginTop: 8, height: 6, borderRadius: 3, background: 'rgba(255,255,255,0.07)', position: 'relative' }}>
            <div style={{
              position: 'absolute', height: '100%', borderRadius: 3,
              left: compound >= 0 ? '50%' : `${(compound + 1) / 2 * 100}%`,
              width: `${Math.abs(compound) * 50}%`,
              background: `rgb(${sentColor})`,
            }} />
            <div style={{ position: 'absolute', left: '50%', top: -2, width: 2, height: 10, background: 'rgba(255,255,255,0.3)' }} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
            <span style={{ fontSize: '0.6rem', color: 'rgba(255,255,255,0.3)' }}>-1.0</span>
            <span style={{ fontSize: '0.6rem', color: 'rgba(255,255,255,0.3)' }}>0</span>
            <span style={{ fontSize: '0.6rem', color: 'rgba(255,255,255,0.3)' }}>+1.0</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function LogicMapCard({ logicMap, dominant }) {
  if (!logicMap || Object.keys(logicMap).length === 0) return null;
  const entries  = Object.entries(logicMap).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const maxAbs   = Math.max(...entries.map(([, v]) => Math.abs(v)), 0.01);

  const modelColors = {
    VADER:      '255,180,0',
    BERT:       '0,119,255',
    GoEmotions: '162,67,220',
    Context:    '20,184,166',
  };

  return (
    <div style={S.stageCard}>
      <div style={S.stageHeader}>
        <span style={{ fontSize: '1.1rem' }}>⚖️</span>
        <div>
          <div style={{ ...S.stageTitle, color: 'rgb(255,255,255)' }}>Model Contributions</div>
          <div style={S.stageDesc}>Impact of each model on final prediction</div>
        </div>
      </div>
      <div style={{ marginTop: 14 }}>
        {entries.map(([model, val]) => {
          const c    = modelColors[model] || '150,150,150';
          const w    = (Math.abs(val) / maxAbs) * 100;
          const sign = val >= 0 ? '+' : '';
          return (
            <div key={model} style={{ marginBottom: 10 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                <span style={{ fontSize: '0.75rem', color: 'rgba(255,255,255,0.7)' }}>{model}</span>
                <span style={{
                  fontSize: '0.72rem', fontWeight: 700,
                  color: val >= 0 ? `rgb(${c})` : '#ff5252',
                }}>
                  {sign}{(val * 100).toFixed(0)}%
                </span>
              </div>
              <div style={{ height: 5, borderRadius: 3, background: 'rgba(255,255,255,0.07)' }}>
                <div style={{
                  height: '100%', width: `${w}%`, borderRadius: 3,
                  background: val >= 0 ? `linear-gradient(90deg,rgb(${c}),rgba(${c},0.4))` : '#ff5252',
                  transition: 'width 0.5s ease',
                }} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ValenceBar({ value }) {
  const clamped = Math.max(-1, Math.min(1, value ?? 0));
  const isPos   = clamped >= 0;
  const pct     = Math.abs(clamped) * 50;
  const color   = isPos ? '0,210,120' : '255,82,82';
  return (
    <div style={{ position: 'relative', height: 6, borderRadius: 3, background: 'rgba(255,255,255,0.07)', margin: '4px 0 2px' }}>
      <div style={{ position: 'absolute', left: '50%', top: 0, width: 1, height: '100%', background: 'rgba(255,255,255,0.2)' }} />
      <div style={{
        position: 'absolute', top: 0, height: '100%', borderRadius: 3,
        width: `${pct}%`, background: `rgb(${color})`,
        left: isPos ? '50%' : `${50 - pct}%`,
      }} />
    </div>
  );
}

function ContextCard({ decision, context }) {
  const { reasoning, raw_emotions, context_snapshot: cs } = context;
  const { sarcasm_score, conflict, logic_map } = decision;

  const contextPct   = logic_map?.Context ?? null;
  const contextShift = raw_emotions?.context_shift
    ? (typeof raw_emotions.context_shift === 'string'
        ? JSON.parse(raw_emotions.context_shift) : raw_emotions.context_shift)
    : null;

  const hasSarcasm  = sarcasm_score != null && sarcasm_score > 0.15;
  const hasConflict = !!conflict;
  const hasShift    = !!contextShift;
  const hasReason   = reasoning && Object.keys(reasoning).length > 0;
  const hasSnapshot = !!cs;

  const row = (label, val, color = 'rgba(255,255,255,0.55)') => (
    <div style={S.contextRow}>
      <span style={S.ctxLabel}>{label}</span>
      <span style={{ ...S.ctxValue, color }}>{val}</span>
    </div>
  );

  return (
    <div style={{ ...S.stageCard, borderColor: 'rgba(255,100,150,0.3)', background: 'rgba(255,100,150,0.04)' }}>
      <div style={S.stageHeader}>
        <span style={{ fontSize: '1.1rem' }}>🔗</span>
        <div>
          <div style={{ ...S.stageTitle, color: 'rgb(255,100,150)' }}>Context Engine</div>
          <div style={S.stageDesc}>Episodic memory + conversation history influence</div>
        </div>
        {contextPct != null && (
          <div style={{
            marginLeft: 'auto', padding: '3px 10px', borderRadius: 20, fontSize: '0.78rem',
            background: 'rgba(255,100,150,0.12)', color: 'rgb(255,100,150)',
            border: '1px solid rgba(255,100,150,0.25)', fontWeight: 700,
          }}>
            {(contextPct * 100).toFixed(0)}% weight
          </div>
        )}
      </div>

      <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 10 }}>

        {/* Snapshot grid */}
        {hasSnapshot && (
          <div style={{ padding: '10px 12px', background: 'rgba(255,255,255,0.04)', borderRadius: 8 }}>
            <div style={{ fontSize: '0.68rem', color: 'rgba(255,255,255,0.35)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.07em' }}>
              Conversation State Injected
            </div>
            {row('Previous emotion', cs.prev_emotion || 'neutral',
              `rgb(${rgb(cs.prev_emotion || 'neutral')})`)}
            <div style={S.contextRow}>
              <span style={S.ctxLabel}>EMA valence</span>
              <div style={{ flex: 1, margin: '0 12px' }}><ValenceBar value={cs.cur_valence} /></div>
              <span style={{ ...S.ctxValue, color: (cs.cur_valence ?? 0) >= 0 ? 'rgb(0,210,120)' : 'rgb(255,82,82)' }}>
                {(cs.cur_valence ?? 0) >= 0 ? '+' : ''}{((cs.cur_valence ?? 0) * 100).toFixed(1)}%
              </span>
            </div>

            {cs.ce_available && (
              <>
                <div style={{ height: 1, background: 'rgba(255,255,255,0.07)', margin: '8px 0' }} />
                <div style={{ fontSize: '0.68rem', color: 'rgba(255,255,255,0.35)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.07em' }}>
                  Episodic Memory (Qdrant)
                </div>
                <div style={S.contextRow}>
                  <span style={S.ctxLabel}>Historical valence</span>
                  <div style={{ flex: 1, margin: '0 12px' }}><ValenceBar value={cs.historical_valence} /></div>
                  <span style={{ ...S.ctxValue, color: cs.historical_valence >= 0 ? 'rgb(0,210,120)' : 'rgb(255,82,82)' }}>
                    {cs.historical_valence >= 0 ? '+' : ''}{(cs.historical_valence * 100).toFixed(1)}%
                  </span>
                </div>
                {row('Topic resonance',
                  cs.topic_resonance > 0
                    ? `${(cs.topic_resonance * 100).toFixed(0)}% similarity to past`
                    : 'No similar history found',
                  cs.topic_resonance > 0.3 ? 'rgb(255,200,80)' : 'rgba(255,255,255,0.4)')}
                {row('Volatility',
                  cs.volatility > 0.01
                    ? `${(cs.volatility * 100).toFixed(1)}% (${cs.volatility > 0.1 ? 'High' : cs.volatility > 0.03 ? 'Moderate' : 'Low'})`
                    : 'Stable',
                  cs.volatility > 0.1 ? 'rgb(255,150,50)' : 'rgba(255,255,255,0.45)')}
              </>
            )}

            {/* CDM State Machine */}
            {cs.cdm_available && cs.cdm_state_probs && (
              <>
                <div style={{ height: 1, background: 'rgba(255,255,255,0.07)', margin: '8px 0' }} />
                <div style={{ fontSize: '0.68rem', color: 'rgba(255,255,255,0.35)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.07em' }}>
                  CDM State Machine
                </div>
                {CDM_STATES.map(({ id, short, color }) => {
                  const prob = cs.cdm_state_probs[id] ?? 0;
                  const isCurrent = cs.cdm_current_state === id;
                  return (
                    <div key={id} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                      <span style={{
                        width: 58, fontSize: '0.62rem', fontWeight: isCurrent ? 700 : 400,
                        color: isCurrent ? `rgb(${color})` : 'rgba(255,255,255,0.35)',
                        flexShrink: 0,
                      }}>
                        {isCurrent ? '▶ ' : ''}{short}
                      </span>
                      <div style={{ flex: 1, height: 4, borderRadius: 2, background: 'rgba(255,255,255,0.07)' }}>
                        <div style={{
                          height: '100%', borderRadius: 2,
                          width: `${Math.min(prob * 100, 100)}%`,
                          background: isCurrent ? `rgb(${color})` : `rgba(${color},0.4)`,
                          transition: 'width .3s ease',
                        }} />
                      </div>
                      <span style={{ fontSize: '0.62rem', color: isCurrent ? `rgb(${color})` : 'rgba(255,255,255,0.3)', width: 30, textAlign: 'right' }}>
                        {(prob * 100).toFixed(0)}%
                      </span>
                    </div>
                  );
                })}
                {cs.cdm_residency > 0 && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4 }}>
                    <span style={{ width: 58, fontSize: '0.62rem', color: 'rgba(255,255,255,0.35)', flexShrink: 0 }}>residency</span>
                    <div style={{ flex: 1, height: 3, borderRadius: 2, background: 'rgba(255,255,255,0.07)' }}>
                      <div style={{
                        height: '100%', borderRadius: 2,
                        width: `${Math.min(cs.cdm_residency * 100, 100)}%`,
                        background: `rgba(${CDM_STATES[cs.cdm_current_state]?.color ?? '150,150,150'},0.6)`,
                      }} />
                    </div>
                    <span style={{ fontSize: '0.62rem', color: 'rgba(255,255,255,0.3)', width: 30, textAlign: 'right' }}>
                      {(cs.cdm_residency * 100).toFixed(0)}%
                    </span>
                  </div>
                )}
              </>
            )}

            {!cs.ce_available && (
              <div style={{ fontSize: '0.72rem', color: 'rgba(255,255,255,0.3)', fontStyle: 'italic', marginTop: 4 }}>
                Episodic memory arrived after aggregation window — base EMA context used.
              </div>
            )}
          </div>
        )}

        {/* Context contribution bar */}
        {contextPct != null && (
          <div style={S.contextRow}>
            <span style={S.ctxLabel}>Context block weight</span>
            <div style={{ flex: 1, height: 6, borderRadius: 3, background: 'rgba(255,255,255,0.07)', margin: '0 12px' }}>
              <div style={{
                height: '100%', width: `${Math.min(contextPct * 100, 100)}%`,
                borderRadius: 3, background: 'linear-gradient(90deg, rgb(255,100,150), rgba(255,100,150,0.4))',
              }} />
            </div>
            <span style={{ ...S.ctxValue, color: 'rgb(255,100,150)' }}>{(contextPct * 100).toFixed(0)}%</span>
          </div>
        )}

        {/* Mood shift */}
        {hasShift && (
          <div style={S.contextChip}>
            <span style={{ fontSize: '0.8rem' }}>⚡</span>
            <span>Mood shift: </span>
            <strong style={{ color: `rgb(${rgb(contextShift.from)})` }}>{contextShift.from}</strong>
            <span style={{ color: 'rgba(255,255,255,0.4)', margin: '0 4px' }}>→</span>
            <strong style={{ color: `rgb(${rgb(contextShift.to)})` }}>{contextShift.to}</strong>
            <span style={{ marginLeft: 6, fontSize: '0.68rem', padding: '1px 7px', background: 'rgba(255,255,255,0.08)', borderRadius: 10 }}>
              {contextShift.significance}
            </span>
          </div>
        )}

        {/* Sarcasm */}
        {hasSarcasm && (
          <div style={{ ...S.contextChip, background: 'rgba(255,200,0,0.08)', borderColor: 'rgba(255,200,0,0.2)' }}>
            <span style={{ fontSize: '0.8rem' }}>😏</span>
            <span style={{ color: 'rgba(255,255,255,0.7)' }}>Sarcasm / irony — </span>
            <strong style={{ color: 'rgb(255,200,0)' }}>score {(sarcasm_score * 100).toFixed(0)}%</strong>
          </div>
        )}

        {/* Conflict */}
        {hasConflict && (
          <div style={{ ...S.contextChip, background: 'rgba(255,82,82,0.08)', borderColor: 'rgba(255,82,82,0.2)' }}>
            <span style={{ fontSize: '0.8rem' }}>⚠️</span>
            <span style={{ color: 'rgba(255,255,255,0.7)' }}>Emotional conflict: </span>
            <strong style={{ color: 'rgb(255,82,82)' }}>{conflict}</strong>
          </div>
        )}

        {/* Reasoning */}
        {hasReason && (
          <div style={{ padding: '10px 12px', background: 'rgba(255,255,255,0.04)', borderRadius: 8 }}>
            <div style={{ fontSize: '0.68rem', color: 'rgba(255,255,255,0.4)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.07em' }}>
              LLM Reasoning
            </div>
            {reasoning.type && (
              <div style={{ fontSize: '0.78rem', color: 'rgba(255,255,255,0.8)', marginBottom: 4 }}>
                <strong style={{ color: 'rgb(255,100,150)' }}>{reasoning.type}</strong>
                {reasoning.details && <> — {reasoning.details}</>}
              </div>
            )}
            {reasoning.action && (
              <div style={{ fontSize: '0.72rem', color: 'rgba(255,255,255,0.45)', fontStyle: 'italic' }}>
                {reasoning.action}
              </div>
            )}
          </div>
        )}

        {!hasSnapshot && !hasShift && !hasSarcasm && !hasConflict && !hasReason && contextPct == null && (
          <div style={S.emptyMsg}>No context influence detected for this message.</div>
        )}
      </div>
    </div>
  );
}

function DecisionCard({ decision }) {
  const { dominant, confidence, decision_mode, sarcasm_score } = decision;
  const domRgb = rgb(dominant);

  return (
    <div style={{ ...S.stageCard, borderColor: `rgba(${domRgb},0.35)`, background: `rgba(${domRgb},0.05)` }}>
      <div style={S.stageHeader}>
        <span style={{ fontSize: '1.1rem' }}>🎯</span>
        <div>
          <div style={{ ...S.stageTitle, color: `rgb(${domRgb})` }}>Meta-Learner Decision</div>
          <div style={S.stageDesc}>Final prediction from 67-dim feature vector</div>
        </div>
        <div style={{
          marginLeft: 'auto', fontSize: '0.68rem', padding: '2px 8px',
          background: 'rgba(255,255,255,0.06)', color: 'rgba(255,255,255,0.5)',
          border: '1px solid rgba(255,255,255,0.1)', borderRadius: 20,
        }}>
          {decision_mode}
        </div>
      </div>

      <div style={{ display: 'flex', gap: 16, marginTop: 16, alignItems: 'center' }}>
        {/* Confidence ring */}
        <div style={{ position: 'relative', width: 80, height: 80, flexShrink: 0 }}>
          <svg width="80" height="80" style={{ transform: 'rotate(-90deg)' }}>
            <circle cx="40" cy="40" r="32" fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="7" />
            <circle
              cx="40" cy="40" r="32" fill="none"
              stroke={`rgb(${domRgb})`} strokeWidth="7"
              strokeDasharray={`${2 * Math.PI * 32}`}
              strokeDashoffset={`${2 * Math.PI * 32 * (1 - (confidence ?? 0))}`}
              strokeLinecap="round"
              style={{ transition: 'stroke-dashoffset 1s ease' }}
            />
          </svg>
          <div style={{
            position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center',
          }}>
            <span style={{ fontSize: '0.9rem', fontWeight: 800, color: `rgb(${domRgb})` }}>
              {confidence != null ? `${(confidence * 100).toFixed(0)}%` : '—'}
            </span>
          </div>
        </div>

        <div style={{ flex: 1 }}>
          <div style={{ fontSize: '0.65rem', color: 'rgba(255,255,255,0.4)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 6 }}>
            Dominant Emotion
          </div>
          <div style={{ fontSize: '1.5rem', fontWeight: 800, color: `rgb(${domRgb})`, textTransform: 'capitalize', marginBottom: 6 }}>
            {dominant || '—'}
          </div>
          {sarcasm_score > 0 && (
            <div style={{ fontSize: '0.72rem', color: 'rgba(255,255,255,0.45)' }}>
              Sarcasm: {(sarcasm_score * 100).toFixed(0)}%
            </div>
          )}
        </div>
      </div>

      {/* Top aggregated emotions */}
      {decision.aggregated && Object.keys(decision.aggregated).length > 0 && (
        <div style={{ marginTop: 14 }}>
          <div style={S.miniLabel}>Top aggregated scores</div>
          {Object.entries(decision.aggregated)
            .filter(([k]) => !k.startsWith('vader_'))
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5)
            .map(([label, val]) => (
              <Bar key={label} label={label} value={val} color={rgb(label)} highlight={label === dominant} />
            ))}
        </div>
      )}
    </div>
  );
}

function TrajectoryCard({ trajectory, currentDominant, lstmTrajectory }) {
  const hasHistory = trajectory && trajectory.length > 0;
  const hasLSTM    = lstmTrajectory && lstmTrajectory.model_available;

  if (!hasHistory && !hasLSTM) return null;

  const all = hasHistory
    ? [...trajectory, { dominant: currentDominant, isCurrent: true }]
    : [{ dominant: currentDominant, isCurrent: true }];

  const predicted    = lstmTrajectory?.top_predicted;
  const predictedMap = lstmTrajectory?.predicted_next || {};
  const top5Entries  = Object.entries(predictedMap).sort((a, b) => b[1] - a[1]).slice(0, 5);
  const predRgb      = predicted ? rgb(predicted) : '100,180,255';

  return (
    <div style={S.stageCard}>
      <div style={S.stageHeader}>
        <span style={{ fontSize: '1.1rem' }}>📈</span>
        <div>
          <div style={{ ...S.stageTitle, color: 'rgb(100,200,255)' }}>Conversation Trajectory</div>
          <div style={S.stageDesc}>Emotional history + LSTM next-message prediction</div>
        </div>
      </div>

      {/* History chain */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 0, marginTop: 16, overflowX: 'auto', paddingBottom: 4 }}>
        {all.map((item, i) => {
          const emoRgb    = rgb(item.dominant);
          const isCurrent = item.isCurrent;
          return (
            <React.Fragment key={i}>
              {i > 0 && (
                <div style={{ width: 20, height: 2, flexShrink: 0, background: 'rgba(255,255,255,0.15)' }} />
              )}
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4, flexShrink: 0 }}>
                <div style={{
                  width: isCurrent ? 36 : 26, height: isCurrent ? 36 : 26, borderRadius: '50%',
                  background: `rgba(${emoRgb},${isCurrent ? 0.3 : 0.15})`,
                  border: `2px solid rgba(${emoRgb},${isCurrent ? 0.8 : 0.4})`,
                  boxShadow: isCurrent ? `0 0 12px rgba(${emoRgb},0.4)` : 'none',
                }} />
                <span style={{
                  fontSize: isCurrent ? '0.68rem' : '0.6rem',
                  color: isCurrent ? `rgb(${emoRgb})` : 'rgba(255,255,255,0.45)',
                  textTransform: 'capitalize', fontWeight: isCurrent ? 700 : 400,
                  textAlign: 'center', maxWidth: 60,
                }}>
                  {item.dominant || 'neutral'}
                </span>
              </div>
            </React.Fragment>
          );
        })}

        {/* Predicted next node */}
        {hasLSTM && predicted && (
          <>
            <div style={{ width: 24, height: 2, flexShrink: 0, background: `rgba(${predRgb},0.4)`, borderTop: '2px dashed rgba(255,255,255,0.2)' }} />
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4, flexShrink: 0 }}>
              <div style={{
                width: 30, height: 30, borderRadius: '50%', flexShrink: 0,
                background: `rgba(${predRgb},0.1)`,
                border: `2px dashed rgba(${predRgb},0.6)`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <span style={{ fontSize: '0.6rem', color: `rgba(${predRgb},0.7)` }}>?</span>
              </div>
              <span style={{ fontSize: '0.6rem', color: `rgba(${predRgb},0.8)`, textTransform: 'capitalize', textAlign: 'center', maxWidth: 60 }}>
                {predicted}
              </span>
            </div>
          </>
        )}
      </div>

      {/* LSTM predicted next emotion breakdown */}
      {hasLSTM && top5Entries.length > 0 && (
        <div style={{ marginTop: 18, paddingTop: 14, borderTop: '1px solid rgba(255,255,255,0.07)' }}>
          <div style={{ fontSize: '0.72rem', color: 'rgba(255,255,255,0.4)', marginBottom: 10, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
            LSTM — predicted next message
          </div>
          {top5Entries.map(([emo, score]) => (
            <Bar key={emo} label={emo} value={score} color={rgb(emo)} highlight={emo === predicted} />
          ))}
        </div>
      )}

      {!hasLSTM && (
        <div style={{ marginTop: 12, fontSize: '0.72rem', color: 'rgba(255,255,255,0.3)', fontStyle: 'italic' }}>
          LSTM trajectory model not yet loaded — train with more conversations to enable next-message prediction.
        </div>
      )}
    </div>
  );
}

// ── Message List ──────────────────────────────────────────────────────────────

function MessageList({ items, selectedId, onSelect, loading }) {
  const [search, setSearch] = useState('');

  const filtered = items.filter(m =>
    m.text.toLowerCase().includes(search.toLowerCase()) ||
    (m.display_name || '').toLowerCase().includes(search.toLowerCase()) ||
    (m.dominant || '').toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div style={S.listPanel}>
      <div style={S.listHeader}>
        <div style={{ fontWeight: 700, fontSize: '0.9rem', color: '#fff' }}>Recent Analyses</div>
        <div style={{ fontSize: '0.72rem', color: 'rgba(255,255,255,0.35)' }}>{items.length} messages</div>
      </div>

      <div style={{ padding: '8px 12px' }}>
        <input
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Search text, user, emotion…"
          style={S.searchInput}
        />
      </div>

      <div style={{ flex: 1, overflowY: 'auto' }}>
        {loading && <div style={S.emptyMsg}>Loading…</div>}
        {!loading && filtered.length === 0 && (
          <div style={S.emptyMsg}>No messages found.</div>
        )}
        {filtered.map(m => {
          const emoRgb  = rgb(m.dominant);
          const isActive = m.message_id === selectedId;
          return (
            <div
              key={m.message_id}
              onClick={() => onSelect(m.message_id)}
              style={{
                ...S.listItem,
                background: isActive ? 'rgba(0,119,255,0.12)' : 'transparent',
                borderLeft: isActive ? '3px solid #0077ff' : '3px solid transparent',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <span style={{ fontSize: '0.72rem', color: 'rgba(255,255,255,0.45)' }}>
                  {m.display_name}
                </span>
                {m.dominant !== '—' && (
                  <span style={{
                    fontSize: '0.65rem', padding: '1px 7px', borderRadius: 20,
                    background: `rgba(${emoRgb},0.12)`, color: `rgb(${emoRgb})`,
                    border: `1px solid rgba(${emoRgb},0.25)`, textTransform: 'capitalize',
                  }}>
                    {m.dominant}
                    {m.confidence != null && ` · ${m.confidence}%`}
                  </span>
                )}
              </div>
              <div style={{
                fontSize: '0.8rem', color: 'rgba(255,255,255,0.75)',
                overflow: 'hidden', textOverflow: 'ellipsis',
                display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical',
              }}>
                {m.text || <span style={{ color: 'rgba(255,255,255,0.3)', fontStyle: 'italic' }}>No text</span>}
              </div>
              {!m.has_pipeline && (
                <div style={{ fontSize: '0.64rem', color: 'rgba(255,255,255,0.25)', marginTop: 4 }}>
                  ⚠ No pipeline data
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────

export default function AdminPipelinePage({ currentUser, onBack }) {
  const [items, setItems]         = useState([]);
  const [loadingList, setLoadingList] = useState(true);
  const [selectedId, setSelectedId]   = useState(null);
  const [detail, setDetail]           = useState(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [error, setError]               = useState('');

  const loadList = useCallback(async () => {
    setLoadingList(true);
    setError('');
    try {
      const res = await adminAPI.recentAnalyses(80);
      setItems(res.data);
      if (res.data.length > 0 && !selectedId) {
        const first = res.data.find(m => m.has_pipeline) || res.data[0];
        setSelectedId(first.message_id);
      }
    } catch (e) {
      setError(e.apiMessage || 'Failed to load analyses.');
    } finally {
      setLoadingList(false);
    }
  }, []);

  useEffect(() => { loadList(); }, [loadList]);

  useEffect(() => {
    if (!selectedId) return;
    setLoadingDetail(true);
    setDetail(null);
    adminAPI.pipelineDetail(selectedId)
      .then(res => setDetail(res.data))
      .catch(e  => setError(e.apiMessage || 'Failed to load pipeline.'))
      .finally(() => setLoadingDetail(false));
  }, [selectedId]);

  return (
    <div style={S.page}>
      {/* ── Top bar ── */}
      <div style={S.topBar}>
        <button onClick={onBack} style={S.backBtn}>← Back</button>
        <div>
          <span style={S.pageTitle}>🔬 Pipeline Inspector</span>
          <span style={S.pageSub}>Full step-by-step ML analysis breakdown</span>
        </div>
        <button onClick={loadList} style={S.refreshBtn} disabled={loadingList}>
          {loadingList ? '⟳' : '⟳ Refresh'}
        </button>
      </div>

      {error && <div style={S.errorBox}>⚠ {error}</div>}

      <div style={S.body}>
        {/* ── Left: message list ── */}
        <MessageList
          items={items}
          selectedId={selectedId}
          onSelect={setSelectedId}
          loading={loadingList}
        />

        {/* ── Right: pipeline detail ── */}
        <div style={S.detailPanel}>
          {!selectedId && !loadingDetail && (
            <div style={S.placeholder}>Select a message to inspect its pipeline.</div>
          )}

          {loadingDetail && (
            <div style={S.placeholder}>Loading pipeline…</div>
          )}

          {detail && !loadingDetail && (
            <>
              {/* Message header */}
              <div style={S.msgHeader}>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: '0.68rem', color: 'rgba(255,255,255,0.35)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.07em' }}>
                    {detail.message.display_name} · {fmt(detail.message.timestamp)} · conv {detail.message.conversation_id}
                  </div>
                  <div style={{ fontSize: '1.05rem', color: '#fff', fontWeight: 500, lineHeight: 1.5 }}>
                    "{detail.message.text}"
                  </div>
                </div>
                {detail.ground_truth && (
                  <div style={{
                    padding: '4px 12px', borderRadius: 20,
                    background: 'rgba(52,199,89,0.12)', color: 'rgb(52,199,89)',
                    border: '1px solid rgba(52,199,89,0.25)', fontSize: '0.75rem', fontWeight: 600,
                  }}>
                    ✓ Verified: {detail.ground_truth}
                  </div>
                )}
              </div>

              {/* Pipeline flow */}
              <div style={S.pipeline}>
                {/* Step 1 — VADER */}
                <PipelineStep num={1} label="Sentiment Analysis">
                  <VaderCard data={detail.stages.vader} />
                </PipelineStep>

                {/* Step 2 — BERT */}
                <PipelineStep num={2} label="Transformer (BERT)">
                  <StageCard stageKey="bert" data={detail.stages.bert} />
                </PipelineStep>

                {/* Step 3 — GoEmotions */}
                <PipelineStep num={3} label="GoEmotions (28-class)">
                  <StageCard stageKey="goemotions" data={detail.stages.goemotions} />
                </PipelineStep>

                {/* Step 4 — Context */}
                <PipelineStep num={4} label="Context Engine">
                  <ContextCard decision={detail.decision} context={detail.context} />
                </PipelineStep>

                {/* Step 5 — Meta-Learner */}
                <PipelineStep num={5} label="Meta-Learner Decision">
                  <DecisionCard decision={detail.decision} />
                </PipelineStep>

                {/* Step 6 — Logic Map */}
                {detail.decision.logic_map && Object.keys(detail.decision.logic_map).length > 0 && (
                  <PipelineStep num={6} label="Model Contributions">
                    <LogicMapCard logicMap={detail.decision.logic_map} dominant={detail.decision.dominant} />
                  </PipelineStep>
                )}

                {/* Step 8 — Trajectory */}
                <PipelineStep num={8} label="Mood Trajectory">
                  <TrajectoryCard
                    trajectory={detail.trajectory}
                    currentDominant={detail.decision.dominant}
                    lstmTrajectory={detail.context?.lstm_trajectory}
                  />
                </PipelineStep>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function PipelineStep({ num, label, children }) {
  return (
    <div style={{ display: 'flex', gap: 16, marginBottom: 8 }}>
      {/* Connector */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flexShrink: 0 }}>
        <div style={S.stepBubble}>{num}</div>
        <div style={S.stepLine} />
      </div>
      <div style={{ flex: 1, paddingBottom: 8 }}>
        <div style={S.stepLabel}>{label}</div>
        {children}
      </div>
    </div>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const S = {
  page: {
    display: 'flex', flexDirection: 'column', height: '100vh',
    background: '#0b0e1a', color: '#fff', fontFamily: 'inherit', overflow: 'hidden',
  },
  topBar: {
    display: 'flex', alignItems: 'center', gap: 16,
    padding: '14px 24px', borderBottom: '1px solid rgba(255,255,255,0.08)',
    background: 'rgba(255,255,255,0.02)', flexShrink: 0,
  },
  backBtn: {
    padding: '7px 14px', background: 'rgba(255,255,255,0.07)',
    border: '1px solid rgba(255,255,255,0.12)', borderRadius: 8,
    color: '#fff', cursor: 'pointer', fontSize: '0.82rem',
  },
  pageTitle: { fontSize: '1rem', fontWeight: 700, color: '#fff', marginRight: 10 },
  pageSub:   { fontSize: '0.75rem', color: 'rgba(255,255,255,0.4)' },
  refreshBtn: {
    marginLeft: 'auto', padding: '7px 14px',
    background: 'rgba(0,119,255,0.12)', border: '1px solid rgba(0,119,255,0.25)',
    borderRadius: 8, color: '#0077ff', cursor: 'pointer', fontSize: '0.82rem',
  },
  errorBox: {
    margin: '8px 24px', padding: '10px 14px',
    background: 'rgba(255,60,60,0.1)', border: '1px solid rgba(255,60,60,0.25)',
    borderRadius: 10, color: '#ff7070', fontSize: '0.84rem', flexShrink: 0,
  },
  body: {
    display: 'flex', flex: 1, overflow: 'hidden',
  },
  listPanel: {
    width: 280, flexShrink: 0, borderRight: '1px solid rgba(255,255,255,0.07)',
    display: 'flex', flexDirection: 'column', overflow: 'hidden',
    background: 'rgba(255,255,255,0.01)',
  },
  listHeader: {
    padding: '14px 14px 8px', display: 'flex', justifyContent: 'space-between',
    alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.06)',
  },
  searchInput: {
    width: '100%', padding: '8px 10px', boxSizing: 'border-box',
    background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.1)',
    borderRadius: 8, color: '#fff', fontSize: '0.8rem', outline: 'none',
  },
  listItem: {
    padding: '10px 14px', cursor: 'pointer', borderBottom: '1px solid rgba(255,255,255,0.04)',
    transition: 'background 0.15s',
  },
  detailPanel: {
    flex: 1, overflow: 'auto', padding: '20px 24px',
  },
  placeholder: {
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    height: '60%', color: 'rgba(255,255,255,0.3)', fontSize: '0.9rem',
  },
  msgHeader: {
    display: 'flex', gap: 16, alignItems: 'flex-start',
    padding: '16px 20px', background: 'rgba(255,255,255,0.03)',
    border: '1px solid rgba(255,255,255,0.08)', borderRadius: 14,
    marginBottom: 24,
  },
  pipeline: { display: 'flex', flexDirection: 'column' },
  stepBubble: {
    width: 28, height: 28, borderRadius: '50%',
    background: 'rgba(0,119,255,0.15)', border: '1.5px solid rgba(0,119,255,0.4)',
    color: '#0077ff', fontSize: '0.72rem', fontWeight: 700,
    display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
  },
  stepLine: {
    width: 1, flex: 1, background: 'rgba(255,255,255,0.07)', minHeight: 16, marginTop: 4,
  },
  stepLabel: {
    fontSize: '0.68rem', fontWeight: 600, color: 'rgba(255,255,255,0.4)',
    textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 8,
  },
  stageCard: {
    background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)',
    borderRadius: 14, padding: '14px 16px', marginBottom: 4,
  },
  stageHeader: { display: 'flex', alignItems: 'center', gap: 10 },
  stageTitle: { fontSize: '0.85rem', fontWeight: 700 },
  stageDesc:  { fontSize: '0.68rem', color: 'rgba(255,255,255,0.35)', marginTop: 2 },
  emptyMsg:   { fontSize: '0.78rem', color: 'rgba(255,255,255,0.3)', textAlign: 'center', padding: '16px 0' },
  miniLabel:  { fontSize: '0.65rem', color: 'rgba(255,255,255,0.35)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 8 },
  contextRow: { display: 'flex', alignItems: 'center' },
  ctxLabel:   { fontSize: '0.75rem', color: 'rgba(255,255,255,0.5)', whiteSpace: 'nowrap' },
  ctxValue:   { fontSize: '0.75rem', fontWeight: 700, whiteSpace: 'nowrap' },
  contextChip: {
    display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap',
    padding: '7px 10px', background: 'rgba(255,100,150,0.06)',
    border: '1px solid rgba(255,100,150,0.15)', borderRadius: 8,
    fontSize: '0.75rem', color: 'rgba(255,255,255,0.75)',
  },
};

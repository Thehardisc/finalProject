import { useState } from 'react';
import { EmotionPalette } from './EmotionPalette';
import CDMStateGraph from './CDMStateGraph';
import PixelFace from './PixelFace';

const STAGES = [
  { key: 'vader',       label: 'VADER',          color: '234,179,8',   sub: 'Lexical sentiment · 4-dim',     modelKey: 'vader'       },
  { key: 'bert',        label: 'BERT Ekman',      color: '59,130,246',  sub: 'Transformer · 7 Ekman classes', modelKey: 'basic_bert'  },
  { key: 'goemotions',  label: 'GoEmotions',      color: '168,85,247',  sub: 'Fine-grained · 28 classes',     modelKey: 'go_emotions' },
  { key: 'context',     label: 'Context Engine',  color: '20,184,166',  sub: 'CDM · 23-dim semantic state',   modelKey: null          },
  { key: 'meta',        label: 'Meta-Learner',    color: '16,185,129',  sub: 'MLP · 62-dim → final decision', modelKey: null          },
];

const PARTIAL_MAP = {
  vader:      'vader',
  basic_bert: 'bert',
  go_emotions:'goemotions',
};

export default function TelemetryPanel({ processing, lastAnalysis, partialModels = new Set() }) {
  const [tab, setTab] = useState('pipeline');

  const d       = lastAnalysis?.data;
  const pl      = d ? null : null;
  const logMap  = d?.logic_map    ?? {};
  const conf    = d?.meta_confidence ?? null;
  const snap    = d?.context_snapshot ?? null;
  const traj    = d?.lstm_trajectory;
  const domEmo  = d?.final_dominant_emotion;
  const domRgb  = domEmo ? (EmotionPalette[domEmo.toLowerCase()] || EmotionPalette.neutral) : null;

  const stageStatus = (stage) => {
    if (!processing && !d) return 'idle';
    if (d) return 'done';
    if (stage.modelKey && partialModels.has(stage.modelKey)) return 'done';
    if (stage.key === 'context' || stage.key === 'meta') return 'idle';
    if (processing) return 'active';
    return 'idle';
  };

  const busy = processing || STAGES.some(s => stageStatus(s) === 'active');

  const latencyLabel = d ? null : null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflowY: 'auto', padding: '16px 14px' }}>

      <div style={{ marginBottom: 14 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ fontWeight: 700, fontSize: '0.82rem', color: 'var(--ig-txt)' }}>Pipeline Telemetry</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <span style={{
              display: 'inline-block', width: 7, height: 7, borderRadius: '50%',
              background: busy ? 'rgb(234,179,8)' : 'rgb(16,185,129)',
              boxShadow: busy ? '0 0 8px rgba(234,179,8,.65)' : '0 0 4px rgba(16,185,129,.5)',
              animation: busy ? 'tp-pulse .75s ease-in-out infinite' : 'none',
            }} />
            <span style={{ fontSize: '0.69rem', fontWeight: 600, color: busy ? 'rgb(234,179,8)' : 'rgb(16,185,129)' }}>
              {busy ? 'Processing…' : d ? 'Healthy' : 'Idle'}
            </span>
          </div>
        </div>
        <div style={{ fontSize: '0.67rem', color: 'var(--ig-txt3)', marginTop: 3 }}>
          4-model multimodal NLP + CDM context
        </div>
      </div>

      <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 14, paddingTop: 4 }}>
        <PixelFace emotion={domEmo || 'neutral'} scale={5} showLabel={true} />
      </div>

      <div style={{ display: 'flex', gap: 4, marginBottom: 14 }}>
        {['pipeline', 'cdm'].map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            flex: 1, padding: '5px 0', borderRadius: 7, cursor: 'pointer',
            background: tab === t ? 'rgba(139,92,246,.10)' : 'transparent',
            border: `1px solid ${tab === t ? 'rgba(139,92,246,.30)' : 'rgba(var(--ig-ink-rgb),.08)'}`,
            color: tab === t ? 'rgb(139,92,246)' : 'var(--ig-txt2)',
            fontSize: '0.70rem', fontWeight: 600,
            transition: 'all .15s',
          }}>
            {t === 'pipeline' ? 'Pipeline' : 'State Machine'}
          </button>
        ))}
      </div>

      {tab === 'pipeline' && (
        <>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 16 }}>
            {STAGES.map((stage) => {
              const status = stageStatus(stage);
              const active = status === 'active';
              const done   = status === 'done';
              const { color } = stage;
              return (
                <div key={stage.key} style={{
                  background: active ? `rgba(${color},.07)` : 'var(--ig-surf2)',
                  border: `1px solid ${active ? `rgba(${color},.22)` : done ? `rgba(${color},.15)` : 'rgba(var(--ig-ink-rgb),.06)'}`,
                  borderRadius: 9, padding: '8px 11px',
                  transition: 'background .2s, border-color .2s',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 5 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                      <div style={{
                        width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
                        background: active ? `rgb(${color})` : done ? `rgba(${color},.60)` : 'rgba(var(--ig-ink-rgb),.14)',
                        boxShadow: active ? `0 0 7px rgba(${color},.75)` : 'none',
                        animation: active ? 'tp-pulse .65s ease-in-out infinite' : 'none',
                        transition: 'background .15s',
                      }} />
                      <span style={{ fontSize: '0.78rem', fontWeight: 600, color: active ? `rgb(${color})` : 'rgba(var(--ig-ink-rgb),.80)' }}>
                        {stage.label}
                      </span>
                    </div>
                    {active && (
                      <span style={{ fontSize: '0.65rem', color: `rgb(${color})`, fontWeight: 700, letterSpacing: 2 }}>···</span>
                    )}
                    {done && (
                      <span style={{ fontSize: '0.65rem', fontWeight: 700, color: `rgb(${color})`, background: `rgba(${color},.10)`, padding: '2px 6px', borderRadius: 20 }}>
                        ✓
                      </span>
                    )}
                  </div>
                  <div style={{ height: 3, borderRadius: 2, background: 'rgba(var(--ig-ink-rgb),.08)', overflow: 'hidden' }}>
                    <div style={{
                      height: '100%',
                      width: active ? '65%' : done ? '100%' : '0%',
                      background: `rgb(${color})`,
                      borderRadius: 2, opacity: done ? .40 : 1,
                      transition: active ? 'width .18s linear' : 'width .12s ease',
                    }} />
                  </div>
                  <div style={{ fontSize: '0.62rem', color: 'var(--ig-txt3)', marginTop: 4 }}>{stage.sub}</div>
                </div>
              );
            })}
          </div>

          {d && Object.keys(logMap).length > 0 && (
            <div style={{
              background: domRgb ? `rgba(${domRgb},.05)` : 'var(--ig-surf2)',
              border: `1px solid ${domRgb ? `rgba(${domRgb},.15)` : 'rgba(var(--ig-ink-rgb),.07)'}`,
              borderRadius: 10, padding: '11px 13px', marginBottom: 14,
            }}>
              <div style={{ fontSize: '0.63rem', color: 'var(--ig-txt3)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '.07em', marginBottom: 9 }}>
                Feature Attribution
              </div>
              {Object.entries(logMap).map(([block, v]) => (
                <div key={block} style={{ marginBottom: 6 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                    <span style={{ fontSize: '0.63rem', color: 'var(--ig-txt2)' }}>{block}</span>
                    <span style={{ fontSize: '0.63rem', fontWeight: 700, color: `rgb(${domRgb || '88,86,214'})` }}>
                      {(Math.abs(v) * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div style={{ height: 2, background: 'rgba(var(--ig-ink-rgb),.08)', borderRadius: 1 }}>
                    <div style={{
                      height: '100%',
                      width: `${Math.min(Math.abs(v) * 100, 100)}%`,
                      background: `rgb(${domRgb || '88,86,214'})`,
                      borderRadius: 1, transition: 'width .4s ease',
                    }} />
                  </div>
                </div>
              ))}
            </div>
          )}

          {d && domRgb && (
            <div style={{
              background: `rgba(${domRgb},.06)`,
              border: `1px solid rgba(${domRgb},.18)`,
              borderRadius: 10, padding: '11px 13px',
            }}>
              <div style={{ fontSize: '0.63rem', color: 'var(--ig-txt3)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '.07em', marginBottom: 7 }}>
                Last Inference
              </div>
              <div style={{ fontSize: '0.92rem', fontWeight: 800, color: `rgb(${domRgb})`, textTransform: 'capitalize', marginBottom: 2 }}>
                {domEmo}
              </div>
              {conf != null && (
                <div style={{ fontSize: '0.73rem', color: 'var(--ig-txt2)', marginBottom: traj ? 8 : 0 }}>
                  {(conf * 100).toFixed(0)}% confidence
                </div>
              )}
              {traj?.top_predicted && (
                <div style={{ fontSize: '0.68rem', color: 'var(--ig-txt3)', marginTop: 4 }}>
                  Next predicted:{' '}
                  <span style={{ color: `rgb(${domRgb})`, fontWeight: 700, textTransform: 'capitalize' }}>
                    {traj.top_predicted}
                  </span>
                </div>
              )}
            </div>
          )}

          {!d && (
            <div style={{ textAlign: 'center', padding: '20px 0', color: 'var(--ig-txt3)', fontSize: '0.8rem' }}>
              Send a message to see<br />live model activations
            </div>
          )}
        </>
      )}

      {tab === 'cdm' && (
        <CDMStateGraph snapshot={snap} />
      )}

      <style>{`
        @keyframes tp-pulse {
          0%,100%{ opacity:1; transform:scale(1); }
          50%    { opacity:.4; transform:scale(.8); }
        }
      `}</style>
    </div>
  );
}

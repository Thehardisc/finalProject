import React from 'react';

const CDM_STATES = [
  { id: 0, label: 'Neutral\nExplore',    short: 'NEUTRAL',    color: '107,114,128' },
  { id: 1, label: 'Ascending\nPositive', short: 'ASCENDING',  color: '34,197,94'   },
  { id: 2, label: 'Conflict\nRise',      short: 'CONFLICT',   color: '239,68,68'   },
  { id: 3, label: 'Topic\nAnchor',       short: 'ANCHOR',     color: '59,130,246'  },
  { id: 4, label: 'Convergence',         short: 'CONVERGE',   color: '20,184,166'  },
  { id: 5, label: 'Emotional\nPeak',     short: 'PEAK',       color: '168,85,247'  },
  { id: 6, label: 'Diverging',           short: 'DIVERG',     color: '249,115,22'  },
];

// Positions in a 260×170 SVG canvas — hex-ring around center (0)
const POS = [
  { x: 130, y:  85 }, // 0 center
  { x:  65, y:  38 }, // 1 top-left
  { x:  65, y: 132 }, // 2 bottom-left
  { x:  20, y:  85 }, // 3 left
  { x: 195, y:  38 }, // 4 top-right
  { x: 195, y: 132 }, // 5 bottom-right
  { x: 240, y:  85 }, // 6 right
];

// Semantically meaningful transitions (src, dst)
const EDGES = [
  [0,1],[0,2],[0,3],[0,4],[0,5],[0,6],
  [1,4],[1,5],
  [2,5],[2,6],
  [4,1],
  [5,6],
];

const MIN_R = 10;
const MAX_R = 24;

export default function CDMStateGraph({ snapshot }) {
  const probs    = snapshot?.cdm_state_probs;
  const current  = snapshot?.cdm_current_state;
  const residency = snapshot?.cdm_residency ?? 0;
  const available = snapshot?.cdm_available ?? false;

  // Normalise probabilities for radius scaling
  const normProbs = (probs && probs.some(p => p > 0))
    ? probs
    : CDM_STATES.map((_, i) => i === 0 ? 1 : 0);

  const maxP = Math.max(...normProbs, 0.001);
  const radius = (i) => MIN_R + (normProbs[i] / maxP) * (MAX_R - MIN_R);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ fontSize: '0.72rem', fontWeight: 700, color: '#374151' }}>
          Conversation Dynamics
        </span>
        <span style={{
          fontSize: '0.62rem', padding: '2px 7px', borderRadius: 20,
          background: available ? 'rgba(20,184,166,.12)' : 'rgba(0,0,0,.06)',
          color: available ? 'rgb(20,184,166)' : '#9ca3af',
          fontWeight: 600,
        }}>
          {available ? 'LIVE' : 'NO DATA'}
        </span>
      </div>

      {/* SVG Graph */}
      <svg
        viewBox="0 0 260 170"
        style={{ width: '100%', height: 'auto', overflow: 'visible' }}
      >
        {/* Edges */}
        {EDGES.map(([a, b], i) => {
          const pa = POS[a], pb = POS[b];
          const active = available && (a === current || b === current);
          return (
            <line
              key={i}
              x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y}
              stroke={active ? `rgba(${CDM_STATES[a].color},.35)` : 'rgba(0,0,0,.07)'}
              strokeWidth={active ? 1.5 : 1}
            />
          );
        })}

        {/* Nodes */}
        {CDM_STATES.map(({ id, short, color }) => {
          const { x, y } = POS[id];
          const r         = radius(id);
          const isCurrent = available && id === current;
          const prob       = normProbs[id];

          return (
            <g key={id}>
              {/* Glow ring for current state */}
              {isCurrent && (
                <circle
                  cx={x} cy={y} r={r + 6}
                  fill="none"
                  stroke={`rgba(${color},.30)`}
                  strokeWidth={4}
                />
              )}
              {/* Node circle */}
              <circle
                cx={x} cy={y} r={r}
                fill={isCurrent ? `rgba(${color},.20)` : `rgba(${color},.08)`}
                stroke={`rgba(${color},${isCurrent ? '.80' : '.35'})`}
                strokeWidth={isCurrent ? 2 : 1}
              />
              {/* Label */}
              <text
                x={x} y={y + r + 9}
                textAnchor="middle"
                fontSize="6.5"
                fill={isCurrent ? `rgb(${color})` : '#6b7280'}
                fontWeight={isCurrent ? '700' : '400'}
              >
                {short}
              </text>
              {/* Probability badge */}
              {available && prob > 0.05 && (
                <text
                  x={x} y={y + 2.5}
                  textAnchor="middle"
                  fontSize="6"
                  fill={isCurrent ? `rgb(${color})` : '#9ca3af'}
                  fontWeight="600"
                >
                  {(prob * 100).toFixed(0)}%
                </text>
              )}
            </g>
          );
        })}
      </svg>

      {/* Current state label + residency bar */}
      {available && current != null && (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
            <span style={{
              fontSize: '0.70rem', fontWeight: 700,
              color: `rgb(${CDM_STATES[current].color})`,
            }}>
              {CDM_STATES[current].label.replace('\n', ' ')}
            </span>
            <span style={{ fontSize: '0.62rem', color: '#9ca3af' }}>
              residency {(residency * 100).toFixed(0)}%
            </span>
          </div>
          <div style={{ height: 3, borderRadius: 2, background: 'rgba(0,0,0,.07)' }}>
            <div style={{
              height: '100%',
              width: `${Math.min(residency * 100, 100)}%`,
              background: `rgb(${CDM_STATES[current].color})`,
              borderRadius: 2,
              transition: 'width .4s ease',
            }} />
          </div>
        </div>
      )}

      {!available && (
        <div style={{ textAlign: 'center', fontSize: '0.70rem', color: '#9ca3af', padding: '4px 0' }}>
          CDM activates after first message
        </div>
      )}
    </div>
  );
}

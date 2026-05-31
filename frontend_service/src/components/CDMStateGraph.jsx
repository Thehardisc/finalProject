import React from 'react';

// CDM v4.0 — 15 intent-based conversational states
// Color palette: collaborative=teal/green, exploratory=blue/indigo, conflictual=red/orange, repair=violet, neutral=gray
const CDM_STATES = [
  { id:  0, label: 'Neutral',         short: 'NEUTRAL',    color: '107,114,128' },
  { id:  1, label: 'Warmth',          short: 'WARMTH',     color: '20,184,166'  },
  { id:  2, label: 'Praise',          short: 'PRAISE',     color: '34,197,94'   },
  { id:  3, label: 'Help\nRequest',   short: 'HELP',       color: '59,130,246'  },
  { id:  4, label: 'Humor',           short: 'HUMOR',      color: '251,191,36'  },
  { id:  5, label: 'Tension',         short: 'TENSION',    color: '249,115,22'  },
  { id:  6, label: 'Conflict',        short: 'CONFLICT',   color: '239,68,68'   },
  { id:  7, label: 'Argument',        short: 'ARGUMENT',   color: '220,38,38'   },
  { id:  8, label: 'Withdrawal',      short: 'WITHDRAW',   color: '168,85,247'  },
  { id:  9, label: 'Reconciliation',  short: 'RECONCILE',  color: '139,92,246'  },
  { id: 10, label: 'Curiosity',       short: 'CURIOSITY',  color: '14,165,233'  },
  { id: 11, label: 'Assertiveness',   short: 'ASSERT',     color: '99,102,241'  },
  { id: 12, label: 'Empathy',         short: 'EMPATHY',    color: '16,185,129'  },
  { id: 13, label: 'Frustration',     short: 'FRUSTRAT',   color: '234,88,12'   },
  { id: 14, label: 'Agreement',       short: 'AGREE',      color: '52,211,153'  },
];

// SVG canvas: 320 × 230
// Semantic layout:
//   Top-left  (collaborative): WARMTH(1) PRAISE(2) EMPATHY(12)
//   Top-right (exploratory):   CURIOSITY(10) HELP_REQUEST(3) HUMOR(4)
//   Top-center:                ASSERTIVENESS(11)
//   Center:                    NEUTRAL(0)
//   Mid-center:                RECONCILIATION(9) AGREEMENT(14)
//   Bottom-left (disengaging): WITHDRAWAL(8) FRUSTRATION(13)
//   Bottom-right (conflictual):TENSION(5) CONFLICT(6) ARGUMENT(7)
const POS = [
  { x: 160, y: 112 }, //  0 NEUTRAL           center
  { x:  42, y:  40 }, //  1 WARMTH             top-left
  { x:  95, y:  25 }, //  2 PRAISE             top-left
  { x: 288, y:  68 }, //  3 HELP_REQUEST       top-right
  { x: 228, y:  26 }, //  4 HUMOR              top-right
  { x: 255, y: 168 }, //  5 TENSION            bottom-right
  { x: 293, y: 138 }, //  6 CONFLICT           bottom-right
  { x: 293, y: 195 }, //  7 ARGUMENT           bottom-right bottom
  { x:  38, y: 190 }, //  8 WITHDRAWAL         bottom-left
  { x:  95, y: 162 }, //  9 RECONCILIATION     bottom-left area
  { x: 248, y:  38 }, // 10 CURIOSITY          top-right
  { x: 160, y:  42 }, // 11 ASSERTIVENESS      top-center
  { x:  35, y:  88 }, // 12 EMPATHY            left-mid
  { x: 105, y: 185 }, // 13 FRUSTRATION        bottom-left area
  { x: 222, y: 162 }, // 14 AGREEMENT          bottom-center-right
];

// Intent-based edges — semantically meaningful flows between states
const EDGES = [
  // NEUTRAL hub
  [0,1],[0,2],[0,10],[0,11],[0,5],[0,3],[0,4],
  // Collaborative cluster
  [1,2],[1,12],[1,14],[1,9],
  [2,1],[2,14],[2,4],
  [12,1],[12,9],[12,0],
  [14,1],[14,11],[14,0],
  // Exploratory cluster
  [10,3],[10,4],[10,11],[10,0],
  [3,10],[3,12],[3,11],
  [4,1],[4,10],[4,0],
  // Repair path
  [9,1],[9,14],[9,0],
  [8,9],[8,0],
  // Conflict escalation
  [5,6],[5,13],[5,0],
  [6,7],[6,9],[6,5],
  [7,6],[7,9],[7,13],
  [13,6],[13,8],[13,5],
  // Assertiveness cross-links
  [11,6],[11,14],[11,0],[11,3],
];

const MIN_R = 8;
const MAX_R = 20;

export default function CDMStateGraph({ snapshot }) {
  const probs     = snapshot?.cdm_state_probs;
  const current   = snapshot?.cdm_current_state;
  const residency = snapshot?.cdm_residency ?? 0;
  const available = snapshot?.cdm_available ?? false;

  const normProbs = (probs && probs.length === CDM_STATES.length && probs.some(p => p > 0))
    ? probs
    : CDM_STATES.map((_, i) => i === 0 ? 1 : 0);

  const maxP   = Math.max(...normProbs, 0.001);
  const radius = (i) => MIN_R + (normProbs[i] / maxP) * (MAX_R - MIN_R);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ fontSize: '0.72rem', fontWeight: 700, color: '#374151' }}>
          Conversation Intent
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {available && current != null && (
            <span style={{
              fontSize: '0.60rem', padding: '1px 6px', borderRadius: 20,
              background: `rgba(${CDM_STATES[current]?.color},.15)`,
              color: `rgb(${CDM_STATES[current]?.color})`,
              fontWeight: 700, border: `1px solid rgba(${CDM_STATES[current]?.color},.3)`,
            }}>
              {CDM_STATES[current]?.short}
            </span>
          )}
          <span style={{
            fontSize: '0.60rem', padding: '1px 6px', borderRadius: 20,
            background: available ? 'rgba(20,184,166,.12)' : 'rgba(0,0,0,.06)',
            color: available ? 'rgb(20,184,166)' : '#9ca3af',
            fontWeight: 600,
          }}>
            {available ? 'LIVE' : 'NO DATA'}
          </span>
        </div>
      </div>

      {/* SVG Graph */}
      <svg
        viewBox="0 0 320 230"
        style={{ width: '100%', height: 'auto', overflow: 'visible' }}
      >
        {/* Edges */}
        {EDGES.map(([a, b], i) => {
          const pa     = POS[a];
          const pb     = POS[b];
          const active = available && (a === current || b === current);
          return (
            <line
              key={i}
              x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y}
              stroke={active ? `rgba(${CDM_STATES[a].color},.40)` : 'rgba(0,0,0,.06)'}
              strokeWidth={active ? 1.5 : 0.8}
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
              {isCurrent && (
                <circle
                  cx={x} cy={y} r={r + 5}
                  fill="none"
                  stroke={`rgba(${color},.30)`}
                  strokeWidth={3.5}
                />
              )}
              <circle
                cx={x} cy={y} r={r}
                fill={isCurrent ? `rgba(${color},.22)` : `rgba(${color},.07)`}
                stroke={`rgba(${color},${isCurrent ? '.85' : '.30'})`}
                strokeWidth={isCurrent ? 2 : 0.9}
              />
              <text
                x={x} y={y + r + 8}
                textAnchor="middle"
                fontSize="5.5"
                fill={isCurrent ? `rgb(${color})` : '#9ca3af'}
                fontWeight={isCurrent ? '700' : '400'}
              >
                {short}
              </text>
              {available && prob > 0.08 && (
                <text
                  x={x} y={y + 2}
                  textAnchor="middle"
                  fontSize="5.5"
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

      {/* Current state label + residency */}
      {available && current != null && (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
            <span style={{
              fontSize: '0.68rem', fontWeight: 700,
              color: `rgb(${CDM_STATES[current].color})`,
            }}>
              {CDM_STATES[current].label.replace('\n', ' ')}
            </span>
            <span style={{ fontSize: '0.60rem', color: '#9ca3af' }}>
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
        <div style={{ textAlign: 'center', fontSize: '0.68rem', color: '#9ca3af', padding: '4px 0' }}>
          CDM activates after first message
        </div>
      )}
    </div>
  );
}

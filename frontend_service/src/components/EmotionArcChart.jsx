import { useState, useId } from 'react';

export default function EmotionArcChart({ values = [], color = '99,102,241', height = 72 }) {
  const id = useId().replace(/:/g, '');
  const [hovered, setHovered] = useState(null);

  if (values.length < 2) return null;

  const W = 280, H = height;
  const PAD = 8;
  const toX = i => PAD + (i / (values.length - 1)) * (W - PAD * 2);
  const toY = v => (H / 2) - (v * (H / 2 - PAD) * 0.85);
  const pts = values.map((v, i) => [toX(i), toY(v)]);

  let pathD = `M${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`;
  for (let i = 1; i < pts.length; i++) {
    const [px, py] = pts[i - 1];
    const [cx, cy] = pts[i];
    const cpx = (px + cx) / 2;
    pathD += ` C${cpx.toFixed(1)},${py.toFixed(1)} ${cpx.toFixed(1)},${cy.toFixed(1)} ${cx.toFixed(1)},${cy.toFixed(1)}`;
  }
  const areaD = pathD + ` L${pts[pts.length-1][0].toFixed(1)},${H} L${pts[0][0].toFixed(1)},${H} Z`;

  const lastPt  = pts[pts.length - 1];
  const lastVal = values[values.length - 1];
  const dotColor = lastVal > 0.15 ? '52,211,153' : lastVal < -0.15 ? '248,113,113' : color;

  return (
    <div style={{ position: 'relative', userSelect: 'none' }}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        height={H}
        preserveAspectRatio="none"
        style={{ display: 'block', overflow: 'visible' }}
        onMouseLeave={() => setHovered(null)}
      >
        <defs>
          <linearGradient id={`arc-fill-${id}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor={`rgb(${color})`} stopOpacity="0.25" />
            <stop offset="100%" stopColor={`rgb(${color})`} stopOpacity="0.0"  />
          </linearGradient>
          <filter id={`arc-glow-${id}`}>
            <feGaussianBlur stdDeviation="2" result="blur"/>
            <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>

        <line
          x1={PAD} y1={H / 2} x2={W - PAD} y2={H / 2}
          stroke="rgba(var(--ig-ink-rgb),0.10)" strokeWidth="1" strokeDasharray="5 4"
        />

        <path d={areaD} fill={`url(#arc-fill-${id})`} />

        <path
          d={pathD} fill="none"
          stroke={`rgb(${color})`} strokeWidth="1.8"
          strokeLinecap="round" strokeLinejoin="round"
          filter={`url(#arc-glow-${id})`}
        />

        {pts.map(([x, y], i) => (
          <circle
            key={i}
            cx={x} cy={y} r={hovered === i ? 5 : 0}
            fill={`rgb(${color})`}
            style={{ transition: 'r 0.12s ease', cursor: 'crosshair' }}
            onMouseEnter={() => setHovered(i)}
          />
        ))}

        <circle cx={lastPt[0]} cy={lastPt[1]} r="3.5" fill={`rgb(${dotColor})`}
          style={{ filter: `drop-shadow(0 0 5px rgb(${dotColor}))` }}
        />
        <circle cx={lastPt[0]} cy={lastPt[1]} r="7" fill="none"
          stroke={`rgba(${dotColor},0.35)`} strokeWidth="1.5"
          style={{ animation: 'glowPulse 2s ease-in-out infinite' }}
        />
      </svg>

      {hovered !== null && (
        <div style={{
          position: 'absolute',
          left: `${(pts[hovered][0] / W) * 100}%`,
          top: pts[hovered][1] < H / 2 ? pts[hovered][1] + 10 : pts[hovered][1] - 36,
          transform: 'translateX(-50%)',
          pointerEvents: 'none',
          background: 'rgba(10,10,20,0.95)',
          border: '1px solid rgba(255,255,255,0.10)',
          borderRadius: 6,
          padding: '3px 8px',
          fontSize: '0.66rem',
          fontWeight: 700,
          color: '#ededf5',
          whiteSpace: 'nowrap',
          zIndex: 10,
        }}>
          {values[hovered] >= 0 ? '+' : ''}{values[hovered].toFixed(2)}
        </div>
      )}

      <style>{`
        @keyframes glowPulse {
          0%, 100% { opacity: 1;   r: 7; }
          50%       { opacity: 0.3; r: 12; }
        }
      `}</style>
    </div>
  );
}

import { useMemo } from 'react';

function valenceToHsl(valence = 0, volatility = 0) {
  let hue;
  if (valence > 0.25) {
    hue = 215;
  } else if (valence < -0.18) {
    // Negative valence: indigo → magenta → red. Ramping the other way
    // (245 → 28) crosses green/yellow mid-ramp, which tints the whole
    // light-mode panel yellowish.
    const t = Math.min(Math.abs(valence + 0.18) / 0.6, 1);
    hue = Math.round(245 + t * (355 - 245));
  } else {
    hue = 245;
  }
  const sat = 65 + volatility * 12;
  const lit = 56;
  const opacity = Math.min(0.09 + Math.abs(valence) * 0.10 + volatility * 0.04, 0.22);
  return { hue, sat: Math.min(sat, 80), lit, opacity };
}

export default function AmbientOrb({ valence = 0, volatility = 0 }) {
  const { hue, sat, lit, opacity } = useMemo(
    () => valenceToHsl(valence, volatility),
    [valence, volatility],
  );

  return (
    <div
      aria-hidden="true"
      className="fixed w-[680px] h-[680px] rounded-full -right-[200px] -bottom-[220px] blur-[72px] pointer-events-none z-0"
      style={{
        background: `radial-gradient(circle, hsl(${hue} ${sat}% ${lit}% / ${opacity}) 0%, transparent 68%)`,
        transition: 'background var(--shift-duration, 960ms) var(--shift-easing, ease)',
        willChange: 'background',
      }}
    />
  );
}

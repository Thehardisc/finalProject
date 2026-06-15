import React, { useMemo } from 'react';

/**
 * AmbientOrb — Trajectory lighting layer.
 *
 * Renders a single large, fully-blurred radial-gradient anchored to the
 * bottom-right of the viewport.  Color shifts slowly (960 ms) between
 * cool blue (positive valence) ↔ warm amber (tension) ↔ indigo (neutral).
 *
 * Perceptual basis: Weiser & Brown (1996) "Designing Calm Technology" —
 * peripheral cues change state without demanding focal attention.
 * The 960 ms transition is intentionally above the 500 ms startle threshold
 * (Kahneman System 1) so the brain processes it as environment, not alert.
 */

function valenceToHsl(valence = 0, volatility = 0) {
  // hue: 215 (blue) → 245 (indigo) → 28 (amber)
  let hue;
  if (valence > 0.25) {
    hue = 215;                                          // positive: cool blue
  } else if (valence < -0.18) {
    // blend toward amber as valence drops
    const t = Math.min(Math.abs(valence + 0.18) / 0.6, 1);
    hue = Math.round(245 - t * (245 - 28));            // indigo → amber
  } else {
    hue = 245;                                          // baseline: indigo
  }

  const sat = 65 + volatility * 12;                    // volatility widens saturation
  const lit = 56;
  const absVal = Math.abs(valence);
  const opacity = Math.min(0.09 + absVal * 0.10 + volatility * 0.04, 0.22);

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
      style={{
        position: 'fixed',
        width: 680,
        height: 680,
        borderRadius: '50%',
        right: -200,
        bottom: -220,
        background: `radial-gradient(circle,
          hsl(${hue} ${sat}% ${lit}% / ${opacity}) 0%,
          transparent 68%
        )`,
        filter: 'blur(72px)',
        pointerEvents: 'none',
        zIndex: 0,
        transition: `background var(--shift-duration, 960ms) var(--shift-easing, ease)`,
        willChange: 'background',
      }}
    />
  );
}

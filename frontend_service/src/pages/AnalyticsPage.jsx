import React, { useState } from 'react';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, PointElement, LineElement, Filler } from 'chart.js';
import { Doughnut, Line } from 'react-chartjs-2';
import { EmotionPalette, blendEmotionsGradient } from '../components/EmotionPalette';
import AdminDashboard from '../components/AdminDashboard';

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, PointElement, LineElement, Filler);

const ALL_RANGES = ['7d', '30d', '90d', 'all'];
const ACCENT = '0, 119, 255';

function emotionRgb(emo) {
  return EmotionPalette[emo?.toLowerCase()] || EmotionPalette.neutral;
}

export default function AnalyticsPage({
  analyticsData,
  messages,
  currentAnalysis,
  currentUser,
  onBack,
  onRefresh,
}) {
  const [range, setRange] = useState('30d');
  const [showAdmin, setShowAdmin] = useState(false);

  // Build doughnut from current conversation emotion breakdown
  const emotionBreakdown = analyticsData?.emotion_breakdown || {};
  const topEmotions = Object.entries(emotionBreakdown)
    .sort((a, b) => b[1].samples - a[1].samples)
    .slice(0, 8);

  const doughnutData = {
    labels: topEmotions.map(([emo]) => emo),
    datasets: [{
      data: topEmotions.map(([, s]) => s.samples),
      backgroundColor: topEmotions.map(([emo]) => `rgba(${emotionRgb(emo)}, 0.75)`),
      borderColor: topEmotions.map(([emo]) => `rgba(${emotionRgb(emo)}, 0.95)`),
      borderWidth: 2,
    }],
  };

  const doughnutOpts = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { position: 'right', labels: { color: '#374151', font: { size: 11 } } } },
    cutout: '62%',
  };

  // Emotion progression line chart from conversation messages
  const analyzed = messages.filter(m => m.analysis?.data?.bert_emotions);
  const lineLabels = analyzed.map((_, i) => `#${i + 1}`);
  const lineData = {
    labels: lineLabels,
    datasets: [{
      label: 'Top Emotion Score',
      data: analyzed.map(m => {
        const bert = m.analysis.data.bert_emotions;
        return bert.length > 0 ? Math.max(...bert.map(b => b.score)) : 0;
      }),
      borderColor: `rgb(${ACCENT})`,
      backgroundColor: `rgba(${ACCENT}, 0.08)`,
      fill: true, tension: 0.4,
      pointBackgroundColor: analyzed.map(m => {
        const dom = m.analysis?.data?.final_dominant_emotion;
        return `rgb(${emotionRgb(dom)})`;
      }),
      pointRadius: 5, pointHoverRadius: 7,
    }],
  };
  const lineOpts = {
    responsive: true, maintainAspectRatio: false,
    scales: {
      y: { min: 0, max: 1, grid: { color: 'rgba(0,0,0,0.04)' }, ticks: { color: '#9ca3af', font: { size: 10 } } },
      x: { grid: { display: false }, ticks: { color: '#9ca3af', font: { size: 10 } } },
    },
    plugins: { legend: { display: false } },
  };

  const accuracy = analyticsData?.overall_accuracy;
  const totalSamples = analyticsData?.total_verified_samples || 0;

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* ── Top Bar ──────────────────────────────────────────────────────── */}
      <header style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '18px 32px',
        position: 'sticky', top: 0, zIndex: 10,
        background: 'rgba(255,255,255,0.78)',
        backdropFilter: 'blur(24px) saturate(180%)',
        WebkitBackdropFilter: 'blur(24px) saturate(180%)',
        borderBottom: '1px solid rgba(255,255,255,0.92)',
        boxShadow: '0 2px 16px rgba(0,0,0,0.05)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <button onClick={onBack} style={{
            background: 'rgba(0,0,0,0.05)', border: 'none', borderRadius: 10,
            width: 36, height: 36, cursor: 'pointer',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '1rem', color: '#374151',
          }}>←</button>
          <div>
            <h1 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 800, color: '#1c1c2e' }}>
              Ethereal Insights
            </h1>
            <p style={{ margin: 0, fontSize: '0.78rem', color: '#9ca3af' }}>
              {totalSamples} verified samples
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {/* Filter Range sunken tabs */}
          <div style={{
            display: 'flex', gap: 4, padding: '4px',
            background: 'rgba(0,0,0,0.04)', borderRadius: 12,
          }}>
            {ALL_RANGES.map(r => (
              <button key={r} className={`btn-tab ${range === r ? 'active' : ''}`}
                style={{ '--bubble-rgb': ACCENT, padding: '6px 14px', fontSize: '0.8rem' }}
                onClick={() => setRange(r)}>
                {r}
              </button>
            ))}
          </div>

          {currentUser?.role === 'admin' && (
            <button className="btn-tab"
              style={{ '--bubble-rgb': ACCENT }}
              onClick={() => setShowAdmin(s => !s)}>
              Admin
            </button>
          )}

          <button className="btn-sweep"
            style={{ '--bubble-rgb': ACCENT, padding: '10px 22px', fontSize: '0.85rem' }}
            onClick={onRefresh}>
            Generate Insights
          </button>
        </div>
      </header>

      {/* ── Stacked Glass Layers ──────────────────────────────────────────── */}
      <main style={{ flex: 1, padding: '28px 32px 48px', display: 'flex', flexDirection: 'column', gap: 20 }}>

        {/* Layer 0: Admin (if open) */}
        {showAdmin && currentUser?.role === 'admin' && (
          <div className="analytics-layer" style={{ '--bubble-rgb': ACCENT, zIndex: 4 }}>
            <AdminDashboard currentUser={currentUser} />
          </div>
        )}

        {/* Layer 1: Hero Accuracy — sits highest visually */}
        <div className="analytics-layer" style={{ '--bubble-rgb': ACCENT, zIndex: 3, animationDelay: '0s' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16 }}>
            <div>
              <h3 style={{ margin: '0 0 4px', fontSize: '0.78rem', fontWeight: 700, color: '#9ca3af', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                Model Calibration
              </h3>
              {accuracy != null ? (
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                  <span style={{ fontSize: '3rem', fontWeight: 800, color: '#1c1c2e', lineHeight: 1 }}>
                    {Math.round(accuracy * 100)}%
                  </span>
                  <span style={{ fontSize: '0.85rem', color: '#6b7280' }}>Composite Accuracy</span>
                </div>
              ) : (
                <p style={{ color: '#9ca3af', margin: 0, fontSize: '0.9rem' }}>
                  {analyticsData?.status === 'no_data'
                    ? 'Awaiting feedback data — use the thumbs in Chat to calibrate.'
                    : 'Loading...'}
                </p>
              )}
            </div>

            {/* Mini emotion radar — top 4 emotions as color bars */}
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {topEmotions.slice(0, 4).map(([emo, stats]) => {
                const rgb = emotionRgb(emo);
                const pct = Math.round((stats.precision || 0) * 100);
                return (
                  <div key={emo} style={{
                    display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4,
                    minWidth: 56,
                  }}>
                    <div style={{
                      width: 10, height: Math.max(16, pct * 0.8),
                      borderRadius: 99,
                      background: `rgb(${rgb})`,
                      boxShadow: `0 2px 8px rgba(${rgb}, 0.35)`,
                      transition: 'height 0.6s cubic-bezier(0.34,1.2,0.64,1)',
                    }} />
                    <span style={{ fontSize: '0.65rem', color: '#9ca3af', textTransform: 'capitalize', textAlign: 'center' }}>
                      {emo}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Layer 2: Two charts side-by-side */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>

          {/* Doughnut */}
          <div className="analytics-layer" style={{ '--bubble-rgb': ACCENT, zIndex: 2, animationDelay: '0.06s' }}>
            <h4 style={{ margin: '0 0 16px', fontSize: '0.82rem', fontWeight: 700, color: '#374151', textTransform: 'uppercase', letterSpacing: '0.07em' }}>
              Emotion Distribution
            </h4>
            <div style={{ height: 220 }}>
              {topEmotions.length > 0
                ? <Doughnut data={doughnutData} options={doughnutOpts} />
                : <p style={{ color: '#9ca3af', margin: 0, fontSize: '0.85rem' }}>No data yet.</p>}
            </div>
          </div>

          {/* Line chart */}
          <div className="analytics-layer" style={{ '--bubble-rgb': ACCENT, zIndex: 2, animationDelay: '0.10s' }}>
            <h4 style={{ margin: '0 0 16px', fontSize: '0.82rem', fontWeight: 700, color: '#374151', textTransform: 'uppercase', letterSpacing: '0.07em' }}>
              Emotional Intensity — Conversation Arc
            </h4>
            <div style={{ height: 220 }}>
              {analyzed.length > 1
                ? <Line data={lineData} options={lineOpts} />
                : <p style={{ color: '#9ca3af', margin: 0, fontSize: '0.85rem' }}>Send more messages to see the arc.</p>}
            </div>
          </div>
        </div>

        {/* Layer 3: Per-emotion breakdown cards */}
        {topEmotions.length > 0 && (
          <div className="analytics-layer" style={{ '--bubble-rgb': ACCENT, zIndex: 1, animationDelay: '0.16s' }}>
            <h4 style={{ margin: '0 0 18px', fontSize: '0.82rem', fontWeight: 700, color: '#374151', textTransform: 'uppercase', letterSpacing: '0.07em' }}>
              Per-Emotion Precision & Recall
            </h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 14 }}>
              {topEmotions.map(([emo, stats]) => {
                const rgb = emotionRgb(emo);
                return (
                  <div key={emo} style={{
                    padding: '14px 16px',
                    background: `rgba(${rgb}, 0.06)`,
                    border: `1px solid rgba(${rgb}, 0.18)`,
                    borderRadius: 14,
                    boxShadow: `inset 0 1px 0 rgba(255,255,255,0.9)`,
                  }}>
                    <div style={{
                      fontSize: '0.8rem', fontWeight: 700, marginBottom: 10,
                      color: `rgb(${rgb})`, mixBlendMode: 'color-burn',
                      textTransform: 'capitalize', letterSpacing: '0.04em',
                    }}>
                      {emo}
                      <span style={{ float: 'right', fontWeight: 400, opacity: 0.65, fontSize: '0.72rem' }}>
                        {stats.samples} samples
                      </span>
                    </div>

                    {/* Precision */}
                    <div style={{ marginBottom: 8 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: '#6b7280', marginBottom: 4 }}>
                        <span>Precision</span>
                        <span style={{ fontWeight: 600 }}>{Math.round((stats.precision || 0) * 100)}%</span>
                      </div>
                      <div className="stat-bar-track" style={{ '--bubble-rgb': rgb }}>
                        <div className="stat-bar-fill" style={{ width: `${(stats.precision || 0) * 100}%`, background: `rgb(${rgb})` }} />
                      </div>
                    </div>

                    {/* Recall */}
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: '#6b7280', marginBottom: 4 }}>
                        <span>Recall</span>
                        <span style={{ fontWeight: 600 }}>{Math.round((stats.recall || 0) * 100)}%</span>
                      </div>
                      <div className="stat-bar-track" style={{ '--bubble-rgb': rgb }}>
                        <div className="stat-bar-fill" style={{ width: `${(stats.recall || 0) * 100}%`, background: `rgba(${rgb}, 0.7)` }} />
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Empty state */}
        {topEmotions.length === 0 && !analyticsData && (
          <div className="analytics-layer" style={{ '--bubble-rgb': ACCENT, textAlign: 'center', padding: '48px 24px' }}>
            <p style={{ color: '#9ca3af', margin: 0 }}>
              Click <strong>Generate Insights</strong> to load analytics data.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}

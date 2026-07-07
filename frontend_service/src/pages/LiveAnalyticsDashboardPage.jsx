import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  Chart as ChartJS,
  ArcElement, Tooltip, Legend,
  CategoryScale, LinearScale,
  BarElement, PointElement, LineElement, Filler,
} from 'chart.js';
import { Line, Doughnut } from 'react-chartjs-2';
import { EmotionPalette } from '../components/EmotionPalette';

ChartJS.register(
  ArcElement, Tooltip, Legend,
  CategoryScale, LinearScale,
  BarElement, PointElement, LineElement, Filler,
);

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';
const REFRESH_INTERVAL = 30_000;

function emotionRgb(emo) {
  return EmotionPalette[emo?.toLowerCase()] || '150,150,150';
}

function timeAgo(ts) {
  const diff = Date.now() / 1000 - ts;
  if (diff < 60)   return `${Math.round(diff)}s ago`;
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)}h ago`;
  return `${Math.round(diff / 86400)}d ago`;
}

// ── KPI Card ─────────────────────────────────────────────────────────────────

function KpiCard({ label, value, sub, color = '0,119,255' }) {
  return (
    <div style={{
      flex: 1, minWidth: 140,
      padding: '18px 20px',
      background: `rgba(${color},0.06)`,
      border: `1.5px solid rgba(${color},0.18)`,
      borderRadius: 16,
      boxShadow: `inset 0 1px 0 rgba(var(--ig-surf-rgb),0.9), 0 2px 12px rgba(${color},0.06)`,
    }}>
      <div style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--ig-txt3)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 6 }}>
        {label}
      </div>
      <div style={{ fontSize: '2rem', fontWeight: 800, color: `rgb(${color})`, lineHeight: 1, marginBottom: 4 }}>
        {value}
      </div>
      {sub && <div style={{ fontSize: '0.74rem', color: 'var(--ig-txt3)' }}>{sub}</div>}
    </div>
  );
}

// ── Emotion row bar ───────────────────────────────────────────────────────────

function EmotionBar({ emotion, count, total }) {
  const pct = total > 0 ? (count / total) * 100 : 0;
  const rgb = emotionRgb(emotion);
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
      <div style={{ width: 90, fontSize: '0.78rem', color: 'var(--ig-txt2)', textTransform: 'capitalize', textAlign: 'right', flexShrink: 0 }}>
        {emotion}
      </div>
      <div style={{ flex: 1, height: 10, borderRadius: 99, background: 'rgba(var(--ig-ink-rgb),0.08)', overflow: 'hidden' }}>
        <div style={{
          width: `${pct}%`, height: '100%', borderRadius: 99,
          background: `rgba(${rgb},0.75)`,
          boxShadow: `0 0 8px rgba(${rgb},0.35)`,
          transition: 'width 0.6s cubic-bezier(0.34,1.2,0.64,1)',
        }} />
      </div>
      <div style={{ width: 46, fontSize: '0.72rem', color: 'var(--ig-txt3)', textAlign: 'right', flexShrink: 0 }}>
        {count} <span style={{ opacity: 0.6 }}>({pct.toFixed(0)}%)</span>
      </div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function LiveAnalyticsDashboardPage({ currentUser, onBack }) {
  const [analyses, setAnalyses]     = useState([]);
  const [loading, setLoading]       = useState(true);
  const [lastRefresh, setLastRefresh] = useState(null);
  const [filterEmo, setFilterEmo]   = useState(null);

  // chart.js canvases can't resolve CSS variables — compute literals per theme
  const igTheme = (() => {
    try { return JSON.parse(localStorage.getItem('ig_settings') || '{}').theme === 'dark' ? 'dark' : 'light'; }
    catch { return 'light'; }
  })();
  const dark      = igTheme === 'dark';
  const chartTxt  = dark ? '#e9ebf2' : '#374151';
  const chartTick = dark ? 'rgba(230,232,240,0.55)' : '#9ca3af';
  const chartGrid = dark ? 'rgba(230,232,240,0.10)' : 'rgba(0,0,0,0.04)';

  const fetchData = useCallback(async () => {
    try {
      const res = await axios.get(`${API_BASE}/admin/recent-analyses?limit=500`);
      setAnalyses(res.data || []);
      setLastRefresh(new Date());
    } catch (err) {
      console.error('Analytics fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const timer = setInterval(fetchData, REFRESH_INTERVAL);
    return () => clearInterval(timer);
  }, [fetchData]);


  const withEmotion = analyses.filter(a => a.dominant && a.dominant !== '—');
  const withConf    = analyses.filter(a => a.confidence != null);

  const emotionCounts = {};
  withEmotion.forEach(a => {
    emotionCounts[a.dominant] = (emotionCounts[a.dominant] || 0) + 1;
  });
  const sortedEmotions = Object.entries(emotionCounts)
    .sort((a, b) => b[1] - a[1]);

  const avgConf = withConf.length
    ? withConf.reduce((s, a) => s + a.confidence, 0) / withConf.length
    : null;

  const topEmotion = sortedEmotions[0]?.[0] || '—';

  const uniqueConvs = new Set(analyses.map(a => a.conversation_id)).size;

  const timeline = [...withEmotion]
    .sort((a, b) => a.timestamp - b.timestamp)
    .slice(-40);

  const timelineData = {
    labels: timeline.map((_, i) => `#${i + 1}`),
    datasets: [{
      label: 'Confidence %',
      data: timeline.map(a => a.confidence ?? 0),
      borderColor: 'rgba(0,119,255,0.8)',
      backgroundColor: 'rgba(0,119,255,0.07)',
      fill: true, tension: 0.35,
      pointBackgroundColor: timeline.map(a => `rgba(${emotionRgb(a.dominant)},0.85)`),
      pointRadius: 5, pointHoverRadius: 7,
    }],
  };
  const timelineOpts = {
    responsive: true, maintainAspectRatio: false,
    scales: {
      y: { min: 0, max: 100, ticks: { color: chartTick, font: { size: 10 } }, grid: { color: chartGrid } },
      x: { ticks: { color: chartTick, font: { size: 9 }, maxTicksLimit: 10 }, grid: { display: false } },
    },
    plugins: { legend: { display: false }, tooltip: {
      callbacks: {
        afterLabel: (ctx) => `Emotion: ${timeline[ctx.dataIndex]?.dominant || ''}`,
      },
    }},
  };

  const top8 = sortedEmotions.slice(0, 8);
  const doughnutData = {
    labels: top8.map(([e]) => e),
    datasets: [{
      data: top8.map(([, c]) => c),
      backgroundColor: top8.map(([e]) => `rgba(${emotionRgb(e)},0.72)`),
      borderColor: top8.map(([e]) => `rgba(${emotionRgb(e)},0.95)`),
      borderWidth: 2,
    }],
  };
  const doughnutOpts = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { position: 'right', labels: { color: chartTxt, font: { size: 11 }, boxWidth: 12, padding: 10 } } },
    cutout: '60%',
  };

  const feed = [...analyses]
    .filter(a => !filterEmo || a.dominant === filterEmo)
    .sort((a, b) => b.timestamp - a.timestamp)
    .slice(0, 60);


  const layer = {
    background: 'rgba(var(--ig-surf-rgb),0.70)',
    backdropFilter: 'blur(24px) saturate(180%)',
    WebkitBackdropFilter: 'blur(24px) saturate(180%)',
    border: '1px solid rgba(var(--ig-ink-rgb),0.08)',
    borderRadius: 18,
    boxShadow: '0 4px 24px rgba(0,0,0,0.06), inset 0 1px 0 rgba(var(--ig-surf-rgb),0.95)',
    padding: '20px 24px',
  };

  return (
    <div data-ig-theme={igTheme} className="la-scroll" style={{
      height: '100vh', overflowY: 'auto',
      scrollbarWidth: 'thin', scrollbarColor: 'rgba(var(--ig-ink-rgb),0.28) transparent',
      background: dark
        ? 'linear-gradient(135deg,#0a0c12 0%,#10121c 60%,#150f18 100%)'
        : 'linear-gradient(135deg,#f0f4ff 0%,#faf8ff 60%,#fff0f8 100%)',
      color: 'var(--ig-txt)',
      fontFamily: "'Inter', -apple-system, sans-serif",
    }}>
      <style>{`
        .la-scroll::-webkit-scrollbar         { width: 8px; }
        .la-scroll::-webkit-scrollbar-track   { background: transparent; }
        .la-scroll::-webkit-scrollbar-thumb   { background: rgba(var(--ig-ink-rgb),0.18); border-radius: 99px; }
        .la-scroll::-webkit-scrollbar-thumb:hover { background: rgba(var(--ig-ink-rgb),0.32); }
      `}</style>

      <header style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '16px 32px',
        position: 'sticky', top: 0, zIndex: 10,
        background: 'rgba(var(--ig-surf-rgb),0.80)',
        backdropFilter: 'blur(24px) saturate(180%)',
        WebkitBackdropFilter: 'blur(24px) saturate(180%)',
        borderBottom: '1px solid rgba(var(--ig-ink-rgb),0.10)',
        boxShadow: '0 1px 12px rgba(0,0,0,0.05)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <button onClick={onBack} style={{
            background: 'rgba(var(--ig-ink-rgb),0.06)', border: 'none', borderRadius: 10,
            width: 36, height: 36, cursor: 'pointer', fontSize: '1rem', color: 'var(--ig-txt)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>←</button>
          <div>
            <h1 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 800, color: 'var(--ig-txt)' }}>
              Live Analytics
            </h1>
            <p style={{ margin: 0, fontSize: '0.74rem', color: 'var(--ig-txt3)' }}>
              {loading ? 'Loading…' : `${analyses.length} messages · refreshes every 30s`}
              {lastRefresh && ` · updated ${timeAgo(lastRefresh.getTime() / 1000)}`}
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 8 }}>
          {filterEmo && (
            <button onClick={() => setFilterEmo(null)} style={{
              padding: '6px 12px', borderRadius: 8, border: '1px solid var(--ig-bord2)',
              background: 'rgba(255,80,80,0.08)', color: '#e05050',
              cursor: 'pointer', fontSize: '0.76rem', fontWeight: 600,
            }}>
              ✕ {filterEmo}
            </button>
          )}
          <button onClick={fetchData} style={{
            padding: '8px 18px', borderRadius: 10, border: 'none',
            background: 'linear-gradient(135deg,#0077ff,#7000ff)',
            color: '#fff', fontWeight: 700, fontSize: '0.82rem', cursor: 'pointer',
          }}>
            ↻ Refresh
          </button>
        </div>
      </header>

      {loading ? (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '50vh', color: 'var(--ig-txt3)' }}>
          Loading pipeline data…
        </div>
      ) : (
        <main style={{ padding: '24px 32px 48px', display: 'flex', flexDirection: 'column', gap: 20 }}>

          <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
            <KpiCard label="Messages Analyzed" value={withEmotion.length.toLocaleString()} sub={`${analyses.length} total in pipeline`} color="0,119,255" />
            <KpiCard label="Avg Confidence" value={avgConf != null ? `${avgConf.toFixed(1)}%` : '—'} sub="meta-learner output" color="112,0,255" />
            <KpiCard label="Top Emotion" value={topEmotion} sub={`${emotionCounts[topEmotion] || 0} occurrences`} color={emotionRgb(topEmotion)} />
            <KpiCard label="Conversations" value={uniqueConvs} sub="unique threads analyzed" color="0,180,100" />
            <KpiCard label="Distinct Emotions" value={sortedEmotions.length} sub={`out of 28 GoEmotions`} color="255,140,0" />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 20 }}>

            <div style={layer}>
              <h4 style={{ margin: '0 0 16px', fontSize: '0.78rem', fontWeight: 700, color: 'var(--ig-txt2)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>
                Confidence Over Time — Last {timeline.length} Messages
              </h4>
              <div style={{ height: 210 }}>
                {timeline.length > 1
                  ? <Line data={timelineData} options={timelineOpts} />
                  : <p style={{ color: 'var(--ig-txt3)', margin: 0, fontSize: '0.85rem' }}>Need more messages to draw timeline.</p>}
              </div>
            </div>

            <div style={layer}>
              <h4 style={{ margin: '0 0 16px', fontSize: '0.78rem', fontWeight: 700, color: 'var(--ig-txt2)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>
                Emotion Distribution
              </h4>
              <div style={{ height: 210 }}>
                {top8.length > 0
                  ? <Doughnut data={doughnutData} options={doughnutOpts} />
                  : <p style={{ color: 'var(--ig-txt3)', margin: 0 }}>No data yet.</p>}
              </div>
            </div>
          </div>

          <div style={layer}>
            <h4 style={{ margin: '0 0 16px', fontSize: '0.78rem', fontWeight: 700, color: 'var(--ig-txt2)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>
              All Emotions — Click to Filter Feed
            </h4>
            <div style={{ columns: 2, gap: 32 }}>
              {sortedEmotions.map(([emo, count]) => (
                <div
                  key={emo}
                  onClick={() => setFilterEmo(filterEmo === emo ? null : emo)}
                  style={{ cursor: 'pointer', breakInside: 'avoid',
                    opacity: filterEmo && filterEmo !== emo ? 0.4 : 1,
                    transition: 'opacity 0.15s' }}
                >
                  <EmotionBar emotion={emo} count={count} total={withEmotion.length} />
                </div>
              ))}
            </div>
          </div>

          <div style={layer}>
            <h4 style={{ margin: '0 0 16px', fontSize: '0.78rem', fontWeight: 700, color: 'var(--ig-txt2)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>
              Recent Analyses {filterEmo && <span style={{ color: `rgb(${emotionRgb(filterEmo)})` }}>· {filterEmo}</span>}
              <span style={{ float: 'right', fontWeight: 400, color: 'var(--ig-txt3)', textTransform: 'none', letterSpacing: 0 }}>
                {feed.length} messages
              </span>
            </h4>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {feed.map(item => {
                const rgb = emotionRgb(item.dominant);
                return (
                  <div key={item.message_id} style={{
                    display: 'flex', alignItems: 'center', gap: 12,
                    padding: '10px 14px',
                    background: `rgba(${rgb},0.04)`,
                    border: `1px solid rgba(${rgb},0.14)`,
                    borderRadius: 12,
                    transition: 'background 0.12s',
                  }}
                    onMouseOver={e => e.currentTarget.style.background = `rgba(${rgb},0.09)`}
                    onMouseOut={e => e.currentTarget.style.background = `rgba(${rgb},0.04)`}
                  >
                    <div style={{
                      padding: '3px 9px', borderRadius: 99,
                      background: `rgba(${rgb},0.15)`,
                      color: `rgb(${rgb})`,
                      fontSize: '0.70rem', fontWeight: 700,
                      textTransform: 'capitalize', flexShrink: 0, width: 90, textAlign: 'center',
                    }}>
                      {item.dominant || '—'}
                    </div>

                    <div style={{ flex: 1, fontSize: '0.82rem', color: 'var(--ig-txt)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {item.text || '(no text)'}
                    </div>

                    <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexShrink: 0 }}>
                      {item.confidence != null && (
                        <div style={{ fontSize: '0.74rem', fontWeight: 700, color: item.confidence >= 70 ? '#16a34a' : item.confidence >= 45 ? '#d97706' : '#dc2626' }}>
                          {item.confidence.toFixed(0)}%
                        </div>
                      )}
                      <div style={{ fontSize: '0.70rem', color: 'var(--ig-txt3)' }}>
                        {item.display_name}
                      </div>
                      <div style={{ fontSize: '0.70rem', color: 'var(--ig-txt3)', opacity: 0.7 }}>
                        {timeAgo(item.timestamp)}
                      </div>
                    </div>
                  </div>
                );
              })}

              {feed.length === 0 && (
                <p style={{ color: 'var(--ig-txt3)', margin: 0, textAlign: 'center', padding: '32px 0', fontSize: '0.85rem' }}>
                  No messages found{filterEmo ? ` for "${filterEmo}"` : ''}.
                </p>
              )}
            </div>
          </div>

        </main>
      )}
    </div>
  );
}

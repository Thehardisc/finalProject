import AdminDashboard from '../components/AdminDashboard';

export default function AdminDashboardPage({ currentUser, onBack }) {
  const igTheme = (() => {
    try { return JSON.parse(localStorage.getItem('ig_settings') || '{}').theme === 'dark' ? 'dark' : 'light'; }
    catch { return 'light'; }
  })();

  return (
    <div data-ig-theme={igTheme} className="admin-scroll" style={{ height: '100vh', overflowY: 'auto', display: 'flex', flexDirection: 'column', background: 'var(--ig-bg)', color: 'var(--ig-txt)', scrollbarWidth: 'thin', scrollbarColor: 'rgba(var(--ig-ink-rgb),0.28) transparent' }}>
      <style>{`
        .admin-scroll::-webkit-scrollbar         { width: 8px; }
        .admin-scroll::-webkit-scrollbar-track   { background: transparent; }
        .admin-scroll::-webkit-scrollbar-thumb   { background: rgba(var(--ig-ink-rgb),0.18); border-radius: 99px; }
        .admin-scroll::-webkit-scrollbar-thumb:hover { background: rgba(var(--ig-ink-rgb),0.32); }
      `}</style>
      <header style={{
        display: 'flex', alignItems: 'center', gap: 12,
        padding: '18px 32px',
        position: 'sticky', top: 0, zIndex: 10,
        background: 'rgba(var(--ig-surf-rgb),0.80)',
        backdropFilter: 'blur(24px) saturate(180%)',
        WebkitBackdropFilter: 'blur(24px) saturate(180%)',
        borderBottom: '1px solid rgba(var(--ig-ink-rgb),0.10)',
        boxShadow: '0 2px 16px rgba(0,0,0,0.05)',
      }}>
        <button onClick={onBack} style={{
          background: 'rgba(var(--ig-ink-rgb),0.06)', border: 'none', borderRadius: 10,
          width: 36, height: 36, cursor: 'pointer',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '1rem', color: 'var(--ig-txt)',
        }}>←</button>
        <div>
          <h1 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 800, color: 'var(--ig-txt)' }}>
            Admin
          </h1>
          <p style={{ margin: 0, fontSize: '0.78rem', color: 'var(--ig-txt3)' }}>
            Accounts, roles, and access
          </p>
        </div>
      </header>

      <main style={{ flex: 1, padding: '24px 32px 48px', width: '100%', maxWidth: 1120, margin: '0 auto', boxSizing: 'border-box' }}>
        <AdminDashboard currentUser={currentUser} />
      </main>
    </div>
  );
}

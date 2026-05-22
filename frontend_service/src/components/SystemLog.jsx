import React, { useState, useEffect, useRef } from 'react';

const SystemLog = ({ logs }) => {
    const bottomRef = useRef(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [logs]);

    return (
        <div style={{
            display: 'flex', flexDirection: 'column', height: '100%',
            fontFamily: '"Fira Code", monospace', fontSize: '0.75rem',
            background: '#09090b', color: '#a1a1aa',
            borderLeft: '1px solid #27272a'
        }}>
            <div style={{
                padding: '1rem', borderBottom: '1px solid #27272a',
                display: 'flex', alignItems: 'center', gap: '0.5rem',
                background: 'rgba(9, 9, 11, 0.5)', backdropFilter: 'blur(10px)'
            }}>
                <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#22d3ee', boxShadow: '0 0 10px #22d3ee' }}></div>
                <span style={{ fontWeight: 'bold', color: 'white', letterSpacing: '0.05em' }}>SYSTEM_LINK // LOGS</span>
            </div>

            <div style={{ flex: 1, overflowY: 'auto', padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {logs.length === 0 && <span style={{ opacity: 0.3, fontStyle: 'italic' }}>System initialization... waiting for events.</span>}

                {logs.map((log, i) => (
                    <div key={i} style={{ opacity: 0.8, display: 'flex', gap: '0.5rem', animation: 'fadeIn 0.3s ease-in-out' }}>
                        <span style={{ color: '#52525b' }}>[{new Date(log.timestamp).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })}]</span>
                        <span style={{ color: log.type === 'error' ? '#ef4444' : (log.type === 'warn' ? '#fbbf24' : '#22d3ee') }}>
                            {log.module} ::
                        </span>
                        <span>{log.message}</span>
                    </div>
                ))}
                <div ref={bottomRef} />
            </div>

            <style>{`
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(5px); }
                    to { opacity: 0.8; transform: translateY(0); }
                }
            `}</style>
        </div>
    );
};

export default SystemLog;

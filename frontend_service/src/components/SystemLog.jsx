import React, { useEffect, useRef } from 'react';

const SystemLog = ({ logs }) => {
    const bottomRef = useRef(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [logs]);

    return (
        <div className="flex flex-col h-full font-mono text-xs bg-zinc-950 text-zinc-400 border-l border-zinc-800">
            <div className="p-4 border-b border-zinc-800 flex items-center gap-2 bg-zinc-950/50 backdrop-blur-[10px]">
                <div className="w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_10px_#22d3ee]" />
                <span className="font-bold text-white tracking-[0.05em]">SYSTEM_LINK // LOGS</span>
            </div>

            <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-2">
                {logs.length === 0 && (
                    <span className="opacity-30 italic">System initialization... waiting for events.</span>
                )}

                {logs.map((log, i) => (
                    <div key={i} className="opacity-80 flex gap-2">
                        <span className="text-zinc-600">
                            [{new Date(log.timestamp).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })}]
                        </span>
                        <span className={log.type === 'error' ? 'text-red-500' : log.type === 'warn' ? 'text-amber-400' : 'text-cyan-400'}>
                            {log.module} ::
                        </span>
                        <span>{log.message}</span>
                    </div>
                ))}
                <div ref={bottomRef} />
            </div>
        </div>
    );
};

export default SystemLog;

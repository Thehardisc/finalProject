import React, { useEffect } from 'react';

const Modal = ({ isOpen, onClose, children, title }) => {
    if (!isOpen) return null;

    return (
        <div style={{
            position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
            background: 'rgba(0, 0, 0, 0.7)', backdropFilter: 'blur(5px)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            zIndex: 1000
        }} onClick={onClose}>
            <div style={{
                background: '#09090b', // Zinc-950
                border: '1px solid #27272a', // Zinc-800
                borderRadius: '1rem',
                padding: '2rem',
                width: '90%', maxWidth: '600px',
                boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
                position: 'relative'
            }} onClick={e => e.stopPropagation()}>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                    <h2 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 'bold', background: 'linear-gradient(to right, #818cf8, #22d3ee)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>{title}</h2>
                    <button onClick={onClose} style={{ background: 'transparent', border: 'none', color: '#71717a', fontSize: '1.5rem', cursor: 'pointer' }}>&times;</button>
                </div>

                <div style={{ color: 'white' }}>
                    {children}
                </div>
            </div>
        </div>
    );
};

export default Modal;

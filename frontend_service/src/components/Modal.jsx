import React from 'react';

const Modal = ({ isOpen, onClose, children, title }) => {
    if (!isOpen) return null;

    return (
        <div
            className="fixed inset-0 z-[1000] flex items-center justify-center bg-black/70 backdrop-blur-[5px]"
            onClick={onClose}
        >
            <div
                className="relative w-[90%] max-w-[600px] rounded-2xl border border-zinc-800 bg-zinc-950 p-8 shadow-[0_25px_50px_-12px_rgba(0,0,0,0.5)]"
                onClick={e => e.stopPropagation()}
            >
                <div className="mb-6 flex items-center justify-between">
                    <h2 className="m-0 bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-2xl font-bold text-transparent">
                        {title}
                    </h2>
                    <button
                        onClick={onClose}
                        className="cursor-pointer border-none bg-transparent text-2xl text-zinc-500"
                    >
                        &times;
                    </button>
                </div>

                <div className="text-white">{children}</div>
            </div>
        </div>
    );
};

export default Modal;

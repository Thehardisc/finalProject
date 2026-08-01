import { useState, useEffect } from 'react';
import { EmotionPalette } from './EmotionPalette';

const MOCK_MESSAGES = [
    { id: 'm1', text: "Hey! Did you get the email about the project restructuring?", isOwn: true, emotion: "curiosity", conf: 0.8 },
    { id: 'm2', text: "Yeah, I just read it. I can't believe they're cutting the timeline by two weeks.", isOwn: false, emotion: "disapproval", conf: 0.9 },
    { id: 'm3', text: "It's completely ridiculous. We're already working overtime.", isOwn: true, emotion: "annoyance", conf: 0.95 },
    { id: 'm4', text: "I know... I'm feeling totally overwhelmed.", isOwn: false, emotion: "fear", conf: 0.85 },
    { id: 'm5', text: "We'll get through it. Let's just focus on the MVP first.", isOwn: true, emotion: "optimism", conf: 0.75 },
    { id: 'm6', text: "You're right. Thanks for being the calm one here.", isOwn: false, emotion: "gratitude", conf: 0.88 },
    { id: 'm7', text: "Always. I'll set up a quick sync for tomorrow morning.", isOwn: true, emotion: "caring", conf: 0.82 },
    { id: 'm8', text: "Perfect. Have a good night!", isOwn: false, emotion: "relief", conf: 0.90 },
    { id: 'm9', text: "Wait, one more thing. They also want us to rewrite the auth service.", isOwn: true, emotion: "surprise", conf: 0.92 },
    { id: 'm10', text: "Are you serious right now?", isOwn: false, emotion: "anger", conf: 0.96 },
    { id: 'm11', text: "I wish I was joking. Check the slack thread.", isOwn: true, emotion: "sadness", conf: 0.78 },
    { id: 'm12', text: "I can't deal with this right now.", isOwn: false, emotion: "disgust", conf: 0.84 },
    { id: 'm13', text: "I'll handle the auth. Don't worry about it.", isOwn: true, emotion: "caring", conf: 0.89 },
    { id: 'm14', text: "Wow, really? You're a lifesaver.", isOwn: false, emotion: "admiration", conf: 0.94 },
    { id: 'm15', text: "I got you. Get some rest.", isOwn: true, emotion: "optimism", conf: 0.85 }
];

const SCROLL_MESSAGES = [...MOCK_MESSAGES, ...MOCK_MESSAGES.map(m => ({...m, id: m.id + '_2'})), ...MOCK_MESSAGES.map(m => ({...m, id: m.id + '_3'}))];

const INJECTED_CSS = `
@keyframes infiniteScroll {
  from { transform: translateY(0); }
  to { transform: translateY(-33.3333%); }
}

@keyframes analyzingPulse {
  0%, 100% { opacity: 0.3; }
  50% { opacity: 0.8; }
}

.demo-scroll-container {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 24px;
  width: 100%;
  animation: infiniteScroll 45s linear infinite;
}

.demo-bubble {
  max-width: 380px;
  padding: 14px 18px;
  font-size: 0.95rem;
  line-height: 1.5;
  color: rgba(255, 255, 255, 0.8);
  position: relative;
  transition: all 0.5s ease-in-out;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.05);
}

.demo-bubble.analyzed {
  background: var(--emo-bg);
  border: 1px solid var(--emo-border);
  box-shadow: 0 4px 20px var(--emo-shadow);
}

.demo-bubble.own {
  align-self: flex-end;
  border-bottom-right-radius: 4px;
}

.demo-bubble.other {
  align-self: flex-start;
  border-bottom-left-radius: 4px;
}

.demo-bubble-accent {
  position: absolute;
  top: 0; bottom: 0;
  width: 3px;
  background: var(--emo-accent);
  transition: opacity 0.5s ease-in-out;
  opacity: 0;
}

.demo-bubble.analyzed .demo-bubble-accent {
  opacity: 1;
}

.demo-bubble.own .demo-bubble-accent {
  right: 0;
  border-top-right-radius: 12px;
  border-bottom-right-radius: 4px;
}

.demo-bubble.other .demo-bubble-accent {
  left: 0;
  border-top-left-radius: 12px;
  border-bottom-left-radius: 4px;
}

.demo-analyzing-indicator {
  position: absolute;
  bottom: -18px;
  font-size: 0.65rem;
  color: rgba(255,255,255,0.4);
  animation: analyzingPulse 1.5s infinite;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  opacity: 1;
  transition: opacity 0.3s;
}

.demo-bubble.own .demo-analyzing-indicator { right: 4px; }
.demo-bubble.other .demo-analyzing-indicator { left: 4px; }

.demo-analyzed-tag {
  position: absolute;
  bottom: -18px;
  font-size: 0.65rem;
  font-weight: 600;
  color: var(--emo-accent);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  opacity: 0;
  transition: opacity 0.3s;
}

.demo-bubble.analyzed .demo-analyzing-indicator { opacity: 0; }
.demo-bubble.analyzed .demo-analyzed-tag { opacity: 1; }

.demo-bubble.own .demo-analyzed-tag { right: 4px; }
.demo-bubble.other .demo-analyzed-tag { left: 4px; }
`;

function DemoBubble({ msg, delay }) {
    const [isAnalyzed, setIsAnalyzed] = useState(false);

    useEffect(() => {
        const timer = setTimeout(() => setIsAnalyzed(true), delay);
        return () => clearTimeout(timer);
    }, [delay]);

    const rgb = EmotionPalette[msg.emotion] || '148,163,184';
    const style = {
        '--emo-bg': `rgba(${rgb}, 0.04)`,
        '--emo-border': `rgba(${rgb}, 0.15)`,
        '--emo-shadow': `rgba(${rgb}, 0.08)`,
        '--emo-accent': `rgb(${rgb})`,
    };

    return (
        <div className={`demo-bubble ${msg.isOwn ? 'own' : 'other'} ${isAnalyzed ? 'analyzed' : ''}`} style={style}>
            <div className="demo-bubble-accent" />
            <div>{msg.text}</div>
            {!isAnalyzed && <div className="demo-analyzing-indicator">Analyzing...</div>}
            <div className="demo-analyzed-tag">{msg.emotion} · {(msg.conf * 100).toFixed(0)}%</div>
        </div>
    );
}

export default function LiveDemoBackground() {
    return (
        <div
            className="absolute inset-0 overflow-hidden pointer-events-none z-0 opacity-65"
            style={{
                maskImage: 'linear-gradient(to bottom, transparent, black 15%, black 85%, transparent)',
                WebkitMaskImage: 'linear-gradient(to bottom, transparent, black 15%, black 85%, transparent)',
            }}
        >
            <style>{INJECTED_CSS}</style>
            <div className="flex w-full h-full">
                <div className="flex-1 overflow-hidden relative">
                    <div className="demo-scroll-container" style={{ transform: 'translateY(-10%)' }}>
                        {SCROLL_MESSAGES.map((msg, i) => (
                            <DemoBubble key={`c1-${msg.id}-${i}`} msg={msg} delay={2000 + i * 800} />
                        ))}
                    </div>
                </div>
                <div className="flex-1 overflow-hidden relative mt-[120px]">
                    <div className="demo-scroll-container" style={{ animationDuration: '55s' }}>
                        {SCROLL_MESSAGES.map((msg, i) => (
                            <DemoBubble key={`c2-${msg.id}-${i}`} msg={msg} delay={4000 + i * 900} />
                        ))}
                    </div>
                </div>
                <div className="flex-1 overflow-hidden relative -mt-[80px]">
                    <div className="demo-scroll-container" style={{ animationDuration: '50s' }}>
                        {SCROLL_MESSAGES.map((msg, i) => (
                            <DemoBubble key={`c3-${msg.id}-${i}`} msg={msg} delay={1000 + i * 750} />
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}


const PLUTCHIK_MAPPING = {
    "joy":         "joy",
    "admiration":  "trust",
    "fear":        "fear",
    "nervousness": "fear",
    "surprise":    "surprise",
    "sadness":     "sadness",
    "grief":       "sadness",
    "disgust":     "disgust",
    "anger":       "anger",
    "annoyance":   "anger",
    "curiosity":   "anticipation",
};

const PETALS = [
    { name: 'joy',          rotate: 0   },
    { name: 'trust',        rotate: 45  },
    { name: 'fear',         rotate: 90  },
    { name: 'surprise',     rotate: 135 },
    { name: 'sadness',      rotate: 180 },
    { name: 'disgust',      rotate: 225 },
    { name: 'anger',        rotate: 270 },
    { name: 'anticipation', rotate: 315 },
];

const PlutchikWheel = ({ dominantEmotion }) => {
    const core = PLUTCHIK_MAPPING[dominantEmotion?.toLowerCase()] || null;
    return (
        <div className="plutchik-container">
            <svg id="plutchik-svg" viewBox="0 0 200 200">
                <g transform="translate(100,100)">
                    {PETALS.map((petal, i) => (
                        <path
                            key={i}
                            d="M0,0 Q20,-40 0,-80 Q-20,-40 0,0"
                            fill={petal.name === core ? 'var(--accent-primary)' : 'rgba(var(--ig-ink-rgb),0.06)'}
                            stroke="var(--glass-border)"
                            transform={`rotate(${petal.rotate})`}
                        />
                    ))}
                </g>
                <circle cx="100" cy="100" r="10" fill="white" filter="blur(2px)" />
            </svg>
        </div>
    );
};

export default PlutchikWheel;

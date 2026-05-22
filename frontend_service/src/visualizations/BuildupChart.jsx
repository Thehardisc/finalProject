import React from 'react';
import { Line } from 'react-chartjs-2';
import { EMOTION_COLORS } from '../constants/emotions';

/**
 * BuildupChart
 * Line chart showing the emotional intensity progression across recent messages.
 */
const BuildupChart = ({ steps }) => {
    const labels      = steps.map(s => s.text.length > 20 ? '...' + s.text.slice(-18) : s.text);
    const dataPoints  = steps.map(s => s.scores?.length > 0 ? Math.max(...s.scores.map(x => x.score)) : 0);
    const pointColors = steps.map(s => EMOTION_COLORS[s.dominant?.toLowerCase()] || '#00f2ff');

    const data = {
        labels,
        datasets: [{
            label:            'Emotional Intensity',
            data:             dataPoints,
            borderColor:      '#00f2ff',
            backgroundColor:  'rgba(0, 242, 255, 0.1)',
            fill:             true,
            tension:          0.4,
            pointBackgroundColor: pointColors,
            pointRadius:      6
        }]
    };

    const options = {
        responsive:          true,
        maintainAspectRatio: false,
        scales: {
            y: { min: 0, max: 1, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
            x: { grid: { display: false }, ticks: { display: false } }
        },
        plugins: { legend: { display: false } }
    };

    return <Line data={data} options={options} />;
};

export default BuildupChart;

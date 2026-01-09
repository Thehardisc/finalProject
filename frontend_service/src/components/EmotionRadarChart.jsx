import React, { useMemo } from 'react';
import {
    Chart as ChartJS,
    RadialLinearScale,
    PointElement,
    LineElement,
    Filler,
    Tooltip,
    Legend,
} from 'chart.js';
import { Radar } from 'react-chartjs-2';

ChartJS.register(
    RadialLinearScale,
    PointElement,
    LineElement,
    Filler,
    Tooltip,
    Legend
);

const EmotionRadarChart = ({ emotionData }) => {
    // emotionData is expected to be a dictionary: {"joy": 0.8, "sadness": 0.1, ...}

    const chartData = useMemo(() => {
        if (!emotionData) return null;

        // Filter out internal/lexicon keys if we just want GoEmotions
        const labels = Object.keys(emotionData).filter(
            k => !k.startsWith('vader_') && !k.startsWith('emphasis_')
        ).sort();

        const dataPoints = labels.map(label => emotionData[label] || 0);

        return {
            labels,
            datasets: [
                {
                    label: 'Current Emotional State',
                    data: dataPoints,
                    backgroundColor: 'rgba(59, 130, 246, 0.2)',
                    borderColor: 'rgba(59, 130, 246, 1)',
                    borderWidth: 2,
                },
            ],
        };
    }, [emotionData]);

    if (!chartData) return <p>No data available</p>;

    const options = {
        scales: {
            r: {
                angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                grid: { color: 'rgba(255, 255, 255, 0.1)' },
                suggestedMin: 0,
                suggestedMax: 1,
                ticks: { backdropColor: 'transparent', color: '#94a3b8' }
            },
        },
        plugins: {
            legend: {
                labels: {
                    color: 'white'
                }
            }
        }
    };

    return <Radar data={chartData} options={options} />;
};

export default EmotionRadarChart;

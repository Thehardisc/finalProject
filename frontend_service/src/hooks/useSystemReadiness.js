import { useState, useEffect } from 'react';
import { systemAPI } from '../api/client';

/**
 * useSystemReadiness
 * Polls /health/status every 3 seconds until all systems are ready.
 * Returns gating state used by the loading screen.
 */
export function useSystemReadiness(currentUser) {
    const [systemReady,        setSystemReady]        = useState(false);
    const [gateVisible,        setGateVisible]        = useState(false);
    const [trainingInProgress, setTrainingInProgress] = useState(false);
    const [componentStatus,    setComponentStatus]    = useState({
        database: false, redis: false, meta_learner: false
    });

    const checkSystemStatus = async () => {
        try {
            const res  = await systemAPI.status();
            const data = res.data;
            if (data.components) setComponentStatus(data.components);
            if (data.ready) {
                setGateVisible(false);
                setTimeout(() => setSystemReady(true), 800);
                setTrainingInProgress(data.training_in_progress === true);
                return true;
            } else {
                setTrainingInProgress(true);
            }
        } catch (e) {
            console.warn('System status check failed, retrying...');
        }
        return false;
    };

    useEffect(() => {
        if (!currentUser) return;
        setGateVisible(true);
        setSystemReady(false);

        let pollInterval;
        const checkAndSchedule = async () => {
            const isReady = await checkSystemStatus();
            if (isReady) clearInterval(pollInterval);
        };

        checkAndSchedule();
        pollInterval = setInterval(checkAndSchedule, 3000);
        return () => clearInterval(pollInterval);
    }, [currentUser]);

    return { systemReady, gateVisible, trainingInProgress,
             componentStatus, setTrainingInProgress };
}

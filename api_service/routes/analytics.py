"""
api_service/routes/analytics.py — Model calibration analytics endpoint.
"""
import json
import time
from collections import Counter

from fastapi import APIRouter, HTTPException, Depends

from shared.utils.logger import get_logger
from shared.constants import EMOTION_LABELS
from api_service.db.pool import get_pool
from api_service.auth_utils import get_current_user

logger = get_logger("api_service")
router = APIRouter()


@router.get("/analytics/calibration", dependencies=[Depends(get_current_user)])
async def get_calibration_analytics():
    """Get model performance metrics derived from human-verified feedback."""
    pool = get_pool()
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT ground_truth_emotion, emotions_json "
                "FROM emotion_analysis WHERE is_verified = TRUE"
            )

        if not rows:
            return {"status": "no_data",
                    "message": "Provide more feedback to see calibration stats."}

        total_verified = len(rows)
        correct_count  = 0
        tp = Counter()
        fp = Counter()
        fn = Counter()
        confusion = {}

        for row in rows:
            actual = row['ground_truth_emotion']
            try:
                ems = json.loads(row['emotions_json'])
            except (json.JSONDecodeError, TypeError):
                continue
            predicted = ems.get("dominant_emotion", "Neutral")

            if actual not in confusion:
                confusion[actual] = Counter()
            confusion[actual][predicted] += 1

            if actual == predicted:
                correct_count += 1
                tp[actual] += 1
            else:
                fp[predicted] += 1
                fn[actual]    += 1

        actual_counts = Counter(r['ground_truth_emotion'] for r in rows)
        emotion_stats = {}
        for emo in EMOTION_LABELS:
            actual_count = actual_counts.get(emo, 0)
            if actual_count:
                precision = tp[emo] / (tp[emo] + fp[emo]) if (tp[emo] + fp[emo]) > 0 else 0
                recall    = tp[emo] / (tp[emo] + fn[emo]) if (tp[emo] + fn[emo]) > 0 else 0
                f1        = (2 * precision * recall / (precision + recall)
                             if (precision + recall) > 0 else 0)
                emotion_stats[emo] = {
                    "precision": round(precision, 4),
                    "recall":    round(recall,    4),
                    "f1":        round(f1,         4),
                    "samples":   actual_count
                }

        logger.log_stats("Model Calibration Report", {
            "Total Samples":    total_verified,
            "Overall Accuracy": f"{correct_count / total_verified:.2%}",
            "TP Total":         sum(tp.values()),
            "FP Total":         sum(fp.values()),
            "FN Total":         sum(fn.values())
        })

        return {
            "overall_accuracy":        round(correct_count / total_verified, 4),
            "total_verified_samples":  total_verified,
            "emotion_breakdown":       emotion_stats,
            "confusion_matrix":        confusion,
            "timestamp":               time.time()
        }

    except Exception as e:
        logger.error(f"Analytics DB Error: {e}")
        raise HTTPException(status_code=500, detail="Could not calculate analytics.")

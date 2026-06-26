from prometheus_client import Counter, Gauge, Histogram

messages_processed_total = Counter(
    "central_responder_messages_processed_total",
    "Messages aggregated and published to emotion_stream.",
)

aggregation_latency_ms = Histogram(
    "central_responder_aggregation_latency_ms",
    "Per-message aggregation latency (model arrivals → publish) in ms.",
    buckets=(5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000),
)

meta_predictions_total = Counter(
    "central_responder_meta_predictions_total",
    "Meta-learner inference outcomes.",
    labelnames=("outcome",),  # "success" | "fallback"
)

trainer_runs_total = Counter(
    "central_responder_trainer_runs_total",
    "Trainer cycles by outcome.",
    labelnames=("result",),  # "accepted" | "rejected" | "errored"
)

trainer_last_run_timestamp = Gauge(
    "central_responder_trainer_last_run_timestamp_seconds",
    "Unix timestamp of the most recently completed trainer cycle.",
)

trainer_last_accuracy = Gauge(
    "central_responder_trainer_last_test_accuracy",
    "Test accuracy of the most recently completed trainer cycle (regardless of gate).",
)

trainer_last_accepted_accuracy = Gauge(
    "central_responder_trainer_last_accepted_accuracy",
    "Test accuracy of the most recently DEPLOYED model (post-gate).",
)

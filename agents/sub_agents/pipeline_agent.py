from .base_agent import BaseAgent


class PipelineAgent(BaseAgent):
    name = "pipeline"
    description = (
        "Redis Stream message pipeline: ingestion → preprocessing → "
        "partial_analysis_stream (vader/bert/goe/context_engine) → "
        "emotion_stream → conversation_update_stream. "
        "Owns: ingestion_service, preprocessing_service, module_registry, stream topology."
    )
    key_files = [
        "ingestion_service/main.py",
        "preprocessing_service/main.py",
        "shared/module_registry.py",
        "shared/constants.py",
        "docker-compose.yml",
        "start.sh",
    ]

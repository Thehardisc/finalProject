from .base_agent import BaseAgent


class InfraAgent(BaseAgent):
    name = "infra"
    description = (
        "Infrastructure layer: Docker Compose, Redis (streams + state hashes), "
        "PostgreSQL (long-term storage), Qdrant (vector DB). "
        "Shared constants (FEATURE_DIM=116, ML_DIM=42, CDM_CTX_DIM=40, PRIOR_DIM=28, "
        "CONTEXT_DIM=74, N_CDM_STATES=15). "
        "Logging: LOG_LEVEL + LOG_FORMAT env vars, json-file driver 10m×3. "
        "Owns: docker-compose.yml, shared/constants.py, shared/utils/, .env layout, start.sh."
    )
    key_files = [
        "docker-compose.yml",
        "shared/constants.py",
        "shared/utils/logger.py",
        "shared/module_registry.py",
        "start.sh",
        "aggregation_service/main.py",
    ]

from .pipeline_agent import PipelineAgent
from .nlp_agent import NLPAgent
from .meta_learner_agent import MetaLearnerAgent
from .context_engine_agent import ContextEngineAgent
from .trajectory_agent import TrajectoryAgent
from .api_frontend_agent import ApiFrontendAgent
from .infra_agent import InfraAgent
from .aggregation_agent import AggregationAgent
from .persistence_agent import PersistenceAgent
from .llm_agent import LLMAgent

ALL_AGENTS = {
    "pipeline":       PipelineAgent,
    "nlp":            NLPAgent,
    "meta_learner":   MetaLearnerAgent,
    "context_engine": ContextEngineAgent,
    "trajectory":     TrajectoryAgent,
    "api_frontend":   ApiFrontendAgent,
    "infra":          InfraAgent,
    "aggregation":    AggregationAgent,
    "persistence":    PersistenceAgent,
    "llm":            LLMAgent,
}

"""
VOMC-QKV Analysis Pipeline

Analyzes how Q, K, V representations change under varying context lengths
using Variable-Order Markov Chains to detect forward planning in LLMs.
"""

from .state_extractor import QKVExtractor, QKVState, SequenceQKVProfile
from .context_sweep import ContextSweeper, ContextSweepResult, ContextSweepCollection
from .vomc import VOMCBuilder, VOMCAnalysis, VOMCStateSpace, sequences_from_trajectories
from .planning_detector import PlanningDetector, PlanningProfile, PlanningComparison
from .utils import set_seed, setup_logging
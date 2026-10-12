"""BugPredictor: software defect prediction with a genetic algorithm, evaluated without data leakage."""

from .data import BENCHMARK, Dataset, list_releases, load, load_promise
from .evaluation import cross_validate, leaky_cross_validate, stratified_folds, train_predict
from .ga import GeneticOptimizer
from .metrics import evaluate
from .models import MODELS

__version__ = "2.0.0"
__all__ = ["BENCHMARK", "Dataset", "GeneticOptimizer", "MODELS", "cross_validate", "evaluate",
           "leaky_cross_validate", "list_releases", "load", "load_promise", "stratified_folds", "train_predict"]

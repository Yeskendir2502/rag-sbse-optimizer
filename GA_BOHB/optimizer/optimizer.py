from typing import Any, Dict
import traceback

from tqdm import tqdm

from GA_BOHB.initial_configs import INITIAL_CONFIGS
from GA_BOHB.config_space import CONFIG_SPACE
from GA_BOHB.optimizer import evaluator, logger, utils

try:
    import optuna
    from optuna.samplers import TPESampler
except Exception:
    optuna = None


class ConfigOptimizer:
    def __init__(self, dataset: str = "fiqa", use_dummy: bool = True):
        self._trial_counter = 0
        self.dataset = dataset
        self.use_dummy = use_dummy

    def _next_trial_id(self):
        t = self._trial_counter
        self._trial_counter += 1
        return t

    def run_initial_configs(self, dataset="fiqa", use_dummy=True):
        for cfg in tqdm(INITIAL_CONFIGS, desc="Initial configs"):
            tid = self._next_trial_id()
            try:
                ndcg, latency = evaluator.evaluate_config(cfg, dataset, use_dummy=use_dummy)
            except Exception:
                traceback.print_exc()
                ndcg, latency = 0.0, float("inf")

            logger.save_result(cfg, ndcg, latency, tid)

    def run_random_search(self, trials=20, dataset="fiqa", use_dummy=True):
        for _ in tqdm(range(trials), desc="Random search"):
            cfg = utils.random_config()
            tid = self._next_trial_id()
            try:
                ndcg, latency = evaluator.evaluate_config(cfg, dataset, use_dummy=use_dummy)
            except Exception:
                traceback.print_exc()
                ndcg, latency = 0.0, float("inf")

            logger.save_result(cfg, ndcg, latency, tid)

    def bohb_objective(self, trial: Any):
        cfg = {}
        for k, choices in CONFIG_SPACE.items():
            cfg[k] = trial.suggest_categorical(k, choices)

        tid = self._next_trial_id()
        try:
            ndcg, latency = evaluator.evaluate_config(cfg, dataset=self.dataset, use_dummy=self.use_dummy)
        except Exception:
            traceback.print_exc()
            ndcg, latency = 0.0, float("inf")

        logger.save_result(cfg, ndcg, latency, tid)
        score = ndcg - 0.0005 * latency
        return score

    def run_bohb_optimize(self, trials=50, dataset: str | None = None, use_dummy: bool | None = None):
        if optuna is None:
            raise RuntimeError("Optuna is required for BOHB optimization. Install optuna first.")

        if dataset:
            self.dataset = dataset
        if use_dummy is not None:
            self.use_dummy = use_dummy

        sampler = TPESampler()
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(self.bohb_objective, n_trials=trials)
        return study

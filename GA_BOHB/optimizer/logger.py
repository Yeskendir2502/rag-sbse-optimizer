import json
import os


BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "BOHB_results.json")


def _ensure_results_file():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "w") as f:
            json.dump([], f, indent=2)


def save_result(config, ndcg, latency, trial_id):
    _ensure_results_file()
    entry = {
        "trial_id": trial_id,
        "config": config,
        "nDCG": ndcg,
        "latency": latency,
    }

    try:
        with open(RESULTS_PATH, "r") as f:
            data = json.load(f)
    except Exception:
        data = []

    data.append(entry)

    with open(RESULTS_PATH, "w") as f:
        json.dump(data, f, indent=2)

import random
from GA_BOHB.config_space import CONFIG_SPACE


def random_config():
    cfg = {}
    for k, choices in CONFIG_SPACE.items():
        cfg[k] = random.choice(choices)
    return cfg

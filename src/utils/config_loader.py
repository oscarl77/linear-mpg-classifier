import json
import os

def load_config(path="./config.json"):
    with open(path, "r") as f:
        return json.load(f)

def update_config(experiment_name, params, path="./config.json"):
    """Adds a new entry into the config file. In the context of this project
    it saves an experiment's weight vector and bias term.
    """
    config = load_config()

    if "SAVED_SVM_PARAMS" not in config:
        config["SAVED_SVM_PARAMS"] = {}

    config["SAVED_SVM_PARAMS"][experiment_name] = params
    with open(path, "w") as f:
        json.dump(config, f, indent=4)
        #f.flush()
        #os.fsync(f.fileno())
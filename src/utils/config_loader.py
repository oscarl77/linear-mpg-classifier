import json

def load_config(path="./config.json"):
    with open(path, "r") as f:
        return json.load(f)

def update_config(key, value, path="./config.json"):
    config = load_config()

    config[key] = value

    with open(path, "w") as f:
        json.dump(config, f, indent=4)
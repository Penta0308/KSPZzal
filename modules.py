import json
import hashlib


def config_get(key: str):
    with open("config.json", "r", encoding="UTF-8") as f:
        settings = json.load(f)
    try:
        return settings[key]
    except KeyError:
        config_update(key, None)
        return None


def config_update(key: str, val):
    with open("config.json", "r", encoding="UTF-8") as f:
        settings = json.load(f)
    settings[key] = val
    with open("config.json", "w", encoding="UTF-8") as f:
        json.dump(settings, f, indent=4, ensure_ascii=False)


templatehash = config_get("templateimghash")


def hasmeaning(data) -> bool:
    return int(hashlib.blake2b(data).hexdigest(), base=16) not in templatehash

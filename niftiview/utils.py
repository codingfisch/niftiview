import importlib.resources
from json import dump, load
from pathlib import Path
DATA_PATH = str(importlib.resources.files('niftiview')) + '/data'


def save_json(dictionary, filepath):
    with open(filepath, 'w') as file:
        dump(dictionary, file, indent=4)


def load_json(filepath):
    with open(filepath, 'r') as file:
        dictionary = load(file)
    return dictionary


def get_filestem(filepath):
    return str(Path(filepath).name).split('.')[0]

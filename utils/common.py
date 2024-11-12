from pathlib import Path
import json

traits_definitions = {}

def load_traits_definitions():
    """
    Loads personality trait definitions from 'traits_definitions.json'.

    Returns:
        dict: A dictionary containing personality trait definitions.
    """
    current_file = Path(__file__).resolve()
    traits_path = current_file.parent.parent / 'experiment_resources' / 'traits_definitions.json'
    if not traits_path.exists():
        raise FileNotFoundError(f"File {traits_path} does not exist.")

    with traits_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def initialize():
    """
    Initializes the common module by loading trait definitions.
    """
    global traits_definitions
    traits_definitions = load_traits_definitions()
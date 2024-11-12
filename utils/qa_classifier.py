from utils import common
import logging
import os
import pandas as pd
from datetime import datetime
from experiment_resources.language_models import *
import random
import numpy as np
import json
import re

logger = logging.getLogger(__name__)

def run_analysis(df_answers='last', settings=None):
    """
    Runs the analysis on generated texts using the specified settings.

    Args:
        df_answers (str): The experiment folder name or 'last' for the latest.
        settings (dict): Analysis settings.

    Returns:
        None
    """
    if settings is not None:
        text_analysis_settings = settings
    else:
        raise FileNotFoundError(f"Analysis settings does not exist")
        return

    analyzed_files = []

    model = get_model_instance(text_analysis_settings['model_name'])
    temperature = text_analysis_settings['temperature']

    if df_answers == 'last':
        # Find the latest experiment folder
        results_dir = 'results'
        experiment_folders = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d)) and d.startswith('qa_text_generation_')]
        if not experiment_folders:
            print("No experiment folders found.")
            return
        latest_experiment_folder = max(experiment_folders)
        experiment_path = os.path.join(results_dir, latest_experiment_folder)
    elif isinstance(df_answers, str):
        # df_answers is the name of the experiment folder
        experiment_path = os.path.join('results', df_answers)
        if not os.path.exists(experiment_path):
            print(f"Experiment folder '{experiment_path}' does not exist.")
            return
    else:
        print("Invalid input for df_answers. It should be 'last' or the name of the experiment folder.")
        return
    
    # Collect all 'texts_' CSV files in the experiment folder
    answer_files = [os.path.join(experiment_path, f) for f in os.listdir(experiment_path) if f.startswith('texts_') and f.endswith('.csv')]
    if not answer_files:
        print(f"No appropriate CSV files found in the experiment folder '{experiment_path}'.")
        return

    print(answer_files)

    traits_definitions = common.traits_definitions

    for input_file in answer_files:
        try:
            df_texts = pd.read_csv(input_file)
            required_columns = {'model', 'temperature', 'trait', 'trait_score', 'question', 'answer'}
            if not required_columns.issubset(df_texts.columns):
                missing_cols = required_columns - set(df_texts.columns)
                raise ValueError(f"CSV file is missing columns: {', '.join(missing_cols)}")
        except Exception as e:
            print(f"Error loading texts from '{input_file}': {e}")
            continue

        df_analysis = analyze_texts(
            model,
            df_texts,
            traits_definitions['classification'],
            {'temperature': temperature}
        )
        # Save analysis results
        analysis_filename = f"{experiment_path}/analysis_{os.path.basename(input_file)}"
        df_analysis.to_csv(analysis_filename, index=False)
        analyzed_files.append(analysis_filename)
        logger.info(f"Analysis results saved: {analysis_filename}")

    
    # Output list of saved analysis file paths
    print("Analyzed files:")
    for file in analyzed_files:
        print(file)
     

def analyze_texts(model, df_texts, traits_definitions, experiment_settings):
    """
    Analyzes generated texts using a classifier.

    Args:
        model: Instance of the model used for analysis.
        df_texts (pd.DataFrame): DataFrame containing generated texts.
        traits_definitions (dict): Dictionary with personality trait definitions.
        experiment_settings (dict): Experiment settings.

    Returns:
        pd.DataFrame: DataFrame containing analysis results.
    """
    results = []
    temperature = experiment_settings['temperature']

    for index, row in df_texts.iterrows():
        trait_name = row['trait']
        trait_info = None
        # Find trait_key and trait_info by trait name
        for key, info in traits_definitions.items():
            if info['name'] == trait_name:
                trait_key = key
                trait_info = info
                break

        if not trait_info:
            logger.error(f"Trait info not found for trait: {trait_name}")
            continue

        analysis = detect_trait(
            model,
            trait_key,
            trait_info,
            row['question'],
            row['answer'],
            temperature
        )
        parsed_analysis = parsed_results(analysis)
        results.append({
            **row.to_dict(),
            **parsed_analysis,
            'analyzed_trait': trait_info['name']
        })
        logger.info(f"Analyzed text for trait {trait_info['name']}")

    df_analysis = pd.DataFrame(results)
    return df_analysis


def detect_trait(model, trait_key, trait_info, question, answer, temperature):
    """
    Detects a personality trait in the text.

    Args:
        model: Instance of the model used for analysis.
        trait_key (str): Key of the personality trait.
        trait_info (dict): Information about the personality trait.
        question (str): The question asked.
        answer (str): The answer provided.
        temperature (float): Model temperature.

    Returns:
        str: Model's response as a string.
    """
    system_prompt = f"""You will be provided with answers to questions. Detect the score of {trait_info['name']} for the author of the INPUT from the list [-2, -1, 0, 1, 2] or Nondistinguishable. Use INSTRUCTION.
TASK:
1. First, list CLUES (i.e., keywords, phrases, contextual information, semantic relations, semantic meaning, tones, references) that support the score determination of {trait_info['name']} of INPUT.
2. Second, deduce the diagnostic REASONING process from premises (i.e., clues, input) that supports the INPUT score determination (Limit the number of words to 130).
3. Third, based on clues, reasoning and input, determine the score of {trait_info['name']} for the author of INPUT from the list [-2, -1, 0, 1, 2] or Nondistinguishable.
4. Mark what made you choose this score as decision type: Explicit signs, Implicit signs, Intuition, Nondistinguishable.
5. Provide your output in JSON format with the keys: score, clues, reasoning, decision type.
PROVIDE ONLY JSON.

INSTRUCTION:
- Definition: {trait_info.get('definition', '')}
- High score of {trait_info['name']} (maximum 2): '{trait_info['high']}'
- Low score of {trait_info['name']} (minimum -2): '{trait_info['low']}'
- Explicit signs: The person mentions obvious facts that are connected with this trait score.
- Implicit signs: The person mentions facts that may imply them having this trait score.
- Intuition: My intuition tells that the person has this trait score.
- Nondistinguishable: I can't tell what trait score the person has.
- If the text does not contain substantial, significant, and convincing indicators of the trait score, then use Nondistinguishable.
- Choose something other than Nondistinguishable if you have a high degree of confidence in the answer.
"""

    user_prompt = f"""Question: {question} INPUT: {answer}"""
    response, _ = model.generate_response(system_prompt, user_prompt, temperature)
 
    return response.strip()

def parsed_results(raw_analysis):
    """
    Parses the model's response and extracts data, accounting for possible JSON formatting errors.

    Args:
        raw_analysis (str): Raw response from the model.

    Returns:
        dict: Dictionary with keys 'score', 'clues', 'reasoning', 'decision type'.
    """
    # Initialize default values
    analysis_data = {
        'score': None,
        'clues': None,
        'reasoning': None,
        'decision type': None
    }

    # Attempt to parse JSON directly
    required_keys = {'score', 'clues', 'reasoning', 'decision type'}
    try:
        analysis_data = json.loads(raw_analysis)
        # Check for required keys
        if not required_keys.issubset(analysis_data.keys()):
            raise ValueError("Parsed JSON is missing required keys.")
    except (json.JSONDecodeError, ValueError):
        # Try to fix JSON formatting
        pattern = r'\{.*?\}'
        matches = re.findall(pattern, raw_analysis, re.DOTALL)
        if matches:
            json_string = matches[0]
            # Replace erroneous double quotes
            corrected_json_string = json_string.replace('""', '"')
            try:
                analysis_data = json.loads(corrected_json_string)
                if not required_keys.issubset(analysis_data.keys()):
                    raise ValueError("Parsed JSON is missing required keys.")
            except (json.JSONDecodeError, ValueError):
                # Additional correction attempts
                corrected_raw_analysis = raw_analysis.replace('Explicit signs', '"Explicit signs"').replace('Implicit signs', '"Implicit signs"').replace('Intuition', '"Intuition"').replace('Nondistinguishable', '"Nondistinguishable"')
                try:
                    analysis_data = json.loads(corrected_raw_analysis)
                    if not required_keys.issubset(analysis_data.keys()):
                        raise ValueError("Parsed JSON is missing required keys.")
                except (json.JSONDecodeError, ValueError):
                    logger.error(f"Failed to parse analysis after multiple attempts: {raw_analysis}")
        else:
            logger.error(f"No JSON-like content found in analysis: {raw_analysis}")

    analysis_data['raw_response'] = raw_analysis
    return analysis_data
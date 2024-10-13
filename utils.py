# utils.py

import os
import time
import pandas as pd
import logging
# from dotenv import load_dotenv

# Load environment variables
# load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

def match_score(answer, scores_dict):
    """
    Matches the answer to a specific score based on the scores dictionary.

    Args:
        answer (str): The model's answer to a questionnaire question.
        scores_dict (dict): Dictionary mapping textual answers to numerical scores.

    Returns:
        int: Numerical score corresponding to the answer.
    """
    try:
        answer_clean = answer.strip().lower().replace("'", "")
        score = scores_dict[answer_clean]
    except KeyError:
        # If the answer doesn't match keys, use a classification model
        predicted_label = pipe(
            answer_clean,
            candidate_labels=list(scores_dict.keys())
        )['labels'][0]
        logger.warning(f"Invalid answer '{answer}', predicted: '{predicted_label}'")
        score = scores_dict[predicted_label]
    return score

def adjusted_score(row):
    """
    Adjusts the score based on the question type (direct or inverted).

    Args:
        row (pd.Series): DataFrame row containing question data and answer.

    Returns:
        int: Adjusted score.
    """
    if row['Q_type'] == 'inverted':
        return 6 - row['Scores']
    else:
        return row['Scores']

def save_results(df_answers, df_results, model_name, temperature, questionnaire_name, experiment_num):
    """
    Saves experiment results to CSV files.

    Args:
        df_answers (pd.DataFrame): DataFrame with answers to questionnaire questions.
        df_results (pd.DataFrame): DataFrame with experiment results.
        model_name (str): Model name.
        temperature (float): Temperature parameter used for generating answers.
        questionnaire_name (str): Name of the questionnaire.
        experiment_num (int): Experiment number.
    """
    from datetime import datetime

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"results/{questionnaire_name}_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)

    answers_filename = f"{folder_name}/answers_{model_name}_experiment_{experiment_num}.csv"
    results_filename = f"{folder_name}/results_{model_name}_experiment_{experiment_num}.csv"

    df_answers.to_csv(answers_filename, index=False)
    df_results.to_csv(results_filename, index=False)

    logger.info(f"Experiment {experiment_num} saved: Model={model_name}, Temperature={temperature}, Questionnaire={questionnaire_name}, Answers File={answers_filename}, Results File={results_filename}")

def load_traits_definitions(file_path='traits_definitions.json'):
    """
    Loads personality trait definitions from a JSON file.

    Args:
        file_path (str, optional): Path to the definitions file. Defaults to 'traits_definitions.json'.

    Returns:
        dict: Dictionary with personality trait definitions.
    """
    with open(file_path, 'r') as file:
        traits_definitions = json.load(file)
    return traits_definitions

def load_questions(file_path):
    """
    Loads questions from a CSV file.

    Args:
        file_path (str): Path to the questions file.

    Returns:
        list: List of questions.

    Raises:
        Exception: If there's an error loading the questions.
    """
    try:
        questions_df = pd.read_csv(file_path)
        if 'question' not in questions_df.columns:
            raise ValueError("CSV file must contain a 'question' column.")
        questions = questions_df['question'].tolist()
        return questions
    except Exception as e:
        logger.error(f"Error loading questions from '{file_path}': {e}")
        raise

def check_required_columns(df, required_columns):
    """
    Checks for the presence of required columns in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to check.
        required_columns (set): Set of required column names.

    Raises:
        ValueError: If required columns are missing.
    """
    if not required_columns.issubset(df.columns):
        missing_cols = required_columns - set(df.columns)
        raise ValueError(f"CSV file is missing columns: {', '.join(missing_cols)}")


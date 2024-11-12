from utils import common
import logging
import os
import pandas as pd
from datetime import datetime
from experiment_resources.language_models import *
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

logger = logging.getLogger(__name__)

def run_experiment(settings=None):
    """
    Runs the questionnaire experiment using the specified settings.

    Args:
        settings (dict): Experiment settings.

    Returns:
        None
    """
    if settings is not None:
        experiment_settings = settings
    else:
        raise FileNotFoundError(f"Experiment settings does not exist")
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"results/questionnaire_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)

    for model_name, model_params in settings['models'].items():
        model = get_model_instance(model_name)
        temperature = model_params['temperature']
        traits_definitions = common.traits_definitions
        questionnaire_module = settings['questionnaire_module']

        df_answers_full = pd.DataFrame()
        df_results_full = pd.DataFrame()
        for trait_key, questions in questionnaire_module.QUESTIONS.items():
            trait_name = traits_definitions['generation'][trait_key]['name']
            for level in ['high', 'low']:
                trait_prompt = "Act as a person with a "+ level + " score in " + traits_definitions['generation'][trait_key]['name'] + ". " +  traits_definitions['generation'][trait_key][level]
                df_answers, df_results = get_llm_answers(
                    model,
                    trait_key,
                    trait_name,
                    level,
                    trait_prompt,
                    temperature,
                    questionnaire_module
                )
                df_answers_full = pd.concat([df_answers_full, df_answers], ignore_index=True)
                df_results_full = pd.concat([df_results_full, df_results], ignore_index=True)

        # Save the results
        answers_filename = f"{folder_name}/answers_{model.model_name}.csv"
        results_filename = f"{folder_name}/results_{model.model_name}.csv"
        df_answers_full.to_csv(answers_filename, index=False)
        df_results_full.to_csv(results_filename, index=False)
        logger.info(f"Experiment saved: Model={model.model_name}, Temperature={temperature}, Answers File={answers_filename}, Results File={results_filename}")

        print(f"Experiment saved: Model={model.model_name}, Temperature={temperature}, Answers File={answers_filename}, Results File={results_filename}")

        print(df_results_full[['trait', 'trait_score', 'score']])


def get_llm_answers(model, trait_key, trait_name, trait_score, trait_prompt, temperature, questionnaire_module):
    """
    Fills the answer table for a given personality trait.

    Args:
        model: The language model instance.
        trait_key (str): Key of the personality trait.
        trait_name (str): Name of the personality trait.
        trait_score (str): 'high' or 'low' level of the trait.
        trait_prompt (str): Prompt describing the trait level.
        temperature (float): Temperature parameter for the model.
        questionnaire_module: The questionnaire module (e.g., BFI44).

    Returns:
        tuple: A tuple containing DataFrames for answers and results.
    """
    data_answers = []  # List to store rows
    for question_set in questionnaire_module.QUESTIONS[trait_key]:
        # Formulate the prompt using SCORES_DICT from the questionnaire
        provided_options = '\n'.join(f"- {option}" for option in questionnaire_module.SCORES_DICT.keys())
        constant_list = list(questionnaire_module.SCORES_DICT.keys())

        system_prompt = f"""
TASK:
Indicate your level of agreement or disagreement with the statement in the CHARACTERISTICS according to your PERSONALITY. Use only the PROVIDED OPTIONS.

PERSONALITY:
```
{trait_prompt}
```

PROVIDED OPTIONS:
{provided_options}

Provide your output only from the constant list {constant_list} without explanation.
"""

        user_prompt = f"""
CHARACTERISTICS:
```
{question_set['q_statement']}
```
"""

        answer, _ = model.generate_response(system_prompt, user_prompt, temperature)
        data_answers.append({
            'model': model.model_name,
            'temperature': temperature,
            'trait': trait_name,
            'trait_score': trait_score,
            'trait_prompt': trait_prompt,
            'Q_number': question_set['q_num'],
            'Question': question_set['q_statement'],
            'Q_type': question_set['q_type'],
            'Answer': answer.strip()
        })

    df_answers = pd.DataFrame(data_answers)

    scores_dict = questionnaire_module.SCORES_DICT
    df_answers['Scores'] = df_answers['Answer'].apply(lambda x: match_score(x, scores_dict))
    df_answers['Adjusted Scores'] = df_answers.apply(adjusted_score, axis=1)
    average_score = df_answers['Adjusted Scores'].mean()

    df_results = pd.DataFrame([{
        'model': model.model_name,
        'temperature': temperature,
        'trait': trait_name,
        'trait_score': trait_score,
        'trait_prompt': trait_prompt,
        'score': average_score
    }])

    return df_answers, df_results


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

def visualize_results(df_answers='last', visualization_settings=None, visualization_type='histogram'):
    """
    Visualizes the results of the questionnaire experiment.

    Args:
        df_answers (str): The experiment folder name or 'last' for the latest.
        visualization_settings (dict): Settings for visualization.
        visualization_type (str): Type of visualization ('histogram').

    Returns:
        None
    """
    if visualization_type == 'histogram':
        plot_histograms(df_answers=df_answers, visualization_settings=visualization_settings)
    else:
        print('This visualization type does not support')
        logger.warning(f"Invalid visualization type")


def plot_histograms(df_answers='last', visualization_settings=None):
    """
    Plots histograms for the questionnaire results.

    Args:
        df_answers (str): The experiment folder name or 'last' for the latest.
        visualization_settings (dict): Settings for visualization.

    Returns:
        None
    """
    df_answers_full = pd.DataFrame()
    
    if df_answers == 'last':
        # Find the latest experiment folder
        results_dir = 'results'
        experiment_folders = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d)) and d.startswith('questionnaire_')]
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
    
    # Collect all 'answers_' CSV files in the experiment folder
    answer_files = [os.path.join(experiment_path, f) for f in os.listdir(experiment_path) if f.startswith('answers_') and f.endswith('.csv')]
    if not answer_files:
        print(f"No 'answers_' CSV files found in the experiment folder '{experiment_path}'.")
        return
    
    # Read data from files
    for file_path in answer_files:
        try:
            df = pd.read_csv(file_path)
            df_answers_full = pd.concat([df_answers_full, df], ignore_index=True)
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            print(f"Error reading file {file_path}: {e}")
            return

    # Check for necessary columns
    required_columns = {'model', 'trait', 'Adjusted Scores', 'trait_score'}
    if not required_columns.issubset(df_answers_full.columns):
        missing_cols = required_columns - set(df_answers_full.columns)
        logger.error(f"Data is missing columns: {', '.join(missing_cols)}")
        print(f"Error: Data is missing columns: {', '.join(missing_cols)}")
        return

    # Rename 'Adjusted Scores' to 'score' for consistency
    df_answers_full = df_answers_full.rename(columns={'Adjusted Scores': 'score'})

    # Set up plot styles
    sns.set(font_scale=1.5, style="whitegrid")

    rc('font', **{'family': 'serif'})

    # Get unique models and personality traits
    models = df_answers_full['model'].unique()
    traits = df_answers_full['trait'].unique()

    # Determine the number of rows and columns for subplots
    n_rows = len(models)
    n_cols = len(traits)

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

    # Color palette
    colors = ['#6788ee', '#e26952']

    # Trait name mapping
    if visualization_settings:
        trait_name_mapping = visualization_settings.get('trait_name_mapping', {
            'Agreeableness': 'Agreeableness',
            'Conscientiousness': 'Conscientiousness',
            'Extraversion': 'Extraversion',
            'Neuroticism': 'Neuroticism',
            'Openness': 'Openness to experience'
        })

        # Model name mapping
        model_name_mapping = visualization_settings.get('model_name_mapping', {
            'gpt-3.5-turbo': 'GPT-3.5 Turbo',
            'gpt-4': 'GPT-4',
            # Add other models if necessary
        })

    # Ensure axes is a 2D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    # Loop over models and traits to create plots
    for m, model in enumerate(models):
        for t, trait in enumerate(traits):
            ax = axes[m, t]
            data = df_answers_full[(df_answers_full['model'] == model) & (df_answers_full['trait'] == trait)].copy()
            if data.empty:
                logger.warning(f"No data for model {model} and trait {trait}")
                ax.axis('off')
                continue
            # Plot histogram
            sns.histplot(
                data=data.reset_index(drop=True),
                x='score',
                hue='trait_score',
                stat='probability',
                palette=colors,
                element="step",
                kde=True,
                ax=ax,
                alpha=0.7,
                linewidth=3,
                line_kws=dict(linewidth=8)
            )
            # Set Y-axis labels in the first column
            if t == 0:
                ax.set_ylabel(model_name_mapping.get(model, model), fontsize=14)
            else:
                ax.set_ylabel('')
            # Set X-axis labels in the last row
            if m == n_rows -1:
                ax.set_xlabel(trait_name_mapping.get(trait, trait), fontsize=14)
            else:
                ax.set_xlabel('')
            # Set axis limits
            ax.set_xlim((1, 5))
            ax.set_ylim((0, 1))
            # Move legend
            sns.move_legend(
                ax, "lower center",
                bbox_to_anchor=(0.5, 1), ncol=2, title=None, frameon=False,
            )

    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(experiment_path, "histograms.pdf")
    plt.savefig(output_file, bbox_inches='tight')
    logger.info(f"Questionnaire visualization saved to {output_file}")
    print(f"Questionnaire visualization saved to {output_file}")

    plt.show()

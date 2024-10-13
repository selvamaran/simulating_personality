# questionnaire_experiment.py

"""
This module runs the questionnaire experiment using the specified language model and settings.
It simulates responses to questionnaire items based on different personality traits.

Users can modify:
- The language models used.
- The personality traits and their definitions.
- The questionnaire module (e.g., replace BFI44 with another questionnaire).

"""

import os
import pandas as pd
from datetime import datetime
from utils import match_score, adjusted_score
import logging

logger = logging.getLogger(__name__)

def run_questionnaire_experiment(model, temperature, questionnaire_module, experiment_count, traits_definitions):
    """
    Runs the questionnaire experiment with the given model and settings.

    Args:
        model: The language model instance.
        temperature (float): Temperature parameter for the model.
        questionnaire_module: The questionnaire module (e.g., BFI44).
        experiment_count (int): Number of times to repeat the experiment.
        traits_definitions (dict): Dictionary containing personality trait definitions.

    Returns:
        None
    """
    for experiment_num in range(experiment_count):
        df_answers_full = pd.DataFrame()
        df_results_full = pd.DataFrame()
        for trait_key, questions in questionnaire_module.QUESTIONS.items():
            trait_name = traits_definitions[trait_key]['name']
            for level in ['high', 'low']:
                trait_prompt = "Act as a person with a "+ level + " score in " + traits_definitions[trait_key]['name'] + ". " +  traits_definitions[trait_key][level]
                df_answers, df_results = fill_answer_table(
                    model,
                    trait_key,
                    trait_name,
                    level,
                    trait_prompt,
                    experiment_num,
                    temperature,
                    questionnaire_module
                )
                df_answers_full = pd.concat([df_answers_full, df_answers], ignore_index=True)
                df_results_full = pd.concat([df_results_full, df_results], ignore_index=True)

        # Save the results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        folder_name = f"results/questionnaire_{timestamp}"
        os.makedirs(folder_name, exist_ok=True)
        answers_filename = f"{folder_name}/answers_{model.model_name}_experiment_{experiment_num}.csv"
        results_filename = f"{folder_name}/results_{model.model_name}_experiment_{experiment_num}.csv"
        df_answers_full.to_csv(answers_filename, index=False)
        df_results_full.to_csv(results_filename, index=False)
        logger.info(f"Experiment {experiment_num} saved: Model={model.model_name}, Temperature={temperature}, Answers File={answers_filename}, Results File={results_filename}")

def fill_answer_table(model, trait_key, trait_name, trait_score, trait_prompt, experiment_num, temperature, questionnaire_module):
    """
    Fills the answer table for a given personality trait.

    Args:
        model: The language model instance.
        trait_key (str): Key of the personality trait.
        trait_name (str): Name of the personality trait.
        trait_score (str): 'high' or 'low' level of the trait.
        trait_prompt (str): Prompt describing the trait level.
        experiment_num (int): Experiment number.
        temperature (float): Temperature parameter for the model.
        questionnaire_module: The questionnaire module (e.g., BFI44).

    Returns:
        tuple: A tuple containing DataFrames for answers and results.
    """
    data_answers = []  # Create a list to store rows
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
            'experiment_num': experiment_num,
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
        'experiment_num': experiment_num,
        'model': model.model_name,
        'temperature': temperature,
        'trait': trait_name,
        'trait_score': trait_score,
        'trait_prompt': trait_prompt,
        'score': average_score
    }])

    return df_answers, df_results

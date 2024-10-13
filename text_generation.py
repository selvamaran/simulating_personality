# text_generation.py

"""
This module is responsible for generating texts based on specified personality traits and questions.
It uses language models to simulate responses from individuals with different trait scores.

Users can modify:
- The list of models used for text generation.
- The personality traits and their definitions in 'traits_definitions.json'.
- The questions used for text generation.

"""

import os
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def generate_texts(model, traits_definitions, experiment_settings, questions, experiment_num):
    """
    Generates texts based on specified personality traits and questions.

    Args:
        model: The language model instance used for text generation.
        traits_definitions (dict): Dictionary containing personality trait definitions.
        experiment_settings (dict): Experiment settings, including temperature.
        questions (list): List of questions to be answered.
        experiment_num (int): The experiment number.

    Returns:
        pd.DataFrame: DataFrame containing the generated texts.
    """
    data_texts = []  # Create a list to store rows

    temperature = experiment_settings['temperature']

    for trait_key, trait_info in traits_definitions.items():
        trait_name = trait_info['name']
        for question in questions:
            for trait_score in range(1, 6):
                answer = write_story(
                    model,
                    traits_definitions,
                    trait_key,
                    trait_name,
                    trait_score,
                    question,
                    temperature
                )
                data_texts.append({
                    'experiment_num': experiment_num,
                    'model': model.model_name,
                    'temperature': temperature,
                    'trait': trait_name,
                    'trait_score': trait_score,
                    'question': question,
                    'answer': answer
                })
                logger.info(f"Generated text for trait {trait_name}, score {trait_score}, question '{question}', temperature {temperature}")
        
    df_texts = pd.DataFrame(data_texts)
    return df_texts

def write_story(model, traits_definitions, trait_key, trait_name, trait_score, question, temperature):
    """
    Generates an answer to a question based on the specified personality trait and score.

    Args:
        model: The language model instance used for text generation.
        traits_definitions (dict): Dictionary containing personality trait definitions.
        trait_key (str): Key of the personality trait.
        trait_name (str): Name of the personality trait.
        trait_score (int): Score of the personality trait (1 to 5).
        question (str): The question to be answered.
        temperature (float): Temperature parameter for the model.

    Returns:
        str: The generated answer.
    """
    system_prompt = f"""
TASK:
Answer the QUESTION according to your PERSONALITY. Use INSTRUCTION. Use at most 5 sentences. Do not mention your personality traits in the text. Type only the answer, without the information about your personality score.

PERSONALITY:
- Your personality trait "{trait_name}" is rated as {trait_score}.

INSTRUCTION:
- The personality trait is rated from 1 to 5. 1 is the lowest score and 5 is the highest score.
- {traits_definitions[trait_key]['low']}
- {traits_definitions[trait_key]['high']}
"""

    user_prompt = f"""
QUESTION:
```
{question}
```
"""

    answer, _ = model.generate_response(system_prompt, user_prompt, temperature)
    return answer.strip()

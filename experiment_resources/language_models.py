import os
import time
import logging
from abc import ABC, abstractmethod

# Set up logging
logging.basicConfig(
    filename='language_models.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

class LanguageModel(ABC):
    """
    Abstract base class for language models.
    """
    @abstractmethod
    def generate_response(self, system_prompt, user_prompt, temperature):
        pass


class OpenAIModel(LanguageModel):
    """
    Class representing an OpenAI language model.
    """
    def __init__(self, model_name):
        import openai
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=self.api_key)

    def generate_response(self, system_prompt, user_prompt, temperature, max_retries=5):
        """
        Generates a response using the OpenAI language model.

        Args:
            system_prompt (str): The system prompt.
            user_prompt (str): The user's input prompt.
            temperature (float): The temperature for randomness.
            max_retries (int): Maximum number of retries in case of failure.

        Returns:
            tuple: The final response and status ('ok' or 'fail').
        """
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt}
                    ],
                    temperature=temperature
                )
                final_response = response.choices[0].message.content
                logging.info(f"OpenAI: Successfully received response on attempt {attempt}")
                return final_response, 'ok'
            except openai.OpenAIError as e:
                wait_time = 2 ** attempt
                logging.error(f"OpenAI: Attempt {attempt} failed with error: {e}. Retrying in {wait_time} seconds.")
                time.sleep(wait_time)
        logging.critical("OpenAI: Max retries exceeded. Returning 'GPT Fail'.")
        return 'GPT Fail', 'fail'


class AnthropicModel(LanguageModel):
    """
    Class representing an Anthropic language model.
    """
    def __init__(self, model_name):
        from anthropic import Anthropic
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        self.model_name = model_name
        self.client = Anthropic(api_key=self.api_key)

    def generate_response(self, system_prompt, user_prompt, temperature, max_retries=5):
        """
        Generates a response using the Anthropic language model.

        Args:
            system_prompt (str): The system prompt.
            user_prompt (str): The user's input prompt.
            temperature (float): The temperature for randomness.
            max_retries (int): Maximum number of retries in case of failure.

        Returns:
            tuple: The final response and status ('ok' or 'fail').
        """
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    max_tokens=1024,
                    temperature=temperature
                )
                final_response = response.content[0].text.strip()
                logging.info(f"Anthropic: Successfully received response on attempt {attempt}")
                return final_response, 'ok'
            except Exception as e:
                wait_time = 2 ** attempt
                logging.error(f"Anthropic: Attempt {attempt} failed with error: {e}. Retrying in {wait_time} seconds.")
                time.sleep(wait_time)
        logging.critical("Anthropic: Max retries exceeded. Returning 'GPT Fail'.")
        return 'GPT Fail', 'fail'

class MistralModel(LanguageModel):
    """
    Class representing a Mistral AI language model.
    """
    def __init__(self, model_name):
        from mistralai import Mistral, UserMessage
        self.api_key = os.getenv('MISTRAL_API_KEY')
        self.model_name = model_name
        self.client = Mistral(api_key=self.api_key)

    def generate_response(self, system_prompt, user_prompt, temperature, max_retries=5):
        """
        Generates a response using the Mistral language model.

        Args:
            system_prompt (str): The system prompt.
            user_prompt (str): The user's input prompt.
            temperature (float): The temperature for randomness.
            max_retries (int): Maximum number of retries in case of failure.

        Returns:
            tuple: The final response and status ('ok' or 'fail').
        """
        prompt = system_prompt + '\n' + user_prompt
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.complete(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature
                )
                final_response = response.choices[0].message.content.strip()
                logging.info(f"Mistral: Successfully received response on attempt {attempt}")
                return final_response, 'ok'
            except Exception as e:
                wait_time = 2 ** attempt
                logging.error(f"Mistral: Attempt {attempt} failed with error: {e}. Retrying in {wait_time} seconds.")
                time.sleep(wait_time)
        logging.critical("Mistral: Max retries exceeded. Returning 'GPT Fail'.")
        return 'GPT Fail', 'fail'


def get_model_instance(model_name):
    """
    Returns an instance of the specified language model.

    Args:
        model_name (str): Name of the model.

    Returns:
        LanguageModel: An instance of a subclass of LanguageModel.

    Raises:
        ValueError: If the model name is not supported.
    """
    if model_name in ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4']:
        return OpenAIModel(model_name)
    elif model_name in ['claude-3-haiku-20240307']:
        return AnthropicModel(model_name)
    elif model_name in ['open-mixtral-8x22b']:
        return MistralModel(model_name)
    else:
        raise ValueError(f"Model {model_name} is not supported.")

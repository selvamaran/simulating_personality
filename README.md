
# Simulating Personality with Large Language Models

This repository contains an analytical framework developed as part of the research project "**Exploring the Potential of Large Language Models to Simulate Personality**," presented at the [Workshop on Customizable NLP (EMNLP 2024, November 16)](https://customnlp4u-24.github.io/). A link to the research paper will be provided after its publication.

**Contributors**: Maria Molchanova, Anna Mikhailova, Anna Korzanova, Lidiia Ostyakova, Alexandra Dolidze.

## Background

With the advancement of large language models (LLMs), the focus in conversational AI has shifted from merely generating coherent and relevant responses to tackling more complex challenges like personalizing dialogue systems. To enhance user engagement, chatbots are often designed to mimic human behavior, responding within a defined emotional spectrum and aligning with specific values. Our research aims to simulate personality traits according to the Big Five model using LLMs.

## Analytical Framework

In exploring the capacity of LLMs to generate personality-consistent content, we've developed and released an analytical framework that allows researchers to replicate our study with custom parameters. Within this framework:

- **Questionnaire Responses**: Evaluated models respond to items from selected questionnaires, and their answers are used for graphical analysis.
- **Text Generation**: Models generate texts based on specified prompts, which are then automatically analyzed using an LLM-based classifier. This assesses the accuracy and consistency with which models adhere to assigned personality traits during text generation.

By making this framework publicly available on GitHub, we aim to contribute a valuable tool for advancing research in personality simulation using LLMs.

## Customization Opportunities

- **Integrate other LLMs and configurations** (e.g., temperature settings).
- **Experiment with different personality imitation techniques** by modifying prompts—for example, using related facets or behavioral markers.
- **Employ other questionnaires** to detect personality traits.
- **Upload custom questions** for text generation.

*For personality definitions, we used texts from Annabelle G.Y. Lim's [Big Five Personality Traits: The 5-Factor Model of Personality (2023)](https://www.simplypsychology.org/big-five-personality.html).*

## Usage

The framework is organized into steps executed in `main.ipynb`. Below is a detailed description of each step and how to use them.

### 1. Questionnaire Experiment

- **Purpose**: Simulate questionnaire responses for different personality traits using language models.
- **How It Works**: The model is prompted to act as a person with a high or low score in a specific trait and answers questionnaire items accordingly. Responses are saved for analysis.

### 2. Text Generation

- **Purpose**: Generate texts that reflect specific personality traits and scores.
- **How It Works**: The model is given a trait and a score (1 to 5) and answers custom questions based on this trait score. Generated texts are saved for further analysis.

## Installation

Clone the repository:

```bash
git clone https://github.com/mary-silence/simulating_personality.git
```

## Getting Started

An example of how to use the framework is provided in the main.ipynb file.

1. **Initialize the Framework**

   Run the initialization command:

   ```python
   common.initialize()
   ```

2. **Set Experiment Parameters**

   Configure the necessary settings for your experiments.

3. **Questionnaire Experiment**

   To conduct an experiment where the LLM responds to personality questionnaires:

   ```python
   questionnaire.run_experiment(params)
   ```

   - Results are saved in the `results` folder.
   - Visualize the results with:

     ```python
     questionnaire.visualize_results(params)
     ```
   **Available Analysis Tools**:
    - **Histograms**: Visualize histograms showing the distribution of responses.

4. **Text Generation Experiment**

   To conduct experiments with text generation:

   ```python
   qa_text_generation.run_experiment(params)
   ```

   - Generated texts are saved in the `results` folder.
   - Analyze results using the LLM-based classifier:

     ```python
     qa_classifier.run_analysis(params)
     ```

   - Visualize the results:

     ```python
     qa_text_generation.visualize_results(params)
     ```

   **Available Analysis Tools**:

   - **Confusion Matrices**: Assess how accurately LLMs imitate the specified personality traits in the prompts.
   - **Cosine Similarity of Generated Texts**: Available without classifier annotation.

## Customization Options

### Basic

#### Changing Language Models and Settings

- Specify a list of LLMs and temperature settings in `main.ipynb` under `questionnaire_settings` and `text_generation_settings`.
- If the model is among `gpt-3.5-turbo`, `gpt-4`, `claude-3-haiku-20240307`, or `open-mixtral-8x22b`, you can add them directly and adjust settings as shown.
- For models not in this list, follow the instructions in the **Advanced** section below.

#### Modifying Questions for Text Generation (Step 2)

To experiment with new questions:

1. Add questions to the `questions.csv` file.
2. **Or** create a new CSV file with a required `question` column and specify it in the `text_generation_settings` block in `main.ipynb`.

#### Changing Personality Definitions

##### For questionnarie answering and text generation
- Experiment with different ways of defining personality traits (e.g., using related facets or additional linguistic features).
- Edit the `traits_definitions.json` file, which contains personality descriptions used for generation and for the classifier.
- **Note**:  We recommend changing classifier descriptions only if necessary, as the current descriptions provide stable performance.

#### Changing the Classifier Model

- Adjust classifier settings in `main.ipynb` under `text_analysis_settings`.
- For models like `gpt-3.5-turbo`, `gpt-4`, `claude-3-haiku-20240307`, or `open-mixtral-8x22b`, simply change `model_name` or `temperature`.
- To use a different model, see "How to Add a New Large Language Model" below.
- **Note**: We recommend changing the classifier only if necessary, as the current settings and prompts offer verified stable performance.

### Advanced

#### How to Add a New Large Language Model

All LLMs are managed in the `language_models.py` file.

- **For models provided by OpenAI, Anthropic, Mistral AI**:
  - Add the model to the list in the `get_model_instance` function.
- **For models from other vendors or those running locally**:
  1. **Create a new class** for the model.
  2. **Implement a `generate_response` function** that accepts `system_prompt`, `user_prompt`, `temperature`, and `max_retries` (number of retries if an error occurs). If the model doesn't support separate prompts, combine them appropriately within the function.
  3. **Add the model** to `get_model_instance`.
  4. **Specify the new model** in `questionnaire_settings` and `text_generation_settings` in `main.ipynb`.

#### How to Add a New Questionnaire

To incorporate another questionnaire for personality research:

1. **Place the new questionnaire** in the `questionnaires` folder.
2. **Define a `scores_dict`**. Currently, questionnaires support only 5-point Likert scale responses in order.
3. **Specify the questions** in the `QUESTIONS` section, containing keys for personality traits. Each trait corresponds to a list of questions with parameters:
   - **Question number** in the questionnaire.
   - **Question type**:
     - `direct`: Scores are summed directly.
     - `inverted`: Scores are reversed before summing.
4. **Import the questionnaire** into `main.ipynb`, replacing the default.
5. **Specify the new questionnaire** in `questionnaire_settings`.

---

We hope this framework assists you in advancing research on simulating personality traits using LLMs. If you have any questions or need further assistance, please feel free to open an issue or contact the contributors.

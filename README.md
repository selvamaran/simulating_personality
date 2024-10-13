
# Simulating Personality

## Overview
This repository contains a framework designed for conducting research on personality trait analysis using natural language processing (NLP) models. Developed for the EMNLP Workshop on Customizable NLP (CustomNLP4U), the framework allows users to:
- Specify their own language models and settings.
- Utilize different personality questionnaires (e.g., BFI-44, IPIP).
- Load custom questions for text generation.
- Analyze generated texts for personality traits.
- Visualize the results through various plots and similarity analyses.

## Table of Contents
- Background
  - The Big Five Personality Traits
  - Trait Definitions
- Features
- Installation
- Usage
  - 1. Questionnaire Experiment
  - 2. Text Generation
  - 3. Text Analysis
  - 4. Text Similarity Analysis
  - 5. Visualization
- Customization
  - Module Descriptions
- Contributing
- License

## Background

### The Big Five Personality Traits
The Big Five personality traits, also known as the Five-Factor Model (FFM), is one of the most widely accepted frameworks in psychology for understanding human personality. The model proposes that five main dimensions are sufficient to capture the variations in human personality:
- Openness to Experience
- Conscientiousness
- Extraversion
- Agreeableness
- Neuroticism

These traits are often remembered by the acronym OCEAN.

### Trait Definitions
Below are the definitions of each trait, as used in this framework:

#### 1. Openness to Experience
**Definition:** Openness to experience refers to one’s willingness to try new things as well as engage in imaginative and intellectual activities. It includes the ability to “think outside of the box.”
**High Score:** A person with a high score in Openness to Experience tends to be imaginative, curious, and open-minded. They are often eager to explore new ideas, experiences, and creative pursuits, displaying a preference for novelty and variety.
**Low Score:** A person with a low score in Openness to Experience typically prefers routine and familiarity over new experiences, showing caution and skepticism towards change. They tend to be more conventional and practical, often sticking to traditional ways of doing things and avoiding abstract or theoretical discussions.

#### 2. Conscientiousness
**Definition:** Conscientiousness describes a person’s ability to regulate impulse control to engage in goal-directed behaviors. It measures elements such as control, inhibition, and persistence of behavior.
**High Score:** A person with a high score in Conscientiousness is typically very organized, responsible, and dependable. They exhibit a strong sense of duty, aim for high achievement, and carefully plan and execute tasks with attention to detail.
**Low Score:** A person with a low score in Conscientiousness tends to be more spontaneous and may struggle with organization and discipline. They often prioritize flexibility and spontaneity over detailed planning and may overlook details or deadlines.

#### 3. Extraversion
**Definition:** Extraversion reflects the tendency and intensity to which someone seeks interaction with their environment, particularly socially. It encompasses the comfort and assertiveness levels of people in social situations. Additionally, it also reflects the sources from which someone draws energy.
**High Score:** A person with a high score in Extraversion is typically outgoing, energetic, and sociable. They enjoy interacting with others, are often enthusiastic and action-oriented, drawing energy from being around people.
**Low Score:** A person with a low score in Extraversion, often described as introverted, usually prefers solitude or small group interactions over large social gatherings. They tend to be more reserved and less outspoken, drawing energy from their inner world and enjoying quiet environments.

#### 4. Agreeableness
**Definition:** Agreeableness refers to how people tend to treat relationships with others. Unlike extraversion which consists of the pursuit of relationships, agreeableness focuses on people’s orientation and interactions with others.
**High Score:** A person with a high score in Agreeableness is typically compassionate, cooperative, and trusting. They value harmony and are often considerate, friendly, and willing to compromise to avoid conflict.
**Low Score:** A person with a low score in Agreeableness tends to be more competitive and less cooperative. They may prioritize their own interests over others', often appearing skeptical, critical, and less concerned with maintaining harmonious relationships.

#### 5. Neuroticism
**Definition:** Neuroticism describes the overall emotional stability of an individual through how they perceive the world. It takes into account how likely a person is to interpret events as threatening or difficult.
**High Score:** A person with a high score in Neuroticism often experiences emotional instability and is prone to feelings of anxiety, worry, and mood swings. They are more likely to react negatively to stress and can struggle to return to a state of emotional equilibrium after upsetting events.
**Low Score:** A person with a low score in Neuroticism typically exhibits emotional stability and resilience. They remain calm and composed under stress, rarely feeling distressed or overwhelmed by negative emotions.

## Features
- Model Flexibility: Easily integrate different language models (e.g., GPT-3.5, GPT-4, Claude) by specifying them in the settings.
- Custom Questionnaires: Swap out the default BFI-44 questionnaire with other personality assessments.
- Custom Questions: Load your own set of questions for text generation.
- Text Generation: Generate texts that simulate responses from individuals with varying levels of personality traits.
- Text Analysis: Analyze generated texts to detect embedded personality traits.
- Visualization: Generate plots such as confusion matrices and similarity heatmaps to visualize results.
- Modular Design: Organized codebase with modules for different functionalities, making it easy to extend or modify.

## Installation
Clone the repository:
```bash
git clone https://github.com/mary-silence/simulating_personality.git
cd simulating_personality
```
Install the required packages:
```bash
pip install -r requirements.txt
```
Or install them individually:
```bash
pip install openai anthropic mistralai pandas numpy matplotlib seaborn scikit-learn
```
Set up API keys:
```bash
export OPENAI_API_KEY='your-openai-api-key'
export ANTHROPIC_API_KEY='your-anthropic-api-key'
export MISTRAL_API_KEY='your-mistral-api-key'
```

## Usage
The framework is organized into steps executed in `main.ipynb`. Below is a detailed description of each step and how to use them.

### 1. Questionnaire Experiment
**Purpose:** Simulate questionnaire responses for different personality traits using language models.
**How it works:**
The model is prompted to act as a person with a high or low score in a specific trait. It answers questionnaire items accordingly. The responses are saved for analysis.

### 2. Text Generation
**Purpose:** Generate texts that reflect specific personality traits and scores.
**How it works:**
The model is given a trait and a score (1 to 5). It answers custom questions based on this trait score. The generated texts are saved for further analysis.

### 3. Text Analysis
**Purpose:** Analyze the generated texts to detect embedded personality traits.
**How it works:**
A classifier model analyzes each text. It predicts the trait scores based on the content. The analysis results are saved for comparison with the original scores.

### 4. Text Similarity Analysis
**Purpose:** Analyze the similarity between generated texts to understand overlaps between different trait levels.
**How it works:**
Text embeddings are computed for each text. Pairwise similarities are calculated. Similarity matrices are visualized.

### 5. Visualization
**Purpose:** Visualize the results through plots like confusion matrices and histograms.
**Types of Visualizations:**
- Confusion Matrices: Compare the prompted trait scores with the detected scores.
- Histograms: Show the distribution of questionnaire responses.
- Similarity Heatmaps: Display the average similarities between texts of different trait scores.

## Customization
The framework is highly customizable:
- Adding New Models: Extend `models/language_models.py` to include new language models.
- Changing Questionnaires: Replace `questionnaires/BFI44.py` with another questionnaire module.
- Custom Questions: Provide your own questions in a CSV file and specify the path in `text_generation_settings['questions_file']`.
- Modifying Trait Definitions: Edit `traits_definitions.json` to adjust trait descriptions or add new traits.
- Adjusting Analysis Parameters: Modify settings in `text_analysis.py` or `text_similarity_analysis.py` to change analysis behaviors.

## Module Descriptions
- `main.ipynb`: The main notebook that orchestrates the execution of experiments and analyses.
- `models/language_models.py`: Contains classes representing different language models.
- `text_generation.py`: Responsible for generating texts based on traits and questions.
- `text_analysis.py`: Analyzes generated texts to detect personality traits.
- `plotting.py`: Generates visualizations like confusion matrices.
- `text_similarity_analysis.py`: Performs similarity analysis on generated texts.
- `questionnaire_experiment.py`: Runs the questionnaire experiment.
- `questionnaire_visualization.py`: Visualizes questionnaire answers.
- `utils.py`: Contains utility functions for score matching and data handling.
- `traits_definitions.json`: JSON file containing definitions and descriptions of personality traits.
- `questionnaires/BFI44.py`: Module containing the BFI-44 questionnaire.

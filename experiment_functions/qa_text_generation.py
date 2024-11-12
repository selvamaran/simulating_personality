import pandas as pd
from pathlib import Path
import logging
import os
from datetime import datetime
from experiment_resources.language_models import *
from utils import common
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise_distances
from matplotlib import rc

logger = logging.getLogger(__name__)

def run_experiment(settings):
    """
    Runs the text generation experiment using the specified settings.

    Args:
        settings (dict): A dictionary containing experiment settings.

    Returns:
        None
    """
    if settings is not None:
        experiment_settings = settings
    else:
        raise FileNotFoundError(f"Experiment settings does not exist")
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"results/qa_text_generation_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)

    questions = load_data(experiment_settings)
    if questions == 'Fail':
        print("Can't continue experiment")
        return
    else:
        print('Quesions have been loaded')

    traits_definitions = common.traits_definitions

    # Execute text generation
    generated_texts_files = []

    for model_name, model_params in experiment_settings['models'].items():
        model = get_model_instance(model_name)
        temperatures = model_params.get('temperatures', [0.7])  # If temperatures are not specified, use 0.7 by default
        for temperature in temperatures:
            df_texts = get_llm_answers(
                model,
                traits_definitions['generation'],
                {'temperature': temperature},
                questions
            )
            # Save generated texts
            texts_filename = f"{folder_name}/texts_{model_name}_temp_{temperature}.csv"
            df_texts.to_csv(texts_filename, index=False)
            generated_texts_files.append(texts_filename)
            logger.info(f"Texts saved: {texts_filename}")

    # Output the list of paths to the saved files
    print("Generated texts files:")
    for file in generated_texts_files:
        print(file)


def load_data(settings):
    """
    Loads questions from a CSV file specified in the settings.

    Args:
        settings (dict): Experiment settings containing 'questions_file'.

    Returns:
        list: A list of questions if successful, 'Fail' otherwise.
    """
    try:
        current_file = Path(__file__).resolve()
        questions_path = current_file.parent.parent / 'qa_datasets' / settings['questions_file']
        questions_df = pd.read_csv(questions_path)
        if 'question' not in questions_df.columns:
            raise ValueError("CSV file must contain a 'question' column.")
        questions = questions_df['question'].tolist()
        return questions
    except Exception as e:
        print(f"Error loading questions from '{settings['questions_file']}': {e}")
        questions = []
        return 'Fail'


def get_llm_answers(model, traits_definitions, experiment_settings, questions):
    """
    Generates answers using the language model for different traits and questions.

    Args:
        model: The language model instance.
        traits_definitions (dict): Definitions of personality traits.
        experiment_settings (dict): Settings for the experiment, including temperature.
        questions (list): A list of questions.

    Returns:
        pd.DataFrame: DataFrame containing generated texts.
    """
    data_texts = []  # List to store rows
    temperature = experiment_settings['temperature']

    for trait_key, trait_info in traits_definitions.items():
        trait_name = trait_info['name']
        for question in questions:
            for trait_score in range(1, 6):
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
                answer,_ = model.generate_response(system_prompt, user_prompt, temperature)
                answer = answer.strip()

                data_texts.append({
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


def visualize_results(df_answers='last', visualization_settings=None, visualization_type=None):
    """
    Visualizes the results based on the specified type.

    Args:
        df_answers (str): The experiment folder name or 'last' for the latest.
        visualization_settings (dict): Settings for visualization.
        visualization_type (str): Type of visualization ('confusion matrices' or 'similarity').

    Returns:
        None
    """
    if visualization_type.lower() == 'confusion matrices':
        plot_confusion_matrices(df_answers=df_answers, visualization_settings=visualization_settings)
    elif visualization_type.lower() == 'similarity':
        analyze_text_similarity(df_answers=df_answers, visualization_settings=visualization_settings)
    elif visualization_type is None:
        print('Select visualization type: confusion matrices, similarity')
    else:
        print('This visualization type does not support. Select visualization type: confusion matrices, similarity')
        logger.warning(f"Invalid visualization type")


def plot_confusion_matrices(df_answers='last', visualization_settings=None):
    """
    Plots confusion matrices for the analyzed texts.

    Args:
        df_answers (str): The experiment folder name or 'last' for the latest.
        visualization_settings (dict): Settings for visualization.

    Returns:
        None
    """
    df_answers_full = pd.DataFrame()

    # Determine the experiment path
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

    # Collect all 'analysis_' CSV files in the experiment folder
    answer_files = [os.path.join(experiment_path, f) for f in os.listdir(experiment_path) if f.startswith('analysis_') and f.endswith('.csv')]
    if not answer_files:
        print(f"No appropriate CSV files found in the experiment folder '{experiment_path}'.")
        return

    # Get trait definitions
    traits_definitions = common.traits_definitions

    # Extract model_name_mapping from visualization_settings
    if visualization_settings:
        model_name_mapping = visualization_settings.get('model_name_mapping', {})
    else:
        model_name_mapping = {}

    # Initialize lists to accumulate data
    all_dfs = []
    model_names_set = set()
    for file_path in answer_files:
        df_analysis = pd.read_csv(file_path)
        all_dfs.append(df_analysis)
        model_names_set.update(df_analysis['model'].unique())

    # Combine all dataframes
    df_analysis = pd.concat(all_dfs, ignore_index=True)

    # Get unique models and personality traits
    model_names = list(model_names_set)
    traits = [trait_info['name'] for trait_info in traits_definitions['generation'].values()]
    n_rows = len(model_names)
    n_cols = len(traits)

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

    # Ensure axes is a 2D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    score_categories = ['low', 'medium', 'high']
    missing_scores = 0  # Counter for rows with missing 'score' values

    for m, model_name in enumerate(model_names):
        for t, trait_name in enumerate(traits):
            ax = axes[m, t]
            df = df_analysis[(df_analysis['model'] == model_name) & (df_analysis['analyzed_trait'] == trait_name)]

            if df.empty:
                logger.warning(f"No data found for model {model_name} and trait {trait_name}")
                ax.axis('off')
                continue

            # Exclude rows with missing values and 'decision type' == 'Nondistinguishable'
            initial_row_count = df.shape[0]
            df = df.dropna(subset=['score', 'trait_score', 'decision type']).copy()
            df = df[df['decision type'] != 'Nondistinguishable'].copy()
            missing_rows = initial_row_count - df.shape[0]
            missing_scores += missing_rows

            if df.empty:
                logger.warning(f"All data excluded after filtering for model {model_name} and trait {trait_name}")
                ax.axis('off')
                continue

            # Map values
            df['Prompted Score'] = df['trait_score'].apply(map_trait_score)
            df['Detected Score'] = df['score'].apply(map_detected_score)
            df = df.dropna(subset=['Prompted Score', 'Detected Score']).copy()

            if df.empty:
                logger.warning(f"No valid data left after mapping scores for model {model_name} and trait {trait_name}")
                ax.axis('off')
                continue

            # Set category order
            df['Prompted Score'] = pd.Categorical(df['Prompted Score'], categories=score_categories, ordered=True)
            df['Detected Score'] = pd.Categorical(df['Detected Score'], categories=score_categories, ordered=True)

            # Create confusion matrix
            cm = pd.crosstab(
                df['Prompted Score'],
                df['Detected Score'],
                rownames=['Prompted Score'],
                colnames=['Detected Score'],
                dropna=False
            )
            # Ensure the matrix contains all categories in the desired order
            cm = cm.reindex(index=score_categories, columns=score_categories, fill_value=0)

            # Plot the confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_title(trait_name)
            ax.set_xticklabels(score_categories)
            ax.set_yticklabels(score_categories)

            # Use model_name_mapping for labeling
            mapped_model_name = model_name_mapping.get(model_name, model_name)

            # Set model label on the left in the row
            if t == 0:
                ax.set_ylabel(f"{mapped_model_name}\nPrompted Score", fontsize=14)
            else:
                ax.set_ylabel('')

            # Set trait label at the bottom of the column
            if m == n_rows - 1:
                ax.set_xlabel(f"Detected Score", fontsize=14)
            else:
                ax.set_xlabel('')

    plt.tight_layout()

    # Save the plot in the experiment folder
    output_file = os.path.join(experiment_path, "confusion_matrices.pdf")
    plt.savefig(output_file, bbox_inches='tight')
    logger.info(f"Confusion matrix saved: {output_file}")
    print(f"Confusion matrix saved: {output_file}")

    plt.show()

    # Output a warning about the number of excluded rows
    if missing_scores > 0:
        print(f"Warning: {missing_scores} rows were excluded due to missing 'score' values or 'decision type' being 'Nondistinguishable'.")



def map_trait_score(value):
    """
    Maps 'trait_score' values to categories 'low', 'medium', 'high'.

    Args:
        value (int): The 'trait_score' value.

    Returns:
        str: The category 'low', 'medium', or 'high'.
    """
    if value in [1, 2]:
        return 'low'
    elif value == 3:
        return 'medium'
    elif value in [4, 5]:
        return 'high'
    else:
        return None  # For unforeseen values


def map_detected_score(value):
    """
    Maps 'score' values to categories 'low', 'medium', 'high'.

    Args:
        value (int): The 'score' value.

    Returns:
        str: The category 'low', 'medium', or 'high'.
    """
    if value in [-2, -1]:
        return 'low'
    elif value == 0:
        return 'medium'
    elif value in [1, 2]:
        return 'high'
    else:
        return None  # For unforeseen values


def analyze_text_similarity(df_answers='last', visualization_settings=None):
    """
    Analyzes text similarity between generated answers.

    Args:
        df_answers (str): The experiment folder name or 'last' for the latest.
        visualization_settings (dict): Settings for visualization.

    Returns:
        None
    """
    # Determine the experiment path
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

    text_data = pd.DataFrame()
    for file_path in answer_files:
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            text_data = pd.concat([text_data, df], ignore_index=True)
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            print(f"Error reading file {file_path}: {e}")

    # Check for necessary columns
    required_columns = {'model', 'trait', 'trait_score', 'answer'}
    if not required_columns.issubset(text_data.columns):
        missing_cols = required_columns - set(text_data.columns)
        logger.error(f"Data is missing columns: {', '.join(missing_cols)}")
        print(f"Error: Data is missing columns: {', '.join(missing_cols)}")
        return

    text_data = text_data.reset_index(drop=True)

    # Text preprocessing pipeline
    preprocessing = Pipeline(steps=[
        ('tfidf_vectorizer', TfidfVectorizer()),
        ('svd', TruncatedSVD(n_components=500))
    ])

    # Convert answers to lowercase
    text_data['processed_answer'] = text_data['answer'].apply(lambda x: str(x).lower())

    # Process textual data
    try:
        text_vectors = preprocessing.fit_transform(text_data['processed_answer'])
    except Exception as e:
        logger.error(f"Error during text vectorization: {e}")
        print(f"Error during text vectorization: {e}")
        return

    # Compute cosine distances
    distance_matrix = pairwise_distances(text_vectors, metric='cosine')

    # Find the most similar texts for each text
    most_similar_indices = []
    for i in range(distance_matrix.shape[0]):
        # Exclude the text itself (distance 0 at position i)
        similar_indices = np.argsort(distance_matrix[i])[1:6]  # Top-5 similar texts
        most_similar_indices.append(similar_indices)

    # Collect similarities
    similarities = []
    for i in range(distance_matrix.shape[0]):
        for s in most_similar_indices[i]:
            similarities.append({
                'original_trait': text_data['trait'].iloc[i],
                'original_trait_score': text_data['trait_score'].iloc[i],
                'original_model': text_data['model'].iloc[i],
                'similar_trait': text_data['trait'].iloc[s],
                'similar_trait_score': text_data['trait_score'].iloc[s],
                'similar_model': text_data['model'].iloc[s],
                'similarity': 1 - distance_matrix[i][s]
            })

    similarities_df = pd.DataFrame(similarities)

    # Process similarities
    # For each model and trait, compute average similarities
    similarity_matrix = {}
    labels = []
    traits = text_data['trait'].unique()
    models = text_data['model'].unique()

    for model in models:
        model_similarities = []
        for trait in traits:
            # Filter similarities where the original and similar traits match
            trait_sims = similarities_df[
                (similarities_df['original_model'] == model) &
                (similarities_df['original_trait'] == trait) &
                (similarities_df['similar_trait'] == trait)
            ]
            # Create a matrix where rows are original_trait_score, columns are similar_trait_score
            sim_matrix = trait_sims.pivot_table(
                index='original_trait_score',
                columns='similar_trait_score',
                values='similarity',
                aggfunc='mean'
            )
            # Ensure the matrix is 5x5 (trait scores from 1 to 5)
            sim_matrix = sim_matrix.reindex(index=range(1, 6), columns=range(1, 6))
            sim_matrix = sim_matrix.fillna(0)
            model_similarities.append(sim_matrix.values)
            if trait not in labels:
                labels.append(trait)
        similarity_matrix[model] = np.array(model_similarities)

    # Visualization settings
    output_file = os.path.join(experiment_path, "similarity.pdf")

    # Plot the results
    plot_text_similarity(similarity_matrix, labels, models, output_file, visualization_settings)

def plot_text_similarity(similarity_matrix, labels, models, output_file, visualization_settings):
    """
    Visualizes the text similarity matrices for each model.

    Args:
        similarity_matrix (dict): Dictionary containing similarity matrices for each model.
        labels (list): List of personality trait names.
        models (list): List of model names.
        output_file (str): Path to save the plot.
        visualization_settings (dict): Visualization settings.

    Returns:
        None
    """
    # Configure matplotlib
    rc('font', **{'family': 'serif', 'size': 14})
    # Get model name mapping if specified
    model_name_mapping = visualization_settings.get('model_name_mapping', {})
    # Determine the grid size for subplots
    n_models = len(models)
    n_rows = int(np.ceil(np.sqrt(n_models)))
    n_cols = int(np.ceil(n_models / n_rows))

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6 * n_cols, 5 * n_rows))

    # Ensure axes is a 1D array
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    for idx, model in enumerate(models):
        ax = axes[idx]
        sims = similarity_matrix[model]
        # Assume sims is a 3D array: (num_traits, 5, 5)
        num_traits = sims.shape[0]
        combined_sims = np.mean(sims, axis=0)  # Average over personality traits
        im = ax.imshow(combined_sims, cmap='coolwarm', vmin=0, vmax=1)
        # Set axis labels
        ax.set_xticks(np.arange(5))
        ax.set_yticks(np.arange(5))
        ax.set_xticklabels([1, 2, 3, 4, 5])
        ax.set_yticklabels([1, 2, 3, 4, 5])
        # Add text annotations
        for i in range(5):
            for j in range(5):
                text = ax.text(j, i, f"{combined_sims[i, j]:.2f}",
                               ha="center", va="center", color="black")
        # Set title
        ax.set_title(model_name_mapping.get(model, model))
        ax.set_xlabel('Trait Score')
        ax.set_ylabel('Trait Score')

    # Remove unused subplots
    for idx in range(len(models), len(axes)):
        fig.delaxes(axes[idx])

    fig.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()
    logger.info(f"Text similarity visualization saved to {output_file}")
    print(f"Text similarity visualization saved to {output_file}")
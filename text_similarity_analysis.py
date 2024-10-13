# text_similarity_analysis.py

"""
This module analyzes the similarity of generated texts and visualizes the results.
It calculates pairwise distances between text embeddings and generates similarity matrices.

Users can modify:
- The input files used for analysis.
- The parameters of the text preprocessing pipeline.
- The visualization settings, such as model name mappings and save paths.

"""

import os
import pandas as pd
import numpy as np
import logging
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

logger = logging.getLogger(__name__)

def analyze_text_similarity(text_data, analysis_settings):
    """
    Analyzes the similarity of generated texts and visualizes the results.

    Args:
        text_data (pd.DataFrame or None): DataFrame containing generated texts.
                                          If None, data is loaded from files specified in analysis_settings.
        analysis_settings (dict): Analysis settings, including file paths and visualization parameters.

    Returns:
        None
    """
    # If text_data is not provided, load data from files
    if text_data is None:
        input_files = analysis_settings.get('input_files', [])
        if not input_files:
            logger.error("No input files provided for text similarity analysis.")
            print("Error: No input files provided for text similarity analysis.")
            return
        # Read data from files
        text_data = pd.DataFrame()
        for file_path in input_files:
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                text_data = pd.concat([text_data, df], ignore_index=True)
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                print(f"Error reading file {file_path}: {e}")
    else:
        # Use the provided DataFrame
        pass

    # Check for necessary columns
    required_columns = {'model', 'trait', 'trait_score', 'answer'}
    if not required_columns.issubset(text_data.columns):
        missing_cols = required_columns - set(text_data.columns)
        logger.error(f"Data is missing columns: {', '.join(missing_cols)}")
        print(f"Error: Data is missing columns: {', '.join(missing_cols)}")
        return

    # Reset index
    text_data = text_data.reset_index(drop=True)

    # Text preprocessing pipeline (Users can modify the preprocessing steps and parameters)
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
    save_path = analysis_settings.get('save_path', 'results/text_similarity_analysis')
    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(save_path, "text_similarity_colormap.pdf")

    # Plot the results
    plot_text_similarity(similarity_matrix, labels, models, output_file, analysis_settings)

def plot_text_similarity(similarity_matrix, labels, models, output_file, analysis_settings):
    """
    Visualizes the text similarity matrices for each model.

    Args:
        similarity_matrix (dict): Dictionary containing similarity matrices for each model.
        labels (list): List of personality trait names.
        models (list): List of model names.
        output_file (str): Path to save the plot.
        analysis_settings (dict): Analysis settings, including visualization parameters.

    Returns:
        None
    """
    # Configure matplotlib
    rc('font', **{'family': 'serif', 'size': 14})
    # Get model name mapping if specified
    model_name_mapping = analysis_settings.get('model_name_mapping', {})
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

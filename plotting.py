# plotting.py

"""
This module is responsible for plotting confusion matrices for each model and personality trait.

Users can modify:
- The models and traits included in the analysis by providing different data files.
- The mapping of 'trait_score' and 'score' to categories in 'map_trait_score' and 'map_detected_score' functions.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

logger = logging.getLogger(__name__)

def plot_confusion_matrices(df_analysis, model_names, traits_definitions, analysis_file):
    """
    Plots confusion matrices for each model and personality trait.

    Args:
        df_analysis (pd.DataFrame): DataFrame containing analysis results.
        model_names (list): List of model names.
        traits_definitions (dict): Dictionary containing personality trait definitions.
        analysis_file (str): Path to the analysis file (used for saving plots in the same folder).

    Returns:
        None
    """
    traits = [trait_info['name'] for trait_info in traits_definitions.values()]
    n_rows = len(model_names)
    n_cols = len(traits)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    missing_scores = 0  # Counter for rows with missing 'score' values

    # Ensure axes is a 2D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    score_categories = ['low', 'medium', 'high']  # Define the order of categories once

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

            # Create confusion matrix with specified category order
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

            # Set model label on the left in the row
            if t == 0:
                ax.set_ylabel(f"{model_name}\nPrompted Score", fontsize=14)
            else:
                ax.set_ylabel('')

            # Set trait label at the bottom of the column
            if m == n_rows - 1:
                ax.set_xlabel(f"Detected Score", fontsize=14)
            else:
                ax.set_xlabel('')

    plt.tight_layout()

    # Save the plot
    output_folder = os.path.dirname(analysis_file)
    output_file = os.path.join(output_folder, 'confusion_matrix.png')
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

    Note:
        Users can modify this function to change how 'trait_score' values are mapped to categories.

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

    Note:
        Users can modify this function to change how 'score' values are mapped to categories.

    """
    if value in [-2, -1]:
        return 'low'
    elif value == 0:
        return 'medium'
    elif value in [1, 2]:
        return 'high'
    else:
        return None  # For unforeseen values

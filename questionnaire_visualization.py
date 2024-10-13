# questionnaire_visualization.py

"""
This module visualizes the questionnaire answers.

Users can modify:
- The input files used for visualization.
- The trait and model name mappings.
- The save path for the output plots.

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

def visualize_questionnaire_answers(df_answers_full, visualization_settings):
    """
    Visualizes the questionnaire answers.

    Args:
        df_answers_full (pd.DataFrame): DataFrame containing questionnaire answers. If None, data is loaded from files.
        visualization_settings (dict): Visualization settings, including 'input_files' and 'save_path'.

    Returns:
        None
    """
    # If df_answers_full is not provided, load data from files
    if df_answers_full is None:
        input_files = visualization_settings.get('input_files', [])
        if not input_files:
            logger.error("No input files provided for visualization.")
            print("Error: No input files provided for visualization.")
            return
        # Read data from files
        df_answers_full = pd.DataFrame()
        for file_path in input_files:
            try:
                df = pd.read_csv(file_path)
                df_answers_full = pd.concat([df_answers_full, df], ignore_index=True)
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                print(f"Error reading file {file_path}: {e}")
    else:
        # Use the provided DataFrame
        pass

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

    from matplotlib import rc
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

    # Trait name mapping (Users can modify this mapping)
    trait_name_mapping = visualization_settings.get('trait_name_mapping', {
        'Agreeableness': 'Agreeableness',
        'Conscientiousness': 'Conscientiousness',
        'Extraversion': 'Extraversion',
        'Neuroticism': 'Neuroticism',
        'Openness': 'Openness to experience'
    })

    # Model name mapping (Users can modify this mapping)
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
    save_path = visualization_settings.get('save_path', 'results/questionnaire_visualization')
    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(save_path, "questionnaire_visualization.pdf")
    plt.savefig(output_file, bbox_inches='tight')
    logger.info(f"Questionnaire visualization saved to {output_file}")
    print(f"Questionnaire visualization saved to {output_file}")

    plt.show()

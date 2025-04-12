import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy import stats  # Add explicit import for stats module
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve

# Assuming common_utils.py contains the definitions
# We will pass specific column names as arguments to functions rather than relying on globals here
# But import necessary conversion functions
try:
    from common_utils import beta_to_m, logit_to_beta 
except ImportError:
    print("Warning: common_utils not found. Assuming necessary functions are defined elsewhere or not needed.")
    # Define dummy functions if needed for basic script execution without common_utils
    def beta_to_m(beta, epsilon=1e-6):
        beta_clipped = np.clip(beta, epsilon, 1 - epsilon)
        return np.log2(beta_clipped / (1 - beta_clipped))
    def logit_to_beta(logits):
        odds = np.exp(logits)
        return odds / (1 + odds)

# Style setting for prettier plots
sns.set_theme(style="whitegrid")

def plot_beta_distribution(beta: np.ndarray, 
                           group_labels: pd.Series | np.ndarray | None = None, 
                           group_name: str = "Group",
                           bins: int = 50, 
                           alpha: float = 0.7,
                           ax: plt.Axes | None = None) -> plt.Axes:
    """
    Plots the distribution of β-values, with optional grouping.

    Args:
        beta (np.ndarray): Beta values matrix (samples x CpGs). Assumes values in [0, 1].
        group_labels (pd.Series | np.ndarray | None): Array or Series indicating group membership for each sample.
                                                      If None, plots a single joint histogram. Defaults to None.
        group_name (str): Name for the grouping variable (for legend title). Defaults to "Group".
        bins (int): Number of bins for the histogram/density plot. Defaults to 50.
        alpha (float): Transparency of the histogram bars/KDE curves. Defaults to 0.7.
        ax (plt.Axes | None): Matplotlib axes object to plot on. If None, creates a new figure and axes. Defaults to None.

    Returns:
        plt.Axes: The Matplotlib axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # If no group labels provided, plot a single histogram
    if group_labels is None:
        sns.histplot(data=beta.flatten(), bins=bins, stat='density', kde=True, 
                     color='steelblue', alpha=alpha, ax=ax)
        ax.set_title('Distribution of Beta Values')
        
    else:
        # Group-based plotting
        if not isinstance(group_labels, pd.Series):
            group_labels = pd.Series(group_labels, name=group_name)

        unique_groups = group_labels.unique()
        
        # Choose more contrasting colors based on number of groups
        if len(unique_groups) == 2:
            # High contrast for binary groups
            palette = ['darkblue', 'crimson']
        elif len(unique_groups) <= 5:
            # Distinct colors for small number of groups
            palette = ['darkblue', 'crimson', 'forestgreen', 'darkorange', 'purple'][:len(unique_groups)]
        else:
            # For many groups, use a high-contrast palette
            palette = 'Set1'
        
        plot_data = pd.DataFrame({
            'Sample': np.repeat(np.arange(beta.shape[0]), beta.shape[1]),
            'Beta': beta.flatten(),
            group_name: np.repeat(group_labels.values, beta.shape[1])
        })

        sns.histplot(data=plot_data, x='Beta', hue=group_name, 
                     bins=bins, stat='density', common_norm=False, kde=True, 
                     palette=palette, alpha=alpha, ax=ax)
        
        ax.set_title('Distribution of Beta Values by Group')
        ax.legend(title=group_name)
    
    ax.set_xlabel('Beta Value')
    ax.set_ylabel('Density')
    
    return ax

def plot_coverage_distribution(depth: np.ndarray, 
                               group_labels: pd.Series | np.ndarray | None = None, 
                               group_name: str = "Group",
                               log_scale: bool = True,
                               alpha: float = 0.7,
                               ax: plt.Axes | None = None) -> plt.Axes:
    """
    Plots the distribution of per-sample mean sequencing depth, with optional grouping.

    Args:
        depth (np.ndarray): Sequencing depth matrix (samples x CpGs).
        group_labels (pd.Series | np.ndarray | None): Array or Series indicating group membership for each sample.
                                                      If None, plots a single distribution. Defaults to None.
        group_name (str): Name for the grouping variable (for legend/axis title). Defaults to "Group".
        log_scale (bool): Whether to plot the y-axis (mean depth) on a log scale. Defaults to True.
        alpha (float): Transparency of the boxplot elements. Defaults to 0.7.
        ax (plt.Axes | None): Matplotlib axes object to plot on. If None, creates a new figure and axes. Defaults to None.

    Returns:
        plt.Axes: The Matplotlib axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    mean_depth_per_sample = np.mean(depth, axis=1)
    
    # If no group labels provided, plot a single distribution
    if group_labels is None:
        # Create a vertical plot with no x-categories
        sns.violinplot(y=mean_depth_per_sample, color='steelblue', alpha=alpha, ax=ax)
        
        # Alternative: could also use a histogram
        # sns.histplot(mean_depth_per_sample, kde=True, color='steelblue', alpha=alpha, ax=ax)
        
        ax.set_title('Distribution of Mean Sample Sequencing Depth')
        ax.set_xlabel('')  # No x label needed for single distribution
    else:
        # Group-based plotting with boxplots
        if not isinstance(group_labels, pd.Series):
            group_labels = pd.Series(group_labels, name=group_name)
        
        # Choose more contrasting colors based on number of groups
        unique_groups = group_labels.unique()
        if len(unique_groups) == 2:
            # High contrast for binary groups
            palette = ['darkblue', 'crimson']
        elif len(unique_groups) <= 5:
            # Distinct colors for small number of groups
            palette = ['darkblue', 'crimson', 'forestgreen', 'darkorange', 'purple'][:len(unique_groups)]
        else:
            # For many groups, use a high-contrast palette
            palette = 'Set1'
            
        plot_df = pd.DataFrame({
            'Mean Depth': mean_depth_per_sample,
            group_name: group_labels
        })

        sns.boxplot(data=plot_df, x=group_name, y='Mean Depth', palette=palette, alpha=alpha, ax=ax)
        
        ax.set_title('Distribution of Mean Sample Sequencing Depth by Group')
        ax.set_xlabel(group_name)
        
        # Improve x-tick label readability if many groups
        if len(plot_df[group_name].unique()) > 5:
            ax.tick_params(axis='x', rotation=45)
    
    # Common settings for both cases
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('Mean Depth (log scale)')
    else:
        ax.set_ylabel('Mean Depth')

    return ax

def plot_pca(beta: np.ndarray, 
             clinical_df: pd.DataFrame, 
             color_by: str, 
             pc_x: int = 1, 
             pc_y: int = 2, 
             use_m_values: bool = True,
             scale_data: bool = False,
             annotate_samples: list | None = None,
             ax: plt.Axes | None = None) -> tuple[plt.Axes, PCA]:
    """
    Performs PCA on methylation data and plots the results, colored by a clinical variable.

    Args:
        beta (np.ndarray): Beta values matrix (samples x CpGs). Assumes values in [0, 1].
        clinical_df (pd.DataFrame): DataFrame with clinical data. Must be indexed consistently 
                                    with the rows of the beta matrix (e.g., by sample ID).
        color_by (str): Column name in clinical_df to use for coloring points.
        pc_x (int): Principal component to plot on the x-axis (1-based index). Defaults to 1.
        pc_y (int): Principal component to plot on the y-axis (1-based index). Defaults to 2.
        use_m_values (bool): Convert beta values to M-values before PCA. Defaults to True.
        scale_data (bool): Standardize the data (mean=0, variance=1) before PCA. Defaults to False.
                           Often recommended if use_m_values=False or if features have very different scales.
        annotate_samples (list | None): Optional list of sample IDs (matching clinical_df index) 
                                        to annotate on the plot. Defaults to None.
        ax (plt.Axes | None): Matplotlib axes object to plot on. If None, creates a new figure and axes. Defaults to None.

    Returns:
        tuple[plt.Axes, PCA]: The Matplotlib axes object and the fitted PCA object.
    """
    # Only create a new figure if ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        
    if beta.shape[0] != len(clinical_df):
        raise ValueError("Number of samples in beta matrix must match number of rows in clinical_df.")
    if color_by not in clinical_df.columns:
         raise ValueError(f"Coloring variable '{color_by}' not found in clinical_df columns.")
    if pc_x <= 0 or pc_y <= 0:
        raise ValueError("Principal component indices must be positive integers.")

    # Prepare data for PCA
    data_for_pca = beta
    if use_m_values:
        # Don't use print as it can interfere with notebook displays
        # print("Converting beta to M-values for PCA...")
        data_for_pca = beta_to_m(beta)
        # Replace any potential +/- inf resulting from extreme beta values near 0 or 1 after clipping in beta_to_m
        data_for_pca[np.isinf(data_for_pca)] = np.nan 
        # Simple imputation: replace NaN with column mean - more sophisticated methods could be used
        col_means = np.nanmean(data_for_pca, axis=0)
        inds = np.where(np.isnan(data_for_pca))
        data_for_pca[inds] = np.take(col_means, inds[1])

    if scale_data:
        # Don't use print as it can interfere with notebook displays
        # print("Scaling data before PCA...")
        scaler = StandardScaler()
        data_for_pca = scaler.fit_transform(data_for_pca)

    # Perform PCA
    n_components = max(pc_x, pc_y)
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data_for_pca)
    
    variance_explained = pca.explained_variance_ratio_ * 100

    pca_df = pd.DataFrame(pca_result, 
                          columns=[f'PC{i+1}' for i in range(n_components)],
                          index=clinical_df.index)
    pca_df = pd.concat([pca_df, clinical_df[[color_by]]], axis=1)

    # Choose more contrasting colors for better visibility
    if len(pca_df[color_by].unique()) == 2:
        palette = ['darkblue', 'crimson']
    elif len(pca_df[color_by].unique()) <= 5:
        palette = ['darkblue', 'crimson', 'forestgreen', 'darkorange', 'purple'][:len(pca_df[color_by].unique())]
    else:
        palette = 'Set1'
    
    # Plotting
    sns.scatterplot(data=pca_df, x=f'PC{pc_x}', y=f'PC{pc_y}', hue=color_by, 
                   palette=palette, s=50, alpha=0.8, ax=ax)
    
    ax.set_title(f'PCA Plot ({color_by})')
    ax.set_xlabel(f'PC{pc_x} ({variance_explained[pc_x-1]:.2f}% variance)')
    ax.set_ylabel(f'PC{pc_y} ({variance_explained[pc_y-1]:.2f}% variance)')
    ax.legend(title=color_by)

    # Annotate specified samples
    if annotate_samples:
        for sample_id in annotate_samples:
            if sample_id in pca_df.index:
                x_coord = pca_df.loc[sample_id, f'PC{pc_x}']
                y_coord = pca_df.loc[sample_id, f'PC{pc_y}']
                ax.text(x_coord, y_coord, sample_id, fontsize=9, ha='right', va='bottom')
            else:
                print(f"Warning: Sample ID '{sample_id}' for annotation not found in data.")
                
    # Don't call plt.tight_layout() here as it affects the global figure state
    
    return ax, pca

def plot_tsne(beta: np.ndarray, 
              clinical_df: pd.DataFrame, 
              color_by: str, 
              use_m_values: bool = True,
              n_components: int = 2,
              perplexity: float = 30.0,
              learning_rate: float = 'auto',
              metric: str = 'euclidean',
              random_state: int = 42,
              annotate_samples: list | None = None,
              ax: plt.Axes | None = None) -> tuple[plt.Axes, TSNE]:
    """
    Performs t-SNE on methylation data and plots the 2D embedding, colored by a clinical variable.

    Args:
        beta (np.ndarray): Beta values matrix (samples x CpGs). Assumes values in [0, 1].
        clinical_df (pd.DataFrame): DataFrame with clinical data. Must be indexed consistently 
                                    with the rows of the beta matrix (e.g., by sample ID).
        color_by (str): Column name in clinical_df to use for coloring points.
        use_m_values (bool): Convert beta values to M-values before t-SNE. Defaults to True.
        n_components (int): Dimension of the embedded space. Defaults to 2.
        perplexity (float): The perplexity is related to the number of nearest neighbors used. 
                           Larger datasets usually require larger perplexity. Defaults to 30.0.
        learning_rate (float or str): Learning rate for t-SNE optimization. Defaults to 'auto'.
        metric (str): Distance metric used. Defaults to 'euclidean'.
        random_state (int): Random seed for reproducibility. Defaults to 42.
        annotate_samples (list | None): Optional list of sample IDs (matching clinical_df index) 
                                        to annotate on the plot. Defaults to None.
        ax (plt.Axes | None): Matplotlib axes object to plot on. If None, creates a new figure and axes. Defaults to None.

    Returns:
        tuple[plt.Axes, TSNE]: The Matplotlib axes object and the fitted t-SNE reducer object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    if beta.shape[0] != len(clinical_df):
        raise ValueError("Number of samples in beta matrix must match number of rows in clinical_df.")
    if color_by not in clinical_df.columns:
        raise ValueError(f"Coloring variable '{color_by}' not found in clinical_df columns.")

    # Prepare data for t-SNE
    data_for_tsne = beta
    if use_m_values:
        # Don't use print as it can interfere with notebook displays
        # print("Converting beta to M-values for t-SNE...")
        data_for_tsne = beta_to_m(beta)
        # Handle potential infinities as in PCA
        data_for_tsne[np.isinf(data_for_tsne)] = np.nan
        col_means = np.nanmean(data_for_tsne, axis=0)
        inds = np.where(np.isnan(data_for_tsne))
        data_for_tsne[inds] = np.take(col_means, inds[1])

    # Perform t-SNE
    reducer = TSNE(n_components=n_components, 
                  perplexity=perplexity, 
                  learning_rate=learning_rate, 
                  metric=metric, 
                  random_state=random_state)
    embedding = reducer.fit_transform(data_for_tsne)

    tsne_df = pd.DataFrame(embedding, columns=['TSNE1', 'TSNE2'], index=clinical_df.index)
    tsne_df = pd.concat([tsne_df, clinical_df[[color_by]]], axis=1)

    # Choose more contrasting colors for better visibility
    if len(tsne_df[color_by].unique()) == 2:
        palette = ['darkblue', 'crimson']
    elif len(tsne_df[color_by].unique()) <= 5:
        palette = ['darkblue', 'crimson', 'forestgreen', 'darkorange', 'purple'][:len(tsne_df[color_by].unique())]
    else:
        palette = 'Set1'

    # Plotting
    sns.scatterplot(data=tsne_df, x='TSNE1', y='TSNE2', hue=color_by, 
                    palette=palette, s=50, alpha=0.8, ax=ax)
    
    ax.set_title(f't-SNE Plot ({color_by})')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.legend(title=color_by)
    ax.set_xticks([]) # t-SNE axes don't have inherent meaning
    ax.set_yticks([])

    # Annotate specified samples
    if annotate_samples:
        for sample_id in annotate_samples:
            if sample_id in tsne_df.index:
                x_coord = tsne_df.loc[sample_id, 'TSNE1']
                y_coord = tsne_df.loc[sample_id, 'TSNE2']
                ax.text(x_coord, y_coord, sample_id, fontsize=9, ha='right', va='bottom')
            else:
                print(f"Warning: Sample ID '{sample_id}' for annotation not found in data.")

    # Don't call plt.tight_layout() here as it affects the global figure state

    return ax, reducer

def plot_volcano(results_df: pd.DataFrame, 
                 effect_size_col: str, 
                 p_value_col: str, 
                 alpha: float = 0.05, 
                 effect_size_threshold: float | None = None,
                 label_top_n: int = 0,
                 ax: plt.Axes | None = None) -> plt.Axes:
    """
    Creates a volcano plot for differential analysis results.

    Args:
        results_df (pd.DataFrame): DataFrame containing differential analysis results. 
                                   Must include columns for effect size and p-value/FDR.
        effect_size_col (str): Name of the column containing effect sizes (e.g., 'logFC', 'delta_beta').
        p_value_col (str): Name of the column containing p-values or FDR values.
        alpha (float): Significance threshold (e.g., for FDR). Points below this value are highlighted. Defaults to 0.05.
        effect_size_threshold (float | None): Optional threshold for effect size magnitude. If provided, points 
                                            exceeding this magnitude AND below alpha are highlighted differently. 
                                            Defaults to None.
        label_top_n (int): Annotate the top N most significant points (lowest p-value). Requires index to be meaningful (e.g. CpG ID). Defaults to 0.
        ax (plt.Axes | None): Matplotlib axes object to plot on. If None, creates a new figure and axes. Defaults to None.

    Returns:
        plt.Axes: The Matplotlib axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    if effect_size_col not in results_df.columns:
        raise ValueError(f"Effect size column '{effect_size_col}' not found in results_df.")
    if p_value_col not in results_df.columns:
         raise ValueError(f"P-value column '{p_value_col}' not found in results_df.")

    # Prepare data for plotting
    plot_df = results_df.copy()
    plot_df['-log10(p)'] = -np.log10(plot_df[p_value_col])
    
    # Replace inf/-inf which can result from p=0
    max_log_p = np.nanmax(plot_df['-log10(p)'][np.isfinite(plot_df['-log10(p)'])])
    if max_log_p is np.nan: # Handle case where all p-values might be 0 or NaN
         max_log_p = 10 # Assign an arbitrary large value
    plot_df['-log10(p)'].replace([np.inf, -np.inf], max_log_p * 1.1, inplace=True) # Replace with slightly larger than max finite
    plot_df['-log10(p)'].fillna(0, inplace=True) # Replace NaN p-values with 0


    # Determine significance status
    plot_df['Significance'] = 'Not Significant'
    significant_mask = plot_df[p_value_col] < alpha
    
    if effect_size_threshold is not None:
         effect_mask = np.abs(plot_df[effect_size_col]) > effect_size_threshold
         plot_df.loc[significant_mask & effect_mask, 'Significance'] = f'Significant (p<{alpha}, |Effect|>{effect_size_threshold})'
         plot_df.loc[significant_mask & ~effect_mask, 'Significance'] = f'Significant (p<{alpha})'
    else:
        plot_df.loc[significant_mask, 'Significance'] = f'Significant (p<{alpha})'

    # Define colors
    palette = {
        'Not Significant': 'grey', 
        f'Significant (p<{alpha})': 'orange', 
        f'Significant (p<{alpha}, |Effect|>{effect_size_threshold})': 'red'
    }
    if effect_size_threshold is None:
         palette.pop(f'Significant (p<{alpha}, |Effect|>{effect_size_threshold})') # Remove unused key

    # Plotting
    sns.scatterplot(data=plot_df, x=effect_size_col, y='-log10(p)', 
                    hue='Significance', palette=palette, 
                    s=20, alpha=0.6, ax=ax, legend='brief') # Use brief legend if names get long

    # Add significance line
    ax.axhline(-np.log10(alpha), color='grey', linestyle='--', linewidth=1)
    
    # Add effect size lines if threshold provided
    if effect_size_threshold is not None:
        ax.axvline(effect_size_threshold, color='grey', linestyle='--', linewidth=1)
        ax.axvline(-effect_size_threshold, color='grey', linestyle='--', linewidth=1)

    ax.set_title('Volcano Plot of Differential Methylation')
    ax.set_xlabel(f'Effect Size ({effect_size_col})')
    ax.set_ylabel(f'-log10({p_value_col})')
    
    # Annotate top N points by p-value
    if label_top_n > 0:
        top_n_df = plot_df.nsmallest(label_top_n, p_value_col)
        for idx, row in top_n_df.iterrows():
            ax.text(row[effect_size_col], row['-log10(p)'], str(idx), fontsize=8, ha='left', va='bottom')

    # Adjust legend position without calling plt.tight_layout
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    
    # Don't use plt.tight_layout here as it affects the global figure state
    # plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Instead, adjust the axes directly if needed
    if ax.get_figure() is not None:  # Make sure the figure exists
        fig = ax.get_figure()
        fig.subplots_adjust(right=0.75)  # Make room for the legend

    return ax

def plot_methylation_vs_covariate(beta: np.ndarray, 
                                  clinical_df: pd.DataFrame, 
                                  cpg_index: int, 
                                  covariate: str, 
                                  cpg_id: str | None = None,
                                  add_regression: bool = True,
                                  ax: plt.Axes | None = None) -> plt.Axes:
    """
    Plots methylation of a specific CpG site against a continuous clinical covariate.

    Args:
        beta (np.ndarray): Beta values matrix (samples x CpGs). Assumes values in [0, 1].
        clinical_df (pd.DataFrame): DataFrame with clinical data. Must be indexed consistently 
                                    with the rows of the beta matrix.
        cpg_index (int): The column index of the CpG site in the beta matrix to plot.
        covariate (str): The column name in clinical_df for the continuous covariate.
        cpg_id (str | None): Optional ID or name for the CpG site (for title). Defaults to None.
        add_regression (bool): Whether to add a linear regression line to the plot. Defaults to True.
        ax (plt.Axes | None): Matplotlib axes object to plot on. If None, creates a new figure and axes. Defaults to None.

    Returns:
        plt.Axes: The Matplotlib axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if beta.shape[0] != len(clinical_df):
        raise ValueError("Number of samples in beta matrix must match number of rows in clinical_df.")
    if covariate not in clinical_df.columns:
        raise ValueError(f"Covariate '{covariate}' not found in clinical_df columns.")
    if cpg_index < 0 or cpg_index >= beta.shape[1]:
        raise IndexError(f"cpg_index {cpg_index} is out of bounds for beta matrix with {beta.shape[1]} CpGs.")

    # Prepare data
    plot_df = pd.DataFrame({
        'Methylation': beta[:, cpg_index],
        covariate: clinical_df[covariate]
    }, index=clinical_df.index)

    # Plotting
    plot_func = sns.regplot if add_regression else sns.scatterplot
    plot_func(data=plot_df, x=covariate, y='Methylation', 
              scatter_kws={'s': 30, 'alpha': 0.7}, 
              line_kws={'color': 'red'}, 
              ax=ax)

    plot_title = f'Methylation vs. {covariate}'
    if cpg_id:
        plot_title = f'{cpg_id} {plot_title}'
    else:
        plot_title = f'CpG Index {cpg_index} {plot_title}'
        
    ax.set_title(plot_title)
    ax.set_xlabel(covariate)
    ax.set_ylabel('Beta Value')

    return ax

def plot_region_methylation(cpg_indices: list[int], 
                            beta: np.ndarray, 
                            clinical_df: pd.DataFrame, 
                            group_var: str, 
                            region_name: str | None = None,
                            plot_type: str = 'box',
                            ax: plt.Axes | None = None) -> plt.Axes:
    """
    Plots methylation levels within a genomic region (defined by a set of CpG indices) 
    stratified by a clinical group. Plots the average beta value per sample for the region.

    Args:
        cpg_indices (list[int]): List of column indices in the beta matrix corresponding to the region.
        beta (np.ndarray): Beta values matrix (samples x CpGs). Assumes values in [0, 1].
        clinical_df (pd.DataFrame): DataFrame with clinical data. Must be indexed consistently 
                                    with the rows of the beta matrix.
        group_var (str): Column name in clinical_df to use for grouping (e.g., 'disease_status').
        region_name (str | None): Optional name for the region (for title). Defaults to None.
        plot_type (str): Type of plot to generate ('box', 'violin', 'strip'). Defaults to 'box'.
        ax (plt.Axes | None): Matplotlib axes object to plot on. If None, creates a new figure and axes. Defaults to None.

    Returns:
        plt.Axes: The Matplotlib axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if beta.shape[0] != len(clinical_df):
        raise ValueError("Number of samples in beta matrix must match number of rows in clinical_df.")
    if group_var not in clinical_df.columns:
        raise ValueError(f"Grouping variable '{group_var}' not found in clinical_df columns.")
    if not cpg_indices:
        raise ValueError("cpg_indices list cannot be empty.")
    if any(idx < 0 or idx >= beta.shape[1] for idx in cpg_indices):
        raise IndexError(f"cpg_indices contains indices out of bounds for beta matrix with {beta.shape[1]} CpGs.")

    # Calculate mean beta per sample for the specified region
    region_beta = beta[:, cpg_indices]
    mean_region_beta = np.mean(region_beta, axis=1)

    # Prepare data
    plot_df = pd.DataFrame({
        'Mean Region Beta': mean_region_beta,
        group_var: clinical_df[group_var]
    }, index=clinical_df.index)

    # Plotting
    if plot_type == 'box':
        sns.boxplot(data=plot_df, x=group_var, y='Mean Region Beta', palette='viridis', ax=ax)
    elif plot_type == 'violin':
         sns.violinplot(data=plot_df, x=group_var, y='Mean Region Beta', palette='viridis', ax=ax, inner='quartile')
    elif plot_type == 'strip':
        sns.stripplot(data=plot_df, x=group_var, y='Mean Region Beta', palette='viridis', ax=ax, jitter=True, alpha=0.7)
        # Optionally add boxplot overlay for clarity
        sns.boxplot(data=plot_df, x=group_var, y='Mean Region Beta', palette=['white']*plot_df[group_var].nunique(), ax=ax, showfliers=False, showcaps=False, boxprops={'alpha':0})

    else:
        raise ValueError("plot_type must be one of 'box', 'violin', or 'strip'.")

    plot_title = f'Mean Methylation in Region'
    if region_name:
        plot_title += f' ({region_name})'
    plot_title += f' by {group_var}'
    
    ax.set_title(plot_title)
    ax.set_xlabel(group_var)
    ax.set_ylabel('Mean Beta Value in Region')
    
    # Improve x-tick label readability if many groups
    if len(plot_df[group_var].unique()) > 5:
        ax.tick_params(axis='x', rotation=45)

    return ax

def calculate_clopper_pearson_ci(x, n, alpha=0.05):
    """
    Calculate the Clopper-Pearson exact confidence interval for a binomial proportion.
    
    Args:
        x (int): Number of successes.
        n (int): Number of trials.
        alpha (float): Significance level (default: 0.05 for 95% CI).
    
    Returns:
        tuple: Lower and upper bounds of the confidence interval.
    """
    if n == 0:
        return 0.0, 1.0
    
    # Special case if x is 0 or n
    if x == 0:
        lower = 0.0
        upper = 1.0 - (alpha/2)**(1/n)
    elif x == n:
        lower = (alpha/2)**(1/n)
        upper = 1.0
    else:
        lower = stats.beta.ppf(alpha/2, x, n-x+1)
        upper = stats.beta.ppf(1-alpha/2, x+1, n-x)
    
    return lower, upper

def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculates classification performance metrics.
    
    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted class labels.
        y_prob (np.ndarray, optional): Predicted probabilities for the positive class.
                                      Required for AUROC and sensitivity at specificity.
    
    Returns:
        dict: Dictionary containing performance metrics.
    """
    from sklearn.metrics import (roc_auc_score, precision_score, recall_score, 
                               f1_score, confusion_matrix, precision_recall_curve)
    
    metrics = {}
    
    # Calculate basic metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['ppv'] = metrics['precision']  # PPV is the same as precision
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    metrics['sensitivity'] = metrics['recall']  # Sensitivity is the same as recall
    
    # Calculate metrics that require probabilities
    if y_prob is not None:
        metrics['auroc'] = roc_auc_score(y_true, y_prob)
        
        # Calculate sensitivity at specific specificities
        specificities = [0.90, 0.95, 0.99]
        for spec in specificities:
            # Calculate sensitivity at specificity
            # This requires finding the right threshold
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            idx = np.argmin(np.abs(1 - spec - fpr))
            sens_at_spec = tpr[idx]
            metrics[f'sensitivity_at_{int(spec*100)}spec'] = sens_at_spec
            
            # Calculate the number of true positives and actual positives at this threshold
            threshold = thresholds[idx]
            y_pred_at_threshold = (y_prob >= threshold).astype(int)
            positives = np.sum(y_true == 1)
            true_positives = np.sum((y_true == 1) & (y_pred_at_threshold == 1))
            
            # Calculate Clopper-Pearson CI
            if positives > 0:
                lower_ci, upper_ci = calculate_clopper_pearson_ci(true_positives, positives)
                metrics[f'sensitivity_at_{int(spec*100)}spec_lower_ci'] = lower_ci
                metrics[f'sensitivity_at_{int(spec*100)}spec_upper_ci'] = upper_ci
            else:
                metrics[f'sensitivity_at_{int(spec*100)}spec_lower_ci'] = np.nan
                metrics[f'sensitivity_at_{int(spec*100)}spec_upper_ci'] = np.nan
    
    return metrics

def plot_roc_curve(y_true, y_prob, label=None, ax=None, show_sens_at_spec=None):
    """
    Plots the ROC curve.
    
    Args:
        y_true (np.ndarray): Ground truth labels.
        y_prob (np.ndarray): Predicted probabilities for the positive class.
        label (str, optional): Label for the ROC curve. Default is None.
        ax (plt.Axes, optional): Axes to plot on. Default is None.
        show_sens_at_spec (float, optional): Specificity value at which to show sensitivity with CI.
                                            Default is None.
    
    Returns:
        plt.Axes: The axes containing the plot.
    """
    from sklearn.metrics import roc_curve, auc
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    curve_label = f'{label} (AUC = {roc_auc:.3f})' if label else f'AUC = {roc_auc:.3f}'
    ax.plot(fpr, tpr, lw=2, label=curve_label)
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    
    # If requested, show sensitivity at a specific specificity with CI
    if show_sens_at_spec is not None:
        target_spec = show_sens_at_spec
        target_fpr = 1 - target_spec
        
        # Find the closest point to the target specificity
        idx = np.argmin(np.abs(fpr - target_fpr))
        sens_at_spec = tpr[idx]
        spec_at_point = 1 - fpr[idx]
        
        # Calculate Clopper-Pearson CI
        positives = np.sum(y_true == 1)
        threshold = thresholds[idx]
        y_pred_at_threshold = (y_prob >= threshold).astype(int)
        true_positives = np.sum((y_true == 1) & (y_pred_at_threshold == 1))
        
        lower_ci, upper_ci = calculate_clopper_pearson_ci(true_positives, positives)
        
        # Plot the point with error bars for CI
        ax.plot(1-spec_at_point, sens_at_spec, 'ro', ms=8, label=f'Sensitivity at {target_spec:.2f} specificity')
        ax.errorbar(1-spec_at_point, sens_at_spec, yerr=[[sens_at_spec-lower_ci], [upper_ci-sens_at_spec]], 
                  fmt='none', ecolor='r', capsize=5)
        
        # Add text annotation with CI
        ax.annotate(f'Sens: {sens_at_spec:.3f} ({lower_ci:.3f}-{upper_ci:.3f})', 
                  xy=(1-spec_at_point, sens_at_spec), 
                  xytext=(1-spec_at_point+0.1, sens_at_spec-0.1),
                  arrowprops=dict(arrowstyle='->', color='black'),
                  bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")
    
    return ax

def plot_precision_recall_curve(y_true, y_prob, label=None, ax=None):
    """
    Plots the Precision-Recall curve.
    
    Args:
        y_true (np.ndarray): Ground truth labels.
        y_prob (np.ndarray): Predicted probabilities for the positive class.
        label (str, optional): Label for the PR curve. Default is None.
        ax (plt.Axes, optional): Axes to plot on. Default is None.
    
    Returns:
        plt.Axes: The axes containing the plot.
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    # Plot precision-recall curve
    curve_label = f'{label} (AP = {avg_precision:.3f})' if label else f'AP = {avg_precision:.3f}'
    ax.step(recall, precision, where='post', lw=2, label=curve_label)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="best")
    
    return ax

def plot_metrics_comparison(metrics_list, model_names, ax=None):
    """
    Creates a bar chart comparing metrics across different models.
    
    Args:
        metrics_list (list): List of metric dictionaries from calculate_metrics.
        model_names (list): Names of the models corresponding to each metrics dict.
        ax (plt.Axes, optional): Axes to plot on. Default is None.
    
    Returns:
        plt.Axes: The axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Select which metrics to compare
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'specificity']
    
    # Create DataFrame for plotting
    plot_data = {metric: [m.get(metric, np.nan) for m in metrics_list] for metric in metrics_to_plot}
    plot_df = pd.DataFrame(plot_data, index=model_names)
    
    # Plot
    plot_df.plot(kind='bar', ax=ax)
    ax.set_ylim([0, 1.05])
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics Comparison')
    ax.legend(title='Metric')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return ax

def plot_scree(eigenvalues=None, 
             explained_variance_ratio=None, 
             cumulative_explained_variance=None,
             detect_elbow=True,
             beta_values=None,
             use_m_values=True,
             n_components=None,
             ax=None) -> tuple[plt.Axes, int | None]:
    """
    Creates a scree plot to visualize eigenvalues from PCA and optionally detects the elbow point.
    
    Args:
        eigenvalues (np.ndarray, optional): Array of eigenvalues from PCA. If None, beta_values must be provided.
        explained_variance_ratio (np.ndarray, optional): Array of explained variance ratios. Required if eigenvalues are provided.
        cumulative_explained_variance (np.ndarray, optional): Array of cumulative explained variance. Required if eigenvalues are provided.
        detect_elbow (bool): Whether to detect and mark the elbow point. Default is True.
        beta_values (np.ndarray, optional): Beta values matrix to calculate PCA if eigenvalues not provided.
        use_m_values (bool): Whether to use M-values when calculating from beta_values. Default is True.
        n_components (int, optional): Number of components to show in plot. Default is None (all components).
        ax (plt.Axes, optional): Matplotlib axes object to plot on. Default is None.
        
    Returns:
        tuple[plt.Axes, int | None]: The Matplotlib axes object and the detected elbow point index (or None if not detected).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # If eigenvalues not provided, calculate from beta_values
    if eigenvalues is None:
        if beta_values is None:
            raise ValueError("Either eigenvalues or beta_values must be provided.")
        
        # Import the function from models
        try:
            from models import calculate_scree_plot_values
            scree_data = calculate_scree_plot_values(
                beta_values, use_m_values=use_m_values, n_components=n_components
            )
            eigenvalues = scree_data['eigenvalues']
            explained_variance_ratio = scree_data['explained_variance_ratio']
            cumulative_explained_variance = scree_data['cumulative_explained_variance']
        except ImportError:
            raise ImportError("Could not import calculate_scree_plot_values from models.")
    
    # Limit number of components to display if specified
    if n_components is not None and n_components < len(eigenvalues):
        eigenvalues = eigenvalues[:n_components]
        explained_variance_ratio = explained_variance_ratio[:n_components]
        cumulative_explained_variance = cumulative_explained_variance[:n_components]
    
    # Component indices (1-based for display)
    components = np.arange(1, len(eigenvalues) + 1)
    
    # Create figure with two y-axes
    ax2 = ax.twinx()
    
    # Plot eigenvalues as bars
    ax.bar(components, eigenvalues, alpha=0.7, color='steelblue', label='Eigenvalues')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Eigenvalue')
    
    # Plot cumulative explained variance as line
    ax2.plot(components, cumulative_explained_variance * 100, 'r-', marker='o', ms=4, label='Cumulative Explained Variance')
    ax2.set_ylabel('Cumulative Explained Variance (%)')
    ax2.set_ylim([0, 105])
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Detect elbow point if requested
    elbow_idx = None
    if detect_elbow and len(eigenvalues) > 2:
        # Find point of maximum curvature in eigenvalues
        # Normalize the data for better curvature detection
        x = components.copy()
        y = eigenvalues.copy()
        
        # Normalize to [0, 1] range
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        
        # Calculate curvature: curvature = y'' / (1 + y'^2)^(3/2)
        # First, calculate first and second derivatives
        dy = np.gradient(y, x)
        d2y = np.gradient(dy, x)
        
        # Calculate curvature
        curvature = np.abs(d2y) / (1 + dy**2)**(1.5)
        
        # Find point of maximum curvature, skipping first point
        elbow_idx = np.argmax(curvature[1:]) + 1
        
        # Mark the elbow point on the plot
        ax.axvline(x=components[elbow_idx], color='green', linestyle='--', linewidth=1.5)
        ax.scatter(components[elbow_idx], eigenvalues[elbow_idx], 
                color='green', s=100, zorder=5, marker='*', label=f'Elbow Point (PC{components[elbow_idx]})')
        
        # Add annotation for variance explained up to elbow
        variance_at_elbow = cumulative_explained_variance[elbow_idx] * 100
        ax.annotate(f'Elbow at PC{components[elbow_idx]}\n{variance_at_elbow:.1f}% variance explained',
                  xy=(components[elbow_idx], eigenvalues[elbow_idx]),
                  xytext=(components[elbow_idx] + 1, eigenvalues[elbow_idx]),
                  arrowprops=dict(arrowstyle='->', color='black'),
                  bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    # Create combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title('Scree Plot with Cumulative Explained Variance')
    plt.tight_layout()
    
    return ax, elbow_idx

# Example Usage Placeholder (commented out - requires data)
if __name__ == '__main__':
    print("Plotting functions defined. To use them, provide appropriate data.")
    # # --- Conceptual Example ---
    # # Assume beta_values (samples x CpGs), depth_values (samples x CpGs), 
    # # clinical_data (DataFrame with sample index), results (DataFrame from diff analysis) are loaded
    
    # # Example Data (replace with actual data loading)
    # n_samples = 100
    # n_cpgs = 1000
    # beta_values = np.random.rand(n_samples, n_cpgs)
    # depth_values = np.random.randint(5, 100, size=(n_samples, n_cpgs))
    # clinical_data = pd.DataFrame({
    #     'sample_id': [f'S{i+1}' for i in range(n_samples)],
    #     'disease_status': np.random.choice(['Control', 'Case'], n_samples),
    #     'age': np.random.randint(30, 80, n_samples),
    #     'sex': np.random.choice(['M', 'F'], n_samples)
    # }).set_index('sample_id')
    # results_df_example = pd.DataFrame({
    #      'CpG_ID': [f'cg{i+1}' for i in range(n_cpgs)],
    #      'delta_beta': np.random.normal(0, 0.1, n_cpgs),
    #      'FDR': np.random.uniform(0, 1, n_cpgs) ** 2 # Skew towards lower p-values
    # }).set_index('CpG_ID')
    # results_df_example['-log10(FDR)'] = -np.log10(results_df_example['FDR'])


    # # --- Function Calls ---
    # plt.figure() 
    # ax1 = plot_beta_distribution(beta_values, clinical_data['disease_status'], group_name='Disease Status')
    # plt.tight_layout()
    
    # plt.figure()
    # ax2 = plot_coverage_distribution(depth_values, clinical_data['disease_status'], group_name='Disease Status')
    # plt.tight_layout()

    # plt.figure()
    # ax3, pca_model = plot_pca(beta_values, clinical_data, color_by='disease_status', use_m_values=True)
    # plt.tight_layout()
    
    # plt.figure()
    # ax4, umap_reducer = plot_tsne(beta_values, clinical_data, color_by='age', use_m_values=True)
    # plt.tight_layout()
    
    # plt.figure()
    # ax5 = plot_volcano(results_df_example, effect_size_col='delta_beta', p_value_col='FDR', alpha=0.05, effect_size_threshold=0.1, label_top_n=5)
    # # plt.tight_layout() # Volcano plot handles its own layout adjustment

    # plt.figure()
    # specific_cpg_idx = 10 # Example CpG index
    # specific_cpg_id = results_df_example.index[specific_cpg_idx] # Get CpG ID if available
    # ax6 = plot_methylation_vs_covariate(beta_values, clinical_data, cpg_index=specific_cpg_idx, cpg_id=specific_cpg_id, covariate='age')
    # plt.tight_layout()

    # plt.figure()
    # gene_promoter_indices = list(range(20, 40)) # Example region indices
    # ax7 = plot_region_methylation(gene_promoter_indices, beta_values, clinical_data, group_var='disease_status', region_name='Example Gene Promoter', plot_type='violin')
    # plt.tight_layout()
    
    # plt.show() # Display all created plots

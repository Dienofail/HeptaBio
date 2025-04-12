import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Assuming common_utils.py contains the definitions
from common_utils import CLINICAL_SEX_COL, CLINICAL_SAMPLE_ID_COL, logit_to_beta # Import necessary items

def filter_low_coverage(beta: np.ndarray, depth: np.ndarray, 
                        min_cov: int = 5, min_samples_frac: float = 0.8, 
                        min_mean_sample_depth: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filters CpG sites based on minimum coverage depth across a minimum fraction of samples,
    and optionally filters samples based on minimum mean coverage.

    Args:
        beta (np.ndarray): Beta values matrix (samples x CpGs).
        depth (np.ndarray): Sequencing depth matrix (samples x CpGs).
        min_cov (int): Minimum read depth required for a CpG in a sample. Defaults to 5.
        min_samples_frac (float): Minimum fraction of samples that must meet min_cov for a CpG to be kept. Defaults to 0.8.
        min_mean_sample_depth (float | None): Optional. Minimum mean depth across all CpGs for a sample to be kept. 
                                            If None, no sample filtering based on mean depth is performed. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            - beta_filt: Filtered beta matrix.
            - depth_filt: Filtered depth matrix.
            - sample_mask: Boolean array indicating which samples were kept.
            - cpg_mask: Boolean array indicating which CpGs were kept.
    """
    n_samples, n_cpgs = beta.shape
    min_samples = int(n_samples * min_samples_frac)

    # CpG filtering
    cpg_coverage_mask = np.sum(depth >= min_cov, axis=0) >= min_samples
    beta_filt_cpg = beta[:, cpg_coverage_mask]
    depth_filt_cpg = depth[:, cpg_coverage_mask]
    print(f"CpG filtering: Kept {np.sum(cpg_coverage_mask)} out of {n_cpgs} CpGs.")

    # Sample filtering (optional)
    sample_mask = np.ones(n_samples, dtype=bool)
    if min_mean_sample_depth is not None:
        mean_sample_depth = np.mean(depth_filt_cpg, axis=1) # Use depth matrix after CpG filtering
        sample_mask = mean_sample_depth >= min_mean_sample_depth
        print(f"Sample filtering: Kept {np.sum(sample_mask)} out of {n_samples} samples based on mean depth >= {min_mean_sample_depth}.")
    
    beta_filt = beta_filt_cpg[sample_mask, :]
    depth_filt = depth_filt_cpg[sample_mask, :]

    return beta_filt, depth_filt, sample_mask, cpg_coverage_mask

def filter_low_variance_cpgs(beta: np.ndarray, min_variance: float = 0.001) -> tuple[np.ndarray, np.ndarray]:
    """
    Filters CpG sites with low variance across samples.

    Args:
        beta (np.ndarray): Beta values matrix (samples x CpGs).
        min_variance (float): Minimum variance required for a CpG to be kept. Defaults to 0.001.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - beta_filt: Filtered beta matrix.
            - cpg_mask: Boolean array indicating which CpGs were kept.
    """
    cpg_variances = np.var(beta, axis=0)
    cpg_mask = cpg_variances >= min_variance
    beta_filt = beta[:, cpg_mask]
    print(f"Low variance CpG filtering: Kept {np.sum(cpg_mask)} out of {beta.shape[1]} CpGs (min_variance={min_variance}).")
    return beta_filt, cpg_mask


def calc_sample_stats(beta: np.ndarray, depth: np.ndarray, 
                        sample_ids: list | pd.Index | None = None) -> pd.DataFrame:
    """
    Computes per-sample QC statistics.
    Assumes beta is on the 0-1 scale.

    Args:
        beta (np.ndarray): Beta values matrix (samples x CpGs).
        depth (np.ndarray): Sequencing depth matrix (samples x CpGs).
        sample_ids (list | pd.Index | None): Optional list or index of sample IDs corresponding to rows.
                                            If None, uses range index. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with QC statistics indexed by sample ID.
    """
    n_samples, n_cpgs = beta.shape
    if sample_ids is None:
        sample_ids = pd.RangeIndex(n_samples)
    elif len(sample_ids) != n_samples:
        raise ValueError("Length of sample_ids must match the number of samples in beta/depth matrices.")

    stats_df = pd.DataFrame(index=sample_ids)
    stats_df['total_covered_cpgs'] = np.sum(depth > 0, axis=1)
    stats_df['mean_depth'] = np.mean(depth, axis=1)
    stats_df['median_depth'] = np.median(depth, axis=1)
    stats_df['mean_beta'] = np.mean(beta, axis=1)
    stats_df['median_beta'] = np.median(beta, axis=1)
    stats_df['fraction_beta_gt_0.8'] = np.sum(beta > 0.8, axis=1) / n_cpgs
    stats_df['fraction_beta_lt_0.2'] = np.sum(beta < 0.2, axis=1) / n_cpgs
    
    # Add log total reads (often useful for detecting outliers)
    stats_df['log10_total_reads'] = np.log10(np.sum(depth, axis=1) + 1) # Add 1 to avoid log(0)

    return stats_df


def identify_outlier_samples(sample_stats_df: pd.DataFrame, 
                               metrics: list[str] = ['mean_beta', 'log10_total_reads'], 
                               method: str = "IQR", 
                               iqr_multiplier: float = 1.5, 
                               pca_sd_threshold: float = 6.0, 
                               beta_matrix_for_pca: np.ndarray | None = None) -> list:
    """
    Identifies potential outlier samples based on QC statistics.

    Args:
        sample_stats_df (pd.DataFrame): DataFrame of per-sample QC stats (output of calc_sample_stats).
        metrics (list[str]): List of column names in sample_stats_df to use for outlier detection.
                             Defaults to ['mean_beta', 'log10_total_reads'].
        method (str): Method for outlier detection ("IQR" or "PCA"). Defaults to "IQR".
        iqr_multiplier (float): Multiplier for the IQR range. Used only if method="IQR". Defaults to 1.5.
        pca_sd_threshold (float): SD threshold for PCA-based outlier detection. Used only if method="PCA". Defaults to 6.0.
        beta_matrix_for_pca (np.ndarray | None): The beta value matrix (samples x CpGs) required if method="PCA". 
                                                 Must correspond to the samples in sample_stats_df. Defaults to None.

    Returns:
        list: List of sample IDs identified as outliers.
    """
    outlier_indices = set()

    if method.upper() == "IQR":
        for metric in metrics:
            if metric not in sample_stats_df.columns:
                print(f"Warning: Metric '{metric}' not found in sample_stats_df. Skipping.")
                continue
            
            q1 = sample_stats_df[metric].quantile(0.25)
            q3 = sample_stats_df[metric].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
            
            metric_outliers = sample_stats_df.index[(sample_stats_df[metric] < lower_bound) | (sample_stats_df[metric] > upper_bound)]
            outlier_indices.update(metric_outliers)
            if len(metric_outliers) > 0:
                print(f"Found {len(metric_outliers)} outliers based on {metric} (IQR method).")

    elif method.upper() == "PCA":
        if beta_matrix_for_pca is None:
            raise ValueError("beta_matrix_for_pca must be provided for PCA outlier detection.")
        if beta_matrix_for_pca.shape[0] != len(sample_stats_df):
             raise ValueError("Number of samples in beta_matrix_for_pca must match sample_stats_df.")

        print("Performing PCA for outlier detection...")
        # Consider using M-values for PCA as they often have better distributional properties
        # Standardize data before PCA
        scaler = StandardScaler()
        beta_scaled = scaler.fit_transform(beta_matrix_for_pca)
        
        # Use a reasonable number of components (e.g., 2 or up to 5)
        n_components = min(5, beta_scaled.shape[0], beta_scaled.shape[1])
        pca = PCA(n_components=n_components)
        pca_scores = pca.fit_transform(beta_scaled)

        # Check deviation on each principal component
        for i in range(n_components):
            pc_scores = pca_scores[:, i]
            mean_score = np.mean(pc_scores)
            sd_score = np.std(pc_scores)
            lower_bound = mean_score - pca_sd_threshold * sd_score
            upper_bound = mean_score + pca_sd_threshold * sd_score
            pc_outliers = sample_stats_df.index[(pc_scores < lower_bound) | (pc_scores > upper_bound)]
            if len(pc_outliers) > 0:
                 print(f"Found {len(pc_outliers)} outliers based on PC{i+1} ({pca_sd_threshold} SD threshold).")
            outlier_indices.update(pc_outliers)
            
    else:
        raise ValueError(f"Unknown outlier detection method: {method}. Choose 'IQR' or 'PCA'.")

    print(f"Total unique outliers identified: {len(outlier_indices)}")
    return list(outlier_indices)


def check_sex_consistency(beta: np.ndarray, 
                          clinical_df: pd.DataFrame, 
                          sex_cpg_indices: list[int], 
                          sample_ids: list | pd.Index, 
                          sex_col: str = CLINICAL_SEX_COL, 
                          male_label: str = 'Male', 
                          female_label: str = 'Female', 
                          threshold_diff: float = 0.3) -> list:
    """
    Checks for inconsistencies between reported sex in clinical data and average methylation 
    on sex-specific CpGs (e.g., Y chromosome CpGs).
    Assumes beta is on the 0-1 scale.

    Args:
        beta (np.ndarray): Beta values matrix (samples x CpGs).
        clinical_df (pd.DataFrame): DataFrame with clinical data including a sex column.
                                   Should be indexed by sample ID.
        sex_cpg_indices (list[int]): List of column indices in the beta matrix corresponding to sex-specific CpGs 
                                    (e.g., CpGs on the Y chromosome).
        sample_ids (list | pd.Index): List or index of sample IDs corresponding to the rows in beta matrix.
                                      Must match the index of clinical_df.
        sex_col (str): Column name in clinical_df for sex information. Defaults to common_utils.CLINICAL_SEX_COL.
        male_label (str): Label used for males in the sex column. Defaults to 'Male'.
        female_label (str): Label used for females in the sex column. Defaults to 'Female'.
        threshold_diff (float): Average beta difference threshold to flag potential mismatch. 
                                If mean beta on sex CpGs for a 'Female' is > threshold_diff, or 
                                for a 'Male' is < threshold_diff (assuming Y chr CpGs where M > F), flag it.
                                Adjust based on the specific sex CpGs used. Defaults to 0.3.

    Returns:
        list: List of sample IDs with potential sex mismatches.
    """
    if not pd.api.types.is_list_like(sample_ids):
        raise TypeError("sample_ids must be a list or pandas Index.")
        
    if len(sample_ids) != beta.shape[0]:
        raise ValueError("Length of sample_ids must match number of samples in beta matrix.")

    # Ensure clinical_df is indexed by sample ID for easy lookup
    if not clinical_df.index.equals(pd.Index(sample_ids)):
        if CLINICAL_SAMPLE_ID_COL in clinical_df.columns:
            clinical_df = clinical_df.set_index(CLINICAL_SAMPLE_ID_COL)
        else:
             raise ValueError(f"clinical_df must be indexed by sample ID or contain a '{CLINICAL_SAMPLE_ID_COL}' column.")
         # Reindex to match beta matrix order, handling potential missing samples
        clinical_df = clinical_df.reindex(sample_ids)
            
    if sex_col not in clinical_df.columns:
        raise ValueError(f"Sex column '{sex_col}' not found in clinical_df.")

    if not sex_cpg_indices:
        print("Warning: No sex CpG indices provided. Skipping sex consistency check.")
        return []
    
    # Ensure indices are valid
    max_cpg_index = beta.shape[1] - 1
    if any(idx < 0 or idx > max_cpg_index for idx in sex_cpg_indices):
        raise ValueError(f"sex_cpg_indices contains invalid indices (out of bounds for 0-{max_cpg_index}).")

    # Calculate mean beta across the specified sex CpGs for each sample
    mean_sex_cpg_beta = np.mean(beta[:, sex_cpg_indices], axis=1)

    mismatched_samples = []
    for i, sample_id in enumerate(sample_ids):
        reported_sex = clinical_df.loc[sample_id, sex_col]
        avg_beta = mean_sex_cpg_beta[i]

        if pd.isna(reported_sex):
            print(f"Warning: Missing sex information for sample {sample_id}. Skipping check.")
            continue

        # Assuming Y chromosome CpGs (higher methylation in males)
        # Adjust logic if using X chromosome CpGs with different expected patterns
        if reported_sex == female_label and avg_beta > threshold_diff:
            print(f"Potential mismatch: Sample {sample_id} reported as {female_label} but has high avg beta ({avg_beta:.3f}) on sex CpGs.")
            mismatched_samples.append(sample_id)
        elif reported_sex == male_label and avg_beta < threshold_diff: # Might need adjustment if some Y CpGs have low beta even in males
             print(f"Potential mismatch: Sample {sample_id} reported as {male_label} but has low avg beta ({avg_beta:.3f}) on sex CpGs.")
             mismatched_samples.append(sample_id)
        # Consider adding checks for other labels if present
        elif reported_sex not in [male_label, female_label]:
             print(f"Warning: Unknown sex label '{reported_sex}' for sample {sample_id}.")

    print(f"Sex consistency check identified {len(mismatched_samples)} potential mismatches.")
    return mismatched_samples

# Example Usage (Conceptual - requires actual data loading)
if __name__ == '__main__':
    # --- Assume these are loaded --- 
    # beta_logits = np.random.rand(500, 25000) * 8 - 4 # Example logit data
    # depth = np.random.randint(0, 100, size=(500, 25000))
    # clinical_df = pd.DataFrame({
    #     'sample_id': [f'Sample_{i+1}' for i in range(500)],
    #     CLINICAL_SEX_COL: np.random.choice(['Male', 'Female'], 500),
    #     CLINICAL_DISEASE_COL: np.random.choice(['Case', 'Control'], 500)
    # }).set_index('sample_id')
    # # Assume you have a list of indices for Y chromosome CpGs
    # sex_cpg_indices = list(range(24900, 25000)) # Example indices
    # sample_ids = clinical_df.index
    # --------------------------------
    
    # print("--- Initial Data Shapes ---")
    # print(f"Beta logits shape: {beta_logits.shape}")
    # print(f"Depth shape: {depth.shape}")
    # print(f"Clinical data shape: {clinical_df.shape}")

    # print("\n--- Convert Logits to Beta ---")
    # beta = logit_to_beta(beta_logits)

    # print("\n--- Filter Low Coverage CpGs/Samples ---")
    # beta_filt_cov, depth_filt_cov, sample_mask, cpg_mask_cov = filter_low_coverage(beta, depth, min_cov=5, min_samples_frac=0.8, min_mean_sample_depth=10)
    # print(f"Beta shape after coverage filter: {beta_filt_cov.shape}")
    # filtered_sample_ids = sample_ids[sample_mask]
    # filtered_clinical_df = clinical_df.loc[filtered_sample_ids]
    
    # print("\n--- Filter Low Variance CpGs ---")
    # beta_filt_var, cpg_mask_var = filter_low_variance_cpgs(beta_filt_cov, min_variance=0.005)
    # print(f"Beta shape after variance filter: {beta_filt_var.shape}")
    # # Keep track of the final CpG mask if needed for mapping back
    # final_cpg_mask = cpg_mask_cov.copy()
    # final_cpg_mask[final_cpg_mask] = cpg_mask_var # Combine masks

    # print("\n--- Calculate Sample Stats ---")
    # # Use the beta/depth matrices *after* CpG filtering but *before* sample filtering for QC stats
    # # Or recalculate on the final filtered matrices if preferred
    # beta_for_stats = beta[:, cpg_mask_cov] # Use beta filtered only by CpG coverage
    # depth_for_stats = depth[:, cpg_mask_cov]
    # sample_stats = calc_sample_stats(beta_for_stats, depth_for_stats, sample_ids=sample_ids)
    # print("Sample Stats Head:")
    # print(sample_stats.head())

    # print("\n--- Identify Outlier Samples (IQR) ---")
    # # Use stats calculated *before* sample removal to identify outliers among all original samples
    # outlier_samples_iqr = identify_outlier_samples(sample_stats, metrics=['mean_beta', 'log10_total_reads', 'median_depth'], method="IQR")
    # print(f"IQR Outliers: {outlier_samples_iqr}")

    # print("\n--- Identify Outlier Samples (PCA) ---")
    # # Requires the beta matrix corresponding to sample_stats (before sample removal, after CpG filter)
    # outlier_samples_pca = identify_outlier_samples(sample_stats, beta_matrix_for_pca=beta_for_stats, method="PCA")
    # print(f"PCA Outliers: {outlier_samples_pca}")
    # # Often, samples identified by filtering (e.g., low mean depth) and outliers are combined for removal/investigation

    # print("\n--- Check Sex Consistency ---")
    # # Use beta matrix before sample removal but potentially after CpG filtering
    # # Use clinical data aligned with the full sample set
    # sex_mismatches = check_sex_consistency(beta, clinical_df, sex_cpg_indices, sample_ids=sample_ids)
    # print(f"Sex Mismatches: {sex_mismatches}")

    # print("\n--- Final Data for Analysis (Example) ---")
    # # Decide which samples to remove (e.g., low coverage + outliers + sex mismatches)
    # samples_to_remove = set(sample_ids[~sample_mask]) | set(outlier_samples_iqr) | set(sex_mismatches)
    # final_samples_mask = ~sample_ids.isin(samples_to_remove)
    # final_beta = beta[final_samples_mask][:, final_cpg_mask]
    # final_depth = depth[final_samples_mask][:, final_cpg_mask]
    # final_clinical_df = clinical_df.loc[sample_ids[final_samples_mask]]
    # print(f"Final Beta shape: {final_beta.shape}")
    # print(f"Final Clinical data shape: {final_clinical_df.shape}")
    pass # End of example

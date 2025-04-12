import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import our custom modules
from data_simulator import generate_cfDNA_methylation_data_with_gene_effects
import qc
import plotting
from common_utils import (
    CLINICAL_DISEASE_COL, CLINICAL_STAGE_COL, CLINICAL_AGE_COL,
    CLINICAL_SEX_COL, CLINICAL_ALT_COL, CLINICAL_AST_COL,
    CLINICAL_SAMPLE_ID_COL, beta_to_m, logit_to_beta, m_to_beta
)

# --- Configuration ---
N_SAMPLES = 200
N_CPGS = 10000
N_GENES = 500
RANDOM_SEED = 42

# QC Parameters
MIN_COV = 5
MIN_SAMPLES_FRAC = 0.8
MIN_MEAN_SAMPLE_DEPTH = 10
MIN_VARIANCE_BETA = 0.005 # Beta-value variance threshold
IQR_MULTIPLIER = 1.5
PCA_SD_THRESHOLD = 5.0
SEX_CHECK_THRESHOLD = 0.3 # Adjust based on expectation for Y-chr CpGs

# Output directories
PLOTS_DIR = "plots"
TABLES_DIR = "tables"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# --- 1. Generate Simulated Data ---
print("--- Generating Simulated Data ---")
beta_raw, depth_raw, clinical_df_raw, genes_df, cpg_ids_raw, sex_cpg_idx_set = \
    generate_cfDNA_methylation_data_with_gene_effects(
        n_samples=N_SAMPLES,
        n_cpgs=N_CPGS,
        n_genes=N_GENES,
        random_seed=RANDOM_SEED
    )

# Assign sample IDs for easier tracking
sample_ids_raw = [f'Sample_{i+1}' for i in range(N_SAMPLES)]
clinical_df_raw[CLINICAL_SAMPLE_ID_COL] = sample_ids_raw
clinical_df_raw = clinical_df_raw.set_index(CLINICAL_SAMPLE_ID_COL)
# Convert set of indices to list for check_sex_consistency
sex_cpg_indices_raw = list(sex_cpg_idx_set)

print(f"Raw Beta shape: {beta_raw.shape}")
print(f"Raw Depth shape: {depth_raw.shape}")
print(f"Raw Clinical shape: {clinical_df_raw.shape}")
print(f"Raw CpG IDs count: {len(cpg_ids_raw)}")
print(f"Genes DF shape: {genes_df.shape}")
print(f"Number of designated sex CpGs: {len(sex_cpg_indices_raw)}")

# --- 2. Initial Data Visualization ---
print("\n--- Plotting Initial Data Distributions ---")

# Beta distribution
fig_beta_raw, ax_beta_raw = plt.subplots(figsize=(10, 6))
plotting.plot_beta_distribution(
    beta_raw,
    group_labels=clinical_df_raw[CLINICAL_DISEASE_COL],
    group_name=CLINICAL_DISEASE_COL,
    ax=ax_beta_raw
)
ax_beta_raw.set_title("Initial Beta Value Distribution")
fig_beta_raw.savefig(os.path.join(PLOTS_DIR, "01_beta_distribution_raw.png"), bbox_inches='tight')
plt.close(fig_beta_raw)

# Coverage distribution
fig_cov_raw, ax_cov_raw = plt.subplots(figsize=(8, 6))
plotting.plot_coverage_distribution(
    depth_raw,
    group_labels=clinical_df_raw[CLINICAL_DISEASE_COL],
    group_name=CLINICAL_DISEASE_COL,
    log_scale=True,
    ax=ax_cov_raw
)
ax_cov_raw.set_title("Initial Mean Sample Depth Distribution")
fig_cov_raw.savefig(os.path.join(PLOTS_DIR, "02_coverage_distribution_raw.png"), bbox_inches='tight')
plt.close(fig_cov_raw)

# --- 3. QC Step 1: Filter Low Coverage CpGs and Samples ---
print(f"\n--- QC Step 1: Filtering Low Coverage (min_cov={MIN_COV}, min_samples_frac={MIN_SAMPLES_FRAC}, min_mean_depth={MIN_MEAN_SAMPLE_DEPTH}) ---")
beta_filt_cov, depth_filt_cov, sample_mask_cov, cpg_mask_cov = qc.filter_low_coverage(
    beta_raw,
    depth_raw,
    min_cov=MIN_COV,
    min_samples_frac=MIN_SAMPLES_FRAC,
    min_mean_sample_depth=MIN_MEAN_SAMPLE_DEPTH
)

# Update data based on filtering
beta_step1 = beta_filt_cov
depth_step1 = depth_filt_cov
sample_ids_step1 = clinical_df_raw.index[sample_mask_cov]
clinical_df_step1 = clinical_df_raw.loc[sample_ids_step1]
cpg_ids_step1 = [cpg_id for i, cpg_id in enumerate(cpg_ids_raw) if cpg_mask_cov[i]]
# Adjust sex CpG indices based on CpG filtering
sex_cpg_mapping_step1 = {old_idx: new_idx for new_idx, old_idx in enumerate(np.where(cpg_mask_cov)[0])}
sex_cpg_indices_step1 = [sex_cpg_mapping_step1[idx] for idx in sex_cpg_indices_raw if idx in sex_cpg_mapping_step1]

print(f"Shape after coverage filter: Beta={beta_step1.shape}, Depth={depth_step1.shape}, Clinical={clinical_df_step1.shape}")
print(f"Number of sex CpGs remaining after coverage filter: {len(sex_cpg_indices_step1)}")

samples_removed_cov = list(clinical_df_raw.index[~sample_mask_cov])
cpgs_removed_cov = [cpg_id for i, cpg_id in enumerate(cpg_ids_raw) if not cpg_mask_cov[i]]
print(f"Samples removed by coverage filter: {len(samples_removed_cov)}")
print(f"CpGs removed by coverage filter: {len(cpgs_removed_cov)}")

# Save removed lists
pd.Series(samples_removed_cov, name="SampleID").to_csv(os.path.join(TABLES_DIR, "samples_removed_low_coverage.csv"), index=False)
pd.Series(cpgs_removed_cov, name="CpGID").to_csv(os.path.join(TABLES_DIR, "cpgs_removed_low_coverage.csv"), index=False)


# --- 4. QC Step 2: Filter Low Variance CpGs ---
print(f"\n--- QC Step 2: Filtering Low Variance CpGs (min_variance={MIN_VARIANCE_BETA}) ---")
# Apply to data after coverage filtering
beta_filt_var, cpg_mask_var = qc.filter_low_variance_cpgs(
    beta_step1,
    min_variance=MIN_VARIANCE_BETA
)

# Update data based on variance filtering
beta_step2 = beta_filt_var
depth_step2 = depth_step1[:, cpg_mask_var] # Apply mask to depth as well
clinical_df_step2 = clinical_df_step1 # Samples unchanged in this step
sample_ids_step2 = sample_ids_step1
cpg_ids_step2 = [cpg_id for i, cpg_id in enumerate(cpg_ids_step1) if cpg_mask_var[i]]
# Adjust sex CpG indices based on variance filtering
sex_cpg_mapping_step2 = {old_idx: new_idx for new_idx, old_idx in enumerate(np.where(cpg_mask_var)[0])}
sex_cpg_indices_step2 = [sex_cpg_mapping_step2[idx] for idx in sex_cpg_indices_step1 if idx in sex_cpg_mapping_step2]

print(f"Shape after variance filter: Beta={beta_step2.shape}, Depth={depth_step2.shape}, Clinical={clinical_df_step2.shape}")
print(f"Number of sex CpGs remaining after variance filter: {len(sex_cpg_indices_step2)}")

cpgs_removed_var = [cpg_id for i, cpg_id in enumerate(cpg_ids_step1) if not cpg_mask_var[i]]
print(f"CpGs removed by variance filter: {len(cpgs_removed_var)}")
pd.Series(cpgs_removed_var, name="CpGID").to_csv(os.path.join(TABLES_DIR, "cpgs_removed_low_variance.csv"), index=False)

# Plot beta distribution after filtering
fig_beta_filt, ax_beta_filt = plt.subplots(figsize=(10, 6))
plotting.plot_beta_distribution(
    beta_step2,
    group_labels=clinical_df_step2[CLINICAL_DISEASE_COL],
    group_name=CLINICAL_DISEASE_COL,
    ax=ax_beta_filt
)
ax_beta_filt.set_title("Beta Value Distribution After CpG Filtering")
fig_beta_filt.savefig(os.path.join(PLOTS_DIR, "03_beta_distribution_filtered_cpgs.png"), bbox_inches='tight')
plt.close(fig_beta_filt)

# --- 5. QC Step 3: Calculate Sample Statistics ---
print("\n--- QC Step 3: Calculating Sample Statistics ---")
# Calculate stats on data *after* CpG filters but *before* outlier sample removal
# Use the original full set of samples that passed coverage filter for comparison
beta_for_stats = beta_raw[sample_mask_cov][:, cpg_mask_cov] # Filter beta_raw by both masks
depth_for_stats = depth_raw[sample_mask_cov][:, cpg_mask_cov] # Filter depth_raw by both masks
sample_stats_df = qc.calc_sample_stats(beta_for_stats, depth_for_stats, sample_ids=sample_ids_step1) # Use sample IDs from step 1

print("Sample Statistics Head:")
print(sample_stats_df.head())
sample_stats_df.to_csv(os.path.join(TABLES_DIR, "sample_qc_statistics.csv"))

# Plot distributions of key QC metrics
qc_metrics_to_plot = ['mean_depth', 'median_depth', 'mean_beta', 'median_beta', 'log10_total_reads', 'fraction_beta_gt_0.8', 'fraction_beta_lt_0.2']
n_metrics = len(qc_metrics_to_plot)
fig_qc_stats, axes_qc_stats = plt.subplots(nrows=(n_metrics + 1) // 2, ncols=2, figsize=(12, 4 * ((n_metrics + 1) // 2)))
axes_flat = axes_qc_stats.flatten()
for i, metric in enumerate(qc_metrics_to_plot):
    sns.histplot(sample_stats_df[metric], kde=True, ax=axes_flat[i])
    axes_flat[i].set_title(f"Distribution of {metric}")
    axes_flat[i].set_xlabel(metric)
    axes_flat[i].set_ylabel("Frequency")
# Hide any unused subplots
for j in range(i + 1, len(axes_flat)):
    fig_qc_stats.delaxes(axes_flat[j])
fig_qc_stats.suptitle("Distributions of Sample QC Metrics", fontsize=16, y=1.02)
fig_qc_stats.tight_layout()
fig_qc_stats.savefig(os.path.join(PLOTS_DIR, "04_sample_qc_metric_distributions.png"), bbox_inches='tight')
plt.close(fig_qc_stats)


# --- 6. QC Step 4: Identify Outlier Samples ---
print("\n--- QC Step 4: Identifying Outlier Samples ---")
print(f"Using IQR method (multiplier={IQR_MULTIPLIER})...")
outlier_samples_iqr = qc.identify_outlier_samples(
    sample_stats_df,
    metrics=['mean_beta', 'log10_total_reads', 'median_depth'], # Use metrics from calculated stats
    method="IQR",
    iqr_multiplier=IQR_MULTIPLIER
)
print(f"IQR Outliers identified: {len(outlier_samples_iqr)} -> {outlier_samples_iqr}")

print(f"\nUsing PCA method (SD threshold={PCA_SD_THRESHOLD})...")
# Use the beta matrix corresponding to sample_stats_df (beta_for_stats)
try:
    outlier_samples_pca = qc.identify_outlier_samples(
        sample_stats_df,
        beta_matrix_for_pca=beta_for_stats, # Use data after CpG filters, before sample removal
        method="PCA",
        pca_sd_threshold=PCA_SD_THRESHOLD
    )
    print(f"PCA Outliers identified: {len(outlier_samples_pca)} -> {outlier_samples_pca}")
except ValueError as e:
    print(f"PCA outlier detection failed: {e}")
    outlier_samples_pca = []


# Plot PCA colored by different factors, annotating outliers
all_potential_outliers = list(set(outlier_samples_iqr) | set(outlier_samples_pca))
print(f"\nTotal unique potential outliers (IQR or PCA): {len(all_potential_outliers)}")

# Use M-values for PCA plotting as recommended
m_values_step2 = beta_to_m(beta_step2) # Use fully filtered beta for plotting final state

fig_pca, axes_pca = plt.subplots(1, 3, figsize=(24, 7))
# PCA colored by Disease Status
plotting.plot_pca(
    beta_step2, # Use final filtered beta
    clinical_df_step2,
    color_by=CLINICAL_DISEASE_COL,
    use_m_values=True, # Convert to M-values internally
    scale_data=True,   # Scale M-values
    annotate_samples=all_potential_outliers,
    ax=axes_pca[0]
)
axes_pca[0].set_title(f"PCA after CpG filters (Color: {CLINICAL_DISEASE_COL})")

# PCA colored by Age
plotting.plot_pca(
    beta_step2, clinical_df_step2, color_by=CLINICAL_AGE_COL,
    use_m_values=True, scale_data=True, annotate_samples=all_potential_outliers, ax=axes_pca[1]
)
axes_pca[1].set_title(f"PCA after CpG filters (Color: {CLINICAL_AGE_COL})")

# PCA colored by Sex
plotting.plot_pca(
    beta_step2, clinical_df_step2, color_by=CLINICAL_SEX_COL,
    use_m_values=True, scale_data=True, annotate_samples=all_potential_outliers, ax=axes_pca[2]
)
axes_pca[2].set_title(f"PCA after CpG filters (Color: {CLINICAL_SEX_COL})")

fig_pca.suptitle("PCA Plots After CpG Filtering (Outliers Annotated)", fontsize=16, y=1.03)
fig_pca.tight_layout()
fig_pca.savefig(os.path.join(PLOTS_DIR, "05_pca_plots_filtered_annotated.png"), bbox_inches='tight')
plt.close(fig_pca)

# Plot UMAP as well
fig_tsne, axes_tsne = plt.subplots(1, 3, figsize=(24, 7))
# t-SNE colored by Disease Status
plotting.plot_tsne(
    beta_step2, clinical_df_step2, color_by=CLINICAL_DISEASE_COL,
    use_m_values=True, annotate_samples=all_potential_outliers, ax=axes_tsne[0]
)
axes_tsne[0].set_title(f"t-SNE after CpG filters (Color: {CLINICAL_DISEASE_COL})")

# t-SNE colored by Age
plotting.plot_tsne(
    beta_step2, clinical_df_step2, color_by=CLINICAL_AGE_COL,
    use_m_values=True, annotate_samples=all_potential_outliers, ax=axes_tsne[1]
)
axes_tsne[1].set_title(f"t-SNE after CpG filters (Color: {CLINICAL_AGE_COL})")

# t-SNE colored by Sex
plotting.plot_tsne(
    beta_step2, clinical_df_step2, color_by=CLINICAL_SEX_COL,
    use_m_values=True, annotate_samples=all_potential_outliers, ax=axes_tsne[2]
)
axes_tsne[2].set_title(f"t-SNE after CpG filters (Color: {CLINICAL_SEX_COL})")

fig_tsne.suptitle("t-SNE Plots After CpG Filtering (Outliers Annotated)", fontsize=16, y=1.03)
fig_tsne.tight_layout()
fig_tsne.savefig(os.path.join(PLOTS_DIR, "06_tsne_plots_filtered_annotated.png"), bbox_inches='tight')
plt.close(fig_tsne)


# --- 7. QC Step 5: Check Sex Consistency ---
print(f"\n--- QC Step 5: Checking Sex Consistency (threshold={SEX_CHECK_THRESHOLD}) ---")
# Use the data after CpG filtering but before removing outlier samples
# The clinical_df used should align with the beta matrix used (clinical_df_step2, beta_step2)
if len(sex_cpg_indices_step2) > 0:
    sex_mismatches = qc.check_sex_consistency(
        beta=beta_step2,
        clinical_df=clinical_df_step2, # Has correct samples and index
        sex_cpg_indices=sex_cpg_indices_step2, # Use indices valid for beta_step2
        sample_ids=sample_ids_step2, # Pass the corresponding sample IDs
        sex_col=CLINICAL_SEX_COL,
        male_label='M', # Match labels used in simulator
        female_label='F',
        threshold_diff=SEX_CHECK_THRESHOLD
    )
    print(f"Potential Sex Mismatches identified: {len(sex_mismatches)} -> {sex_mismatches}")
    pd.Series(sex_mismatches, name="SampleID").to_csv(os.path.join(TABLES_DIR, "samples_sex_mismatch.csv"), index=False)

else:
    print("Skipping sex consistency check as no sex-specific CpGs remained after filtering.")
    sex_mismatches = []


# --- 8. Final Filtering and Summary ---
print("\n--- Final Data Summary ---")
# Combine all samples flagged for removal
samples_to_remove = set(samples_removed_cov) | set(all_potential_outliers) | set(sex_mismatches)
print(f"Total unique samples flagged for removal (low cov, outlier, or sex mismatch): {len(samples_to_remove)}")
pd.Series(list(samples_to_remove), name="SampleID").to_csv(os.path.join(TABLES_DIR, "samples_removed_all_qc.csv"), index=False)


# Define final data based on samples *not* flagged for removal
final_sample_mask = ~sample_ids_step2.isin(samples_to_remove) # Mask based on samples remaining after step 1/2
beta_final = beta_step2[final_sample_mask, :]
depth_final = depth_step2[final_sample_mask, :]
clinical_df_final = clinical_df_step2.loc[final_sample_mask]
cpg_ids_final = cpg_ids_step2 # CpGs are already filtered
sample_ids_final = clinical_df_final.index

print(f"\nFinal data shapes after all QC:")
print(f"  Beta: {beta_final.shape}")
print(f"  Depth: {depth_final.shape}")
print(f"  Clinical: {clinical_df_final.shape}")
print(f"  CpG IDs: {len(cpg_ids_final)}")

# Optionally save final filtered data (can be large)
# np.savez_compressed(os.path.join(TABLES_DIR, "final_qc_data.npz"),
#                     beta=beta_final,
#                     depth=depth_final,
#                     cpg_ids=cpg_ids_final)
# clinical_df_final.to_csv(os.path.join(TABLES_DIR, "final_qc_clinical_data.csv"))

print(f"\nQC Analysis Complete. Plots saved to '{PLOTS_DIR}/', tables saved to '{TABLES_DIR}/'.")

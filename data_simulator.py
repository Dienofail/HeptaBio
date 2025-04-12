import numpy as np
import pandas as pd
from common_utils import (
    CLINICAL_DISEASE_COL, CLINICAL_STAGE_COL, CLINICAL_AGE_COL, 
    CLINICAL_SEX_COL, CLINICAL_ALT_COL, CLINICAL_AST_COL,
    GENES_CPG_COL, GENES_GENE_COL,
    beta_to_m, logit_to_beta
)

def generate_cfDNA_methylation_data_with_gene_effects(
    n_samples=500,
    n_cpgs=25000,
    n_genes=1000,
    random_seed=None
):
    """
    Generates a synthetic cfDNA methylation dataset with both per-CpG and gene-level associations
    (for disease, fibrosis stage, age, sex) and gene-level variation in sequencing depth.
    
    Returns:
        beta_value (np.ndarray): shape (n_samples, n_cpgs) of β-values in [0,1].
        depth (np.ndarray): shape (n_samples, n_cpgs) of integer read coverage.
        clinical_df (pd.DataFrame): n_samples rows, containing metadata (disease, stage, age, etc.).
        genes_df (pd.DataFrame): n_cpgs rows, mapping each CpG site to one gene.
        cpg_ids (list[str]): length n_cpgs, the CpG identifiers matching columns of beta_value/depth.
        sex_cpg_idx (set): indices of CpGs associated with sex.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # ======================
    # 1. Generate Clinical Data
    # ======================
    n_cases = n_samples // 2
    n_controls = n_samples - n_cases
    
    # Disease status
    statuses = np.array(["Control"] * n_controls + ["Case"] * n_cases)
    
    # Fibrosis stage: 0 for controls, random 1–4 for cases
    stages = np.zeros(n_samples, dtype=int)
    stages[n_controls:] = np.random.choice([1, 2, 3, 4], size=n_cases, p=[0.25, 0.25, 0.25, 0.25])
    
    # Ages: older for cases
    control_ages = np.clip(np.random.normal(loc=50, scale=10, size=n_controls), 30, 80)
    case_ages    = np.clip(np.random.normal(loc=60, scale=10, size=n_cases), 30, 85)
    ages = np.concatenate([control_ages, case_ages])
    
    # Sex: 50% M in controls, 60% M in cases
    control_sex = np.random.choice(["M", "F"], size=n_controls, p=[0.5, 0.5])
    case_sex    = np.random.choice(["M", "F"], size=n_cases,    p=[0.6, 0.4])
    sex = np.concatenate([control_sex, case_sex])
    
    # ALT, AST: higher in cases
    control_alt = np.clip(np.random.normal(loc=30, scale=10, size=n_controls), 5, 200)
    case_alt    = np.clip(np.random.normal(loc=80, scale=30, size=n_cases), 5, 500)
    alt_values  = np.concatenate([control_alt, case_alt])
    
    control_ast = np.clip(np.random.normal(loc=32, scale=10, size=n_controls), 5, 200)
    case_ast    = np.clip(np.random.normal(loc=75, scale=25, size=n_cases), 5, 500)
    ast_values  = np.concatenate([control_ast, case_ast])
    
    # Build the clinical DataFrame
    clinical_df = pd.DataFrame({
        CLINICAL_DISEASE_COL: statuses,
        CLINICAL_STAGE_COL: stages,
        CLINICAL_AGE_COL: ages.astype(int),
        CLINICAL_SEX_COL: sex,
        CLINICAL_ALT_COL: alt_values,
        CLINICAL_AST_COL: ast_values
    })
    
    # ======================
    # 2. Assign Genes and CpGs
    # ======================
    # We'll create gene symbols, then assign ~10–30 CpGs per gene = n_cpgs total.
    import string
    
    # Some known liver-related genes as a seed
    known_liver_genes = [
        "PPARA", "TGFBI", "TGFB1", "ALB", "APOE", "APOA1", "HNF4A", "COL1A1", 
        "COL3A1", "ACTA2", "IL6", "TNF", "EGFR", "PDGFRB", "MMP2", "MMP9",
        "FOXO1", "PPARG", "MYC", "TERT"
    ]
    gene_symbols = []
    
    # Add known genes first (up to n_genes)
    for g in known_liver_genes:
        if len(gene_symbols) < n_genes:
            gene_symbols.append(g)
    
    # Fill in random gene names if needed
    while len(gene_symbols) < n_genes:
        name_length = np.random.randint(3, 6)
        letters = ''.join(np.random.choice(list(string.ascii_uppercase), size=name_length))
        if np.random.rand() < 0.5:
            letters += str(np.random.randint(1, 10))
        if letters not in gene_symbols:
            gene_symbols.append(letters)
    
    np.random.shuffle(gene_symbols)
    
    # We'll assign 10 CpGs per gene initially, then distribute leftover
    cpgs_per_gene = [10] * n_genes
    assigned = 10 * n_genes
    leftover = n_cpgs - assigned
    
    # Distribute leftover among genes, up to 30 per gene
    g_idx = 0
    while leftover > 0:
        if cpgs_per_gene[g_idx] < 30:
            cpgs_per_gene[g_idx] += 1
            leftover -= 1
        g_idx = (g_idx + 1) % n_genes
    
    # Create CpG IDs
    cpg_ids = [f"CpG{idx+1:05d}" for idx in range(n_cpgs)]
    
    # Build a gene -> list of CpG indices
    gene_map = []
    cpg_counter = 0
    for gene_i, gene_name in enumerate(gene_symbols):
        n_for_gene = cpgs_per_gene[gene_i]
        for _ in range(n_for_gene):
            gene_map.append((cpg_counter, gene_i))  # (CpG index, gene index)
            cpg_counter += 1
    
    # Sort by CpG index just for consistency
    gene_map = sorted(gene_map, key=lambda x: x[0])
    
    # Build genes_df with (CpG_ID, GeneSymbol)
    rows = []
    for (cpgi, genei) in gene_map:
        cpg_id = cpg_ids[cpgi]
        g_name = gene_symbols[genei]
        rows.append({GENES_CPG_COL: cpg_id, GENES_GENE_COL: g_name})
    genes_df = pd.DataFrame(rows)
    
    # ======================
    # 3. Define Gene-Level Associations
    # ======================
    # Suppose we designate ~5% of genes to have a disease-level effect,
    # ~3% to have a stage-level effect, ~3% to have age association, ~2% to have sex association
    # (You can adjust these proportions freely.)
    n_disease_genes = int(0.05 * n_genes)  # 5%
    n_stage_genes   = int(0.03 * n_genes)
    n_age_genes     = int(0.03 * n_genes)
    n_sex_genes     = int(0.02 * n_genes)
    
    all_gene_indices = np.arange(n_genes)
    np.random.shuffle(all_gene_indices)
    
    disease_genes = set(all_gene_indices[:n_disease_genes])
    stage_genes   = set(all_gene_indices[n_disease_genes : n_disease_genes + n_stage_genes])
    age_genes     = set(all_gene_indices[n_disease_genes + n_stage_genes : n_disease_genes + n_stage_genes + n_age_genes])
    sex_genes     = set(all_gene_indices[n_disease_genes + n_stage_genes + n_age_genes :
                                        n_disease_genes + n_stage_genes + n_age_genes + n_sex_genes])
    
    # Assign gene-level coefficients in logit space
    gene_intercept = np.zeros(n_genes)  # baseline intercept if we want gene-level baseline, but let's keep it 0 for simplicity
    gene_coef_disease = np.zeros(n_genes)
    gene_coef_stage   = np.zeros(n_genes)
    gene_coef_age     = np.zeros(n_genes)
    gene_coef_sex     = np.zeros(n_genes)
    
    # Disease gene effects
    for g in disease_genes:
        # random sign, random magnitude
        sign = 1 if np.random.rand() < 0.5 else -1
        mag = np.random.exponential(scale=1.0)
        if mag > 4:
            mag = 4
        gene_coef_disease[g] = sign * mag
    
    # Stage gene effects
    for g in stage_genes:
        sign = 1 if np.random.rand() < 0.5 else -1
        mag = np.random.exponential(scale=0.5)
        if mag > 3:
            mag = 3
        gene_coef_stage[g] = sign * mag
    
    # Age gene effects
    for g in age_genes:
        sign = 1 if np.random.rand() < 0.5 else -1
        mag = np.random.uniform(0.01, 0.05)
        gene_coef_age[g] = sign * mag
    
    # Sex gene effects
    for g in sex_genes:
        sign = 1 if np.random.rand() < 0.5 else -1
        mag = np.random.uniform(0.1, 1.0)
        gene_coef_sex[g] = sign * mag
    
    # ======================
    # 4. Per-CpG Effects (as before)
    # ======================
    # We'll keep the original approach to per-CpG disease/stage/age/sex associations
    # but you may reduce or alter them to avoid overshadowing gene-level signals.
    
    # Choose subsets of CpGs
    n_disease_cpg = int(0.03 * n_cpgs)
    n_stage_cpg   = int(0.02 * n_cpgs)
    n_age_cpg     = int(0.02 * n_cpgs)
    n_sex_cpg     = int(0.012 * n_cpgs)
    
    all_cpg_indices = np.arange(n_cpgs)
    np.random.shuffle(all_cpg_indices)
    disease_cpg_idx = set(all_cpg_indices[:n_disease_cpg])
    stage_cpg_idx   = set(all_cpg_indices[n_disease_cpg : n_disease_cpg + n_stage_cpg])
    
    # For simplicity, remove overlap disease vs stage
    overlap_ds = disease_cpg_idx.intersection(stage_cpg_idx)
    if overlap_ds:
        stage_cpg_idx = stage_cpg_idx - overlap_ds
    
    age_cpg_idx = set(all_cpg_indices[n_disease_cpg + n_stage_cpg :
                                     n_disease_cpg + n_stage_cpg + n_age_cpg])
    sex_cpg_idx = set(all_cpg_indices[n_disease_cpg + n_stage_cpg + n_age_cpg :
                                     n_disease_cpg + n_stage_cpg + n_age_cpg + n_sex_cpg])
    
    # Site-level coefficient arrays
    intercept_cpg   = np.zeros(n_cpgs)
    coef_disease_cp = np.zeros(n_cpgs)
    coef_stage_cp   = np.zeros(n_cpgs)
    coef_age_cp     = np.zeros(n_cpgs)
    coef_sex_cp     = np.zeros(n_cpgs)
    
    # Baseline intercept from Beta(0.5, 0.5)
    baseline_beta = np.random.beta(a=0.5, b=0.5, size=n_cpgs)
    baseline_beta = np.clip(baseline_beta, 1e-6, 1 - 1e-6)
    intercept_cpg = np.log(baseline_beta / (1 - baseline_beta))  # natural logit
    
    # Disease CP
    for cpg in disease_cpg_idx:
        base_beta_val = baseline_beta[cpg]
        if base_beta_val > 0.8:
            sign = -1
        elif base_beta_val < 0.2:
            sign = 1
        else:
            sign = 1 if np.random.rand() < 0.5 else -1
        mag = np.random.exponential(scale=1.0)
        if mag > 5:
            mag = 5
        coef_disease_cp[cpg] = sign * mag
    
    # Stage CP
    for cpg in stage_cpg_idx:
        base_beta_val = baseline_beta[cpg]
        if base_beta_val > 0.8:
            sign = -1
        elif base_beta_val < 0.2:
            sign = 1
        else:
            sign = 1 if np.random.rand() < 0.5 else -1
        mag = np.random.exponential(scale=0.5)
        if mag > 3:
            mag = 3
        coef_stage_cp[cpg] = sign * mag
    
    # Age CP
    for cpg in age_cpg_idx:
        sign = 1 if np.random.rand() < 0.5 else -1
        mag = np.random.uniform(0.005, 0.05)
        coef_age_cp[cpg] = sign * mag
    
    # Sex CP
    for cpg in sex_cpg_idx:
        sign = 1 if np.random.rand() < 0.5 else -1
        mag = np.random.uniform(0.1, 1.0)
        coef_sex_cp[cpg] = sign * mag
    
    # ======================
    # 5. Construct M-values for Each Sample, Summing Gene + CpG Effects
    # ======================
    sex_binary = np.array([1 if s == "M" else 0 for s in sex])
    disease_binary = np.array([1 if st == "Case" else 0 for st in statuses])
    age_mean = ages.mean()
    age_centered = ages - age_mean
    
    # We'll build M_values by combining:
    #   M = intercept_cpg + (site-level terms) + (gene-level terms).
    # For the gene-level terms, we need to know which gene each CpG belongs to.
    
    M_values = np.zeros((n_samples, n_cpgs), dtype=float)
    
    # For each CpG, find its gene, gather gene-level coefficients
    # Then add everything for each sample
    # Vectorizing by site might be simpler (one pass over cpgs).
    for (cpgi, genei) in gene_map:
        # site-level intercept
        site_int = intercept_cpg[cpgi]
        # site-level disease, stage, age, sex coefs
        site_disease_coef = coef_disease_cp[cpgi]
        site_stage_coef   = coef_stage_cp[cpgi]
        site_age_coef     = coef_age_cp[cpgi]
        site_sex_coef     = coef_sex_cp[cpgi]
        
        # gene-level intercept
        g_int         = gene_intercept[genei]
        g_disease_coef= gene_coef_disease[genei]
        g_stage_coef  = gene_coef_stage[genei]
        g_age_coef    = gene_coef_age[genei]
        g_sex_coef    = gene_coef_sex[genei]
        
        # Combine for each sample
        # M_ij = (site_int + g_int)
        #       + (site_disease_coef + g_disease_coef)*Disease
        #       + (site_stage_coef + g_stage_coef)*Stage
        #       + (site_age_coef + g_age_coef)*(Age - mean)
        #       + (site_sex_coef + g_sex_coef)*SexBinary
        M_site = (
            site_int + g_int
            + (site_disease_coef + g_disease_coef) * disease_binary
            + (site_stage_coef + g_stage_coef) * stages
            + (site_age_coef + g_age_coef) * age_centered
            + (site_sex_coef + g_sex_coef) * sex_binary
        )
        # shape (n_samples,)
        M_values[:, cpgi] = M_site
    
    # Random residual variation
    # We'll again scale by baseline variability
    base_variability = baseline_beta * (1 - baseline_beta)
    residual_sd = 0.05 + 0.45 * (4.0 * base_variability)
    # shape (n_cpgs,)
    noise_matrix = np.random.normal(0.0, 1.0, size=(n_samples, n_cpgs))
    noise_matrix *= residual_sd  # broadcast across columns
    M_values += noise_matrix
    
    # Convert to beta
    beta_matrix = 1.0 / (1.0 + np.exp(-M_values))
    beta_matrix = np.clip(beta_matrix, 0.0, 1.0)
    
    # ======================
    # 6. Simulate Read Depth (with Gene-Level Variation)
    # ======================
    # We combine a gene coverage factor and a site coverage factor
    # so coverage = mean_depth * gene_factor[genei] * site_factor[cpgi].
    
    mean_depth = 30.0
    size_param = 20.0  # NB dispersion
    
    # 6a. Draw gene-level coverage factors
    gene_cov_factor = np.random.lognormal(mean=0.0, sigma=0.6, size=n_genes)  # each gene a factor ~ logNormal(1, 0.6)
    
    # 6b. Draw site-level coverage factors (some variation within gene)
    site_cov_factor = np.random.lognormal(mean=0.0, sigma=0.4, size=n_cpgs)
    
    # We'll compute a final coverage mean for each site = mean_depth * gene_cov_factor[g] * site_cov_factor[cpg]
    # Then sample from Negative Binomial for each sample.
    depth_matrix = np.zeros((n_samples, n_cpgs), dtype=int)
    
    for (cpgi, genei) in gene_map:
        # final mean coverage
        site_mean = mean_depth * gene_cov_factor[genei] * site_cov_factor[cpgi]
        # NB p = size / (size + mean)
        p_site = size_param / (size_param + site_mean)
        
        # draw coverage for n_samples
        coverage_draws = np.random.negative_binomial(
            n=int(size_param),
            p=p_site,
            size=n_samples
        )
        coverage_draws[coverage_draws < 1] = 1
        depth_matrix[:, cpgi] = coverage_draws
    
    # ======================
    # 7. Final Outputs
    # ======================
    beta_value = beta_matrix.astype(np.float32)
    depth      = depth_matrix.astype(np.int32)
    
    return beta_value, depth, clinical_df, genes_df, cpg_ids, sex_cpg_idx


# Quick test if run as main
if __name__ == "__main__":
    beta_val, depth_val, cdf, gdf, cids, sex_idx = generate_cfDNA_methylation_data_with_gene_effects(
        n_samples=500,
        n_cpgs=25000,
        n_genes=1000,
        random_seed=42
    )
    print("beta_val shape:", beta_val.shape)
    print("depth_val shape:", depth_val.shape)
    print("clinical_df shape:", cdf.shape)
    print("genes_df shape:", gdf.shape)
    print("Example gene assignment:\n", gdf.head(10))
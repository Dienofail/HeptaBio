import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from patsy import dmatrices
import warnings
from joblib import Parallel, delayed
import re
import intervaltree

# Remove rpy2 imports and set HAVE_RPY2 to False
HAVE_RPY2 = False
if not HAVE_RPY2:
    warnings.warn("Running without rpy2. Using Python-only implementations.")

from common_utils import (
    beta_to_m, m_to_beta, logit_to_beta, beta_to_logit,
    CLINICAL_DISEASE_COL, CLINICAL_AGE_COL, CLINICAL_SEX_COL, GENES_GENE_COL, GENES_CPG_COL
)

def design_matrix(clinical_df, formula):
    """
    Constructs a design matrix from clinical data using a formula.
    
    Args:
        clinical_df (pd.DataFrame): DataFrame with clinical data.
        formula (str): Formula string (e.g., "~ Disease + Age + Sex").
            The formula should start with '~' and use column names from clinical_df.
    
    Returns:
        pd.DataFrame: Design matrix with appropriate dummy coding and intercept.
    """
    # Ensure formula starts with ~
    if not formula.startswith('~'):
        formula = '~ ' + formula
    
    # Use patsy to create the design matrix
    from patsy import dmatrix
    design = dmatrix(formula, data=clinical_df, return_type='dataframe')
    
    return design

def _fit_ols_single(y, X):
    """Helper function to fit OLS for a single CpG site."""
    model = sm.OLS(y, X)
    try:
        results = model.fit()
        return results.params, results.pvalues
    except:
        return np.full(X.shape[1], np.nan), np.full(X.shape[1], np.nan)

def test_differential_methylation_python(beta_logits, clinical_df, formula, variable_of_interest=None, 
                                         method="ols", convert_to_mvalues=True, n_jobs=-1):
    """
    Python implementation of differential methylation testing.
    
    Args:
        beta_logits (np.ndarray): Matrix of beta values in logit space (samples x CpGs).
        clinical_df (pd.DataFrame): Clinical data with covariates.
        formula (str): Model formula (e.g., "~ Disease + Age + Sex").
        variable_of_interest (str, optional): Variable to extract effect size and p-value for.
            If None, uses first non-intercept term. Defaults to None.
        method (str, optional): Statistical method, one of "ols" or "logistic". Defaults to "ols".
        convert_to_mvalues (bool, optional): Whether to convert beta to M-values. Defaults to True.
        n_jobs (int, optional): Number of parallel jobs. -1 means all processors. Defaults to -1.
    
    Returns:
        pd.DataFrame: Results with coefficients, p-values, and adjusted p-values for each CpG.
    """
    if method not in ["ols", "logistic"]:
        raise ValueError("Method must be 'ols' or 'logistic'")
    
    # Step 1: Convert logits to beta values
    beta = logit_to_beta(beta_logits)
    
    # Step
    if method == "ols":
        # Convert beta to M-values if requested
        X = design_matrix(clinical_df, formula)
        
        # Get variable of interest index
        if variable_of_interest is None:
            # Use first non-intercept term
            non_intercept_cols = [col for col in X.columns if 'Intercept' not in col]
            if not non_intercept_cols:
                raise ValueError("No non-intercept terms found in design matrix")
            variable_of_interest = non_intercept_cols[0]
        
        if variable_of_interest not in X.columns:
            # Check if it's a categorical variable encoded with prefixes
            possible_cols = [col for col in X.columns if col.startswith(f"{variable_of_interest}[")]
            if possible_cols:
                variable_of_interest = possible_cols[0]
            else:
                raise ValueError(f"Variable {variable_of_interest} not found in design matrix")
        
        var_idx = list(X.columns).index(variable_of_interest)
        
        # Prepare data for modeling
        if convert_to_mvalues:
            model_data = beta_to_m(beta)
        else:
            model_data = beta_logits
        
        # Fit OLS in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_ols_single)(model_data[:, i], X)
            for i in range(model_data.shape[1])
        )
        
        # Unpack results
        coeffs = np.array([r[0][var_idx] for r in results])
        pvals = np.array([r[1][var_idx] for r in results])
        
        # Calculate effect size in beta scale
        if convert_to_mvalues:
            # For M-values, compute delta beta by back-converting
            # Get the intercept and variable coefficient for each CpG
            intercepts = np.array([r[0][0] for r in results])
            # Compute predicted M-values for each group
            baseline_m = intercepts
            effect_m = intercepts + coeffs
            # Convert back to beta scale
            baseline_beta = m_to_beta(baseline_m)
            effect_beta = m_to_beta(effect_m)
            # Delta beta is the difference
            delta_beta = effect_beta - baseline_beta
        else:
            # For logit scale, convert directly
            # Get the intercept and variable coefficient for each CpG
            intercepts = np.array([r[0][0] for r in results])
            # Compute predicted logits for each group
            baseline_logit = intercepts
            effect_logit = intercepts + coeffs
            # Convert to beta scale
            baseline_beta = logit_to_beta(baseline_logit)
            effect_beta = logit_to_beta(effect_logit)
            # Delta beta is the difference
            delta_beta = effect_beta - baseline_beta
    
    elif method == "logistic":
        # For logistic regression, we're predicting disease from methylation
        if CLINICAL_DISEASE_COL not in clinical_df.columns:
            raise ValueError(f"Disease column {CLINICAL_DISEASE_COL} not found in clinical data")
        
        # Need to adapt formula to predict disease status
        # Extract covariates from original formula
        covariates = re.sub(r'^\s*~\s*', '', formula)
        covariates = re.sub(r'\b' + re.escape(CLINICAL_DISEASE_COL) + r'\b', '', covariates)
        covariates = re.sub(r'\+\s*\+', '+', covariates).strip(' +')
        
        # Fit logistic regression for each CpG
        def _fit_logistic_single(cpg_idx):
            # Prepare data
            if convert_to_mvalues:
                methylation = beta_to_m(beta[:, cpg_idx])
            else:
                methylation = beta_logits[:, cpg_idx]
            
            # Create temporary dataframe with methylation and covariates
            temp_df = clinical_df.copy()
            temp_df['Methylation'] = methylation
            
            # Formula: disease ~ methylation + covariates
            if covariates:
                temp_formula = f"{CLINICAL_DISEASE_COL} ~ Methylation + {covariates}"
            else:
                temp_formula = f"{CLINICAL_DISEASE_COL} ~ Methylation"
            
            # Create design matrix
            y, X = dmatrices(temp_formula, data=temp_df, return_type='dataframe')
            
            # Fit logistic regression
            try:
                model = sm.Logit(y, X)
                results = model.fit(disp=0)  # Suppress convergence messages
                methylation_idx = list(X.columns).index('Methylation')
                coef = results.params[methylation_idx]
                pval = results.pvalues[methylation_idx]
                
                # Convert log-odds to delta probability
                # Simple approximation: delta prob ≈ coef * p * (1-p)
                # where p is the baseline probability
                disease_prob = clinical_df[CLINICAL_DISEASE_COL].mean()
                if isinstance(disease_prob, (int, float)):
                    # If disease is encoded as 0/1
                    delta_prob = coef * disease_prob * (1 - disease_prob)
                else:
                    # If disease is categorical, use 0.5 as approximation
                    delta_prob = coef * 0.25
                
                return coef, pval, delta_prob
            except:
                return np.nan, np.nan, np.nan
        
        # Run in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_logistic_single)(i)
            for i in range(beta.shape[1])
        )
        
        # Unpack results
        coeffs = np.array([r[0] for r in results])
        pvals = np.array([r[1] for r in results])
        delta_beta = np.array([r[2] for r in results])
    
    # Adjust p-values
    # Set a reasonable threshold for number of tests to adjust
    valid_pvals = ~np.isnan(pvals)
    adjusted_pvals = np.full_like(pvals, np.nan)
    if np.sum(valid_pvals) > 0:
        adjusted_pvals[valid_pvals] = multipletests(pvals[valid_pvals], method='fdr_bh')[1]
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'effect_size': coeffs,
        'delta_beta': delta_beta,
        'p_value': pvals,
        'fdr': adjusted_pvals
    })
    
    return results_df

def test_differential_methylation(beta_logits, clinical_df, formula, variable_of_interest=None, 
                                 method="ols", convert_to_mvalues=True, use_r=False, n_jobs=-1):
    """
    Performs differential methylation testing using Python implementation.
    
    Args:
        beta_logits (np.ndarray): Matrix of beta values in logit space (samples x CpGs).
        clinical_df (pd.DataFrame): Clinical data with covariates.
        formula (str): Model formula (e.g., "~ Disease + Age + Sex").
        variable_of_interest (str, optional): Variable to extract effect size and p-value for.
            If None, uses first non-intercept term. Defaults to None.
        method (str, optional): Statistical method, one of "ols" or "logistic". Defaults to "ols".
        convert_to_mvalues (bool, optional): Whether to convert beta to M-values. Defaults to True.
        use_r (bool, optional): Ignored parameter (kept for backward compatibility). Python implementation is always used.
        n_jobs (int, optional): Number of parallel jobs for Python implementation. Defaults to -1.
    
    Returns:
        pd.DataFrame: Results with coefficients, p-values, and adjusted p-values for each CpG.
    """
    if use_r:
        warnings.warn("R implementation is not available. Using Python implementation instead.")
        
    return test_differential_methylation_python(
        beta_logits, clinical_df, formula, variable_of_interest, method, convert_to_mvalues, n_jobs
    )

def find_dmr(cpg_results, genes_df, cpg_ids, max_gap=500, min_cpgs=3, p_thresh=0.01, 
             effect_direction_consistency=True):
    """
    Finds differentially methylated regions (DMRs) by grouping nearby significant CpGs.
    
    Args:
        cpg_results (pd.DataFrame): Results from differential methylation testing.
        genes_df (pd.DataFrame): DataFrame mapping CpGs to genes and genomic positions.
        cpg_ids (list): List of CpG IDs in the same order as rows in cpg_results.
        max_gap (int, optional): Maximum gap between CpGs to be considered in the same region (in bp).
            Defaults to 500.
        min_cpgs (int, optional): Minimum number of CpGs needed to form a DMR. Defaults to 3.
        p_thresh (float, optional): P-value threshold for CpGs to be considered in DMRs. Defaults to 0.01.
        effect_direction_consistency (bool, optional): Whether to require consistent effect direction
            within a DMR. Defaults to True.
    
    Returns:
        pd.DataFrame: DMR results with region coordinates, statistics, and gene annotations.
    """
    # Add CpG IDs to results
    results_with_ids = cpg_results.copy()
    results_with_ids['cpg_id'] = cpg_ids
    
    # Merge with genomic positions from genes_df
    if 'chromosome' not in genes_df.columns or 'position' not in genes_df.columns:
        raise ValueError("genes_df must contain 'chromosome' and 'position' columns")
    
    merged = pd.merge(results_with_ids, genes_df, left_on='cpg_id', right_on=GENES_CPG_COL)
    
    # Filter by significance
    significant = merged[merged['p_value'] < p_thresh].copy()
    
    if len(significant) == 0:
        print("No significant CpGs found. Consider increasing p_thresh.")
        return pd.DataFrame()
    
    # Sort by chromosome and position
    significant.sort_values(['chromosome', 'position'], inplace=True)
    
    # Group into regions
    dmrs = []
    current_dmr = []
    
    for i, row in significant.iterrows():
        if not current_dmr:
            # Start a new DMR
            current_dmr = [row]
        else:
            last_cpg = current_dmr[-1]
            
            # Check if on same chromosome and within distance threshold
            same_chrom = row['chromosome'] == last_cpg['chromosome']
            close_enough = abs(row['position'] - last_cpg['position']) <= max_gap
            
            # Check effect direction consistency if required
            direction_ok = True
            if effect_direction_consistency:
                sign_match = np.sign(row['delta_beta']) == np.sign(last_cpg['delta_beta'])
                direction_ok = sign_match
            
            if same_chrom and close_enough and direction_ok:
                # Add to current DMR
                current_dmr.append(row)
            else:
                # Process the completed DMR if it meets minimum size
                if len(current_dmr) >= min_cpgs:
                    dmrs.append(process_dmr(current_dmr))
                
                # Start a new DMR
                current_dmr = [row]
    
    # Process the last DMR if it meets minimum size
    if len(current_dmr) >= min_cpgs:
        dmrs.append(process_dmr(current_dmr))
    
    # Convert to DataFrame
    if dmrs:
        dmr_df = pd.DataFrame(dmrs)
        return dmr_df
    else:
        print("No DMRs found meeting criteria.")
        return pd.DataFrame()

def find_dmr_interval_tree(cpg_results, genes_df, cpg_ids, max_gap=500, min_cpgs=3, p_thresh=0.01, 
                          effect_direction_consistency=True):
    """
    Finds differentially methylated regions (DMRs) using interval trees for efficient grouping 
    of nearby significant CpGs based on chromosomal proximity.
    
    Args:
        cpg_results (pd.DataFrame): Results from differential methylation testing.
        genes_df (pd.DataFrame): DataFrame mapping CpGs to genes and genomic positions.
        cpg_ids (list): List of CpG IDs in the same order as rows in cpg_results.
        max_gap (int, optional): Maximum gap between CpGs to be considered in the same region (in bp).
            Defaults to 500.
        min_cpgs (int, optional): Minimum number of CpGs needed to form a DMR. Defaults to 3.
        p_thresh (float, optional): P-value threshold for CpGs to be considered in DMRs. Defaults to 0.01.
        effect_direction_consistency (bool, optional): Whether to require consistent effect direction
            within a DMR. Defaults to True.
    
    Returns:
        pd.DataFrame: DMR results with region coordinates, statistics, and gene annotations.
    """
    # Import interval tree package
    try:
        from intervaltree import IntervalTree
    except ImportError:
        raise ImportError("The 'intervaltree' package is required. Install with 'pip install intervaltree'.")
    
    # Add CpG IDs to results
    results_with_ids = cpg_results.copy()
    results_with_ids['cpg_id'] = cpg_ids
    
    # Merge with genomic positions from genes_df
    if 'chromosome' not in genes_df.columns or 'position' not in genes_df.columns:
        raise ValueError("genes_df must contain 'chromosome' and 'position' columns")
    
    merged = pd.merge(results_with_ids, genes_df, left_on='cpg_id', right_on=GENES_CPG_COL)
    
    # Filter by significance
    significant = merged[merged['p_value'] < p_thresh].copy()
    
    if len(significant) == 0:
        print("No significant CpGs found. Consider increasing p_thresh.")
        return pd.DataFrame()
    
    # Group CpGs by chromosome
    chrom_groups = significant.groupby('chromosome')
    
    # Create interval tree for each chromosome
    dmrs = []
    for chrom, group in chrom_groups:
        # Sort by position
        group = group.sort_values('position')
        
        # Create interval tree
        tree = IntervalTree()
        
        # Define "effect sign" vector if needed
        if effect_direction_consistency:
            effect_signs = [np.sign(row['delta_beta']) for _, row in group.iterrows()]
        
        # Process each CpG on this chromosome
        for i, (_, cpg) in enumerate(group.iterrows()):
            position = int(cpg['position'])
            
            # Determine interval bounds
            start = position - max_gap
            end = position + max_gap
            
            # Find overlapping intervals
            overlaps = tree.overlap(start, end)
            
            # Check if this CpG can join an existing cluster
            merged_with_existing = False
            for interval in sorted(overlaps):
                interval_data = interval.data
                
                # Check effect direction consistency if required
                direction_ok = True
                if effect_direction_consistency:
                    interval_sign = interval_data['sign']
                    cpg_sign = np.sign(cpg['delta_beta'])
                    direction_ok = (interval_sign == cpg_sign)
                
                if direction_ok:
                    # Update interval bounds
                    new_start = min(interval.begin, start)
                    new_end = max(interval.end, end)
                    
                    # Update interval data
                    new_data = interval_data.copy()
                    new_data['cpgs'].append(cpg)
                    
                    # Remove old interval and add updated one
                    tree.remove(interval)
                    tree.addi(new_start, new_end, new_data)
                    
                    merged_with_existing = True
                    break
            
            # If not merged with existing, create new interval
            if not merged_with_existing:
                sign = np.sign(cpg['delta_beta']) if effect_direction_consistency else None
                tree.addi(start, end, {'cpgs': [cpg], 'sign': sign})
        
        # Process intervals and create DMRs
        for interval in sorted(tree):
            interval_data = interval.data
            cpg_group = interval_data['cpgs']
            
            # Only keep intervals with enough CpGs
            if len(cpg_group) >= min_cpgs:
                # Calculate actual interval bounds from CpG positions
                actual_start = min(cpg['position'] for cpg in cpg_group)
                actual_end = max(cpg['position'] for cpg in cpg_group)
                
                # Skip if gaps between consecutive CpGs are too large
                if max_gap > 0:
                    positions = sorted([cpg['position'] for cpg in cpg_group])
                    gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                    if any(gap > max_gap for gap in gaps):
                        continue
                
                # Create DMR from this cluster
                dmrs.append(process_dmr(cpg_group))
    
    # Convert to DataFrame
    if dmrs:
        dmr_df = pd.DataFrame(dmrs)
        return dmr_df
    else:
        print("No DMRs found meeting criteria.")
        return pd.DataFrame()

def process_dmr(cpg_group):
    """Helper function to process a group of CpGs into a DMR summary."""
    start = min(cpg['position'] for cpg in cpg_group)
    end = max(cpg['position'] for cpg in cpg_group)
    chrom = cpg_group[0]['chromosome']
    
    # Count CpGs
    n_cpgs = len(cpg_group)
    
    # Get p-values and combine using Fisher's method
    pvals = [cpg['p_value'] for cpg in cpg_group]
    # Fisher's method: -2 * sum(ln(p))
    fisher_stat = -2 * np.sum(np.log(pvals))
    # Chi-squared with 2*k degrees of freedom
    fisher_p = 1 - stats.chi2.cdf(fisher_stat, 2 * len(pvals))
    
    # Calculate mean effect size
    mean_delta_beta = np.mean([cpg['delta_beta'] for cpg in cpg_group])
    
    # Get gene annotations (could be multiple genes)
    gene_column = GENES_GENE_COL  # From common_utils 
    genes = set(cpg[gene_column] for cpg in cpg_group if gene_column in cpg)
    genes_str = ','.join(sorted(genes))
    
    # Create DMR record
    dmr = {
        'chromosome': chrom,
        'start': start,
        'end': end,
        'n_cpgs': n_cpgs,
        'mean_delta_beta': mean_delta_beta,
        'combined_p': fisher_p,
        'genes': genes_str,
        'cpg_ids': ','.join(cpg['cpg_id'] for cpg in cpg_group)
    }
    
    return dmr

def correct_for_covariates(beta_logits, clinical_df, variable_to_correct, variable_of_interest=None):
    """
    Corrects methylation data for specified covariates using linear regression.
    
    Args:
        beta_logits (np.ndarray): Matrix of beta values in logit space (samples x CpGs).
        clinical_df (pd.DataFrame): Clinical data with covariates.
        variable_to_correct (str or list): Variable(s) to regress out.
        variable_of_interest (str, optional): Variable whose effect should be preserved.
    
    Returns:
        tuple: (corrected_m_values, corrected_beta_logits)
            - corrected_m_values (np.ndarray): M-values with covariates regressed out
            - corrected_beta_logits (np.ndarray): Beta-logits with covariates regressed out
    """
    # Convert beta_logits to M-values
    beta_values = logit_to_beta(beta_logits)
    m_values = beta_to_m(beta_values)
    
    # Create formula for regression
    if isinstance(variable_to_correct, list):
        correction_formula = "~ " + " + ".join(variable_to_correct)
    else:
        correction_formula = f"~ {variable_to_correct}"
    
    # Create design matrix for correction
    correction_matrix = design_matrix(clinical_df, correction_formula)
    
    # Get variable of interest design matrix if specified
    interest_matrix = None
    if variable_of_interest:
        if isinstance(variable_of_interest, list):
            interest_formula = "~ " + " + ".join(variable_of_interest)
        else:
            interest_formula = f"~ {variable_of_interest}"
        interest_matrix = design_matrix(clinical_df, interest_formula)
    
    # Prep array for results
    n_samples, n_cpgs = m_values.shape
    corrected_m_values = np.zeros_like(m_values)
    
    # Perform regression for each CpG
    for cpg_idx in range(n_cpgs):
        y = m_values[:, cpg_idx]
        
        # Fit linear model
        model = sm.OLS(y, correction_matrix)
        try:
            results = model.fit()
            
            # Calculate predicted values from covariates
            predicted = results.predict(correction_matrix)
            
            # Calculate residuals (data with covariates removed)
            residuals = y - predicted
            
            # If variable of interest is specified, add back its effect
            if interest_matrix is not None:
                # Fit model for variable of interest
                interest_model = sm.OLS(y, interest_matrix)
                try:
                    interest_results = interest_model.fit()
                    # Add back effect of variable of interest
                    interest_effect = interest_results.predict(interest_matrix)
                    residuals += interest_effect
                except:
                    pass  # If fit fails, leave as residuals only
            
            # Store the corrected values
            corrected_m_values[:, cpg_idx] = residuals
            
        except:
            # If regression fails, keep original values
            corrected_m_values[:, cpg_idx] = y
    
    # Convert corrected M-values back to beta values and then to logits
    corrected_beta = m_to_beta(corrected_m_values)
    corrected_beta_logits = beta_to_logit(corrected_beta)
    
    return corrected_m_values, corrected_beta_logits

def gene_level_analysis(cpg_results, genes_df, cpg_ids, p_thresh=0.05):
    """
    Maps CpG-level findings to genes and summarizes at the gene level.
    
    Args:
        cpg_results (pd.DataFrame): Results from differential methylation testing.
        genes_df (pd.DataFrame): DataFrame mapping CpGs to genes.
        cpg_ids (list): List of CpG IDs in the same order as rows in cpg_results.
        p_thresh (float, optional): P-value threshold for considering CpGs significant. Defaults to 0.05.
    
    Returns:
        pd.DataFrame: Gene-level summary statistics.
    """
    # Add CpG IDs to results
    results_with_ids = cpg_results.copy()
    results_with_ids['cpg_id'] = cpg_ids
    
    # Merge with gene annotations
    merged = pd.merge(results_with_ids, genes_df, left_on='cpg_id', right_on=GENES_CPG_COL)
    
    # Group by gene
    gene_stats = []
    for gene, group in merged.groupby(GENES_GENE_COL):
        # Count total CpGs for this gene
        total_cpgs = len(group)
        
        # Count significant CpGs
        sig_cpgs = sum(group['p_value'] < p_thresh)
        
        # Calculate mean and min p-value
        mean_p = group['p_value'].mean()
        min_p = group['p_value'].min()
        min_p_cpg = group.loc[group['p_value'].idxmin(), 'cpg_id'] if sig_cpgs > 0 else None
        
        # Calculate mean effect size (for significant CpGs only)
        sig_group = group[group['p_value'] < p_thresh]
        if len(sig_group) > 0:
            mean_delta_beta = sig_group['delta_beta'].mean()
            
            # Count direction of effects for significant CpGs
            n_hyper = sum(sig_group['delta_beta'] > 0)
            n_hypo = sum(sig_group['delta_beta'] < 0)
            effect_consistency = max(n_hyper, n_hypo) / len(sig_group) if len(sig_group) > 0 else 0
        else:
            mean_delta_beta = 0
            n_hyper = 0
            n_hypo = 0
            effect_consistency = 0
        
        # Store gene-level statistics
        gene_stats.append({
            'gene': gene,
            'total_cpgs': total_cpgs,
            'sig_cpgs': sig_cpgs,
            'mean_p': mean_p,
            'min_p': min_p,
            'min_p_cpg': min_p_cpg,
            'mean_delta_beta': mean_delta_beta,
            'n_hyper': n_hyper,
            'n_hypo': n_hypo,
            'effect_consistency': effect_consistency
        })
    
    # Convert to DataFrame and sort by significance
    gene_df = pd.DataFrame(gene_stats)
    gene_df.sort_values('min_p', inplace=True)
    
    return gene_df

def run_pathway_enrichment_python(gene_list, background_genes=None, organism='hsapiens', 
                                  min_term_size=10, max_term_size=500):
    """
    Python implementation of pathway enrichment analysis.
    
    Args:
        gene_list (list): List of gene symbols to test for enrichment.
        background_genes (list, optional): Background gene list. If None, uses all genes. Defaults to None.
        organism (str, optional): Organism identifier. Defaults to 'hsapiens'.
        min_term_size (int, optional): Minimum GO term size. Defaults to 10.
        max_term_size (int, optional): Maximum GO term size. Defaults to 500.
    
    Returns:
        pd.DataFrame: Enrichment results with GO terms and statistics.
    """
    try:
        import gseapy as gp
        print(f"Successfully imported gseapy version: {gp.__version__}")
    except ImportError as e:
        print(f"gseapy import error: {e}")
        print("gseapy not available. Please install with 'pip install gseapy'")
        print("Using simpler hypergeometric test instead")
        return run_simple_enrichment(gene_list, background_genes)
    
    try:
        # Log input parameters for debugging
        print(f"Running enrichment with {len(gene_list)} genes")
        if background_genes:
            print(f"Background size: {len(background_genes)} genes")
        else:
            print("No background genes provided, using default")
        print(f"Organism: {organism}")
        
        # Run enrichment with gseapy
        print("Starting gseapy.enrichr call...")
        enrichr_results = gp.enrichr(
            gene_list=gene_list,
            gene_sets=['GO_Biological_Process_2021', 'KEGG_2021_Human'],
            organism=organism,
            background=background_genes,
            outdir=None,
            cutoff=0.5  # High cutoff to get all results
        )
        print("gseapy.enrichr call completed successfully")
        
        # Get results as DataFrame
        print("Converting results to DataFrame...")
        results_df = enrichr_results.results
        print(f"Raw results shape: {results_df.shape}")
        
        # Filter by term size if info available
        if 'Genes_in_Term' in results_df.columns:
            print("Filtering by term size...")
            results_df['Term_Size'] = results_df['Genes_in_Term'].str.split(',').str.len()
            results_df = results_df[(results_df['Term_Size'] >= min_term_size) & 
                                    (results_df['Term_Size'] <= max_term_size)]
            print(f"After filtering, results shape: {results_df.shape}")
        else:
            print("Warning: 'Genes_in_Term' column not found, skipping term size filtering")
        
        # Sort by adjusted p-value
        results_df.sort_values('Adjusted P-value', inplace=True)
        
        return results_df
    
    except Exception as e:
        print(f"Error in gseapy enrichment: {str(e)}")
        import traceback
        print(f"Detailed error: {traceback.format_exc()}")
        print("Using simpler hypergeometric test instead")
        return run_simple_enrichment(gene_list, background_genes)

def run_simple_enrichment(gene_list, background_genes=None):
    """
    Simple hypergeometric test for gene enrichment when gseapy is not available.
    
    Args:
        gene_list (list): List of genes to test for enrichment
        background_genes (list, optional): Background gene list. If None, creates a dummy background.
    
    Returns:
        pd.DataFrame: Basic enrichment results
    """
    import pandas as pd
    from scipy import stats
    
    # Create a minimal set of mock pathways for testing
    print("Running simple hypergeometric test for enrichment...")
    
    # If no background provided, create a dummy one
    if background_genes is None or len(background_genes) == 0:
        print("No background genes provided, creating dummy background")
        background_genes = gene_list + [f"DUMMY{i}" for i in range(1, 101)]
    
    # Create some basic mock pathways for demonstration
    pathway_data = {
        "Liver Function": ["ALB", "APOE", "APOA1", "HNF4A"],
        "Inflammation": ["TNF", "IL6", "MMP9", "MMP2"],
        "Fibrosis": ["COL1A1", "COL3A1", "TGFB1", "ACTA2"],
        "Metabolism": ["PPARA", "PPARG", "FOXO1", "MYC"],
        "Growth": ["EGFR", "PDGFRB", "TERT", "TGFBI"]
    }
    
    results = []
    
    # For each pathway, calculate hypergeometric test
    for pathway_name, pathway_genes in pathway_data.items():
        # Count overlaps
        pathway_genes_set = set(pathway_genes)
        background_size = len(background_genes)
        pathway_size = len(pathway_genes)
        
        # Genes in list that are also in pathway
        overlap_genes = [g for g in gene_list if g in pathway_genes_set]
        overlap_count = len(overlap_genes)
        
        # Skip if no overlap
        if overlap_count == 0:
            continue
            
        # Calculate hypergeometric p-value
        # p(X ≥ k) = hypergeom.sf(k-1, M, n, N)
        # M: total population (background_size)
        # n: successes in population (pathway_size)
        # N: sample drawn (len(gene_list))
        # k: successes in sample (overlap_count)
        pval = stats.hypergeom.sf(
            overlap_count - 1, 
            background_size, 
            pathway_size, 
            len(gene_list)
        )
        
        # Record result
        results.append({
            'Term': pathway_name,
            'P-value': pval,
            'Adjusted P-value': pval * len(pathway_data),  # Simple Bonferroni correction
            'Genes': ','.join(overlap_genes),
            'Genes_in_Term': ','.join(pathway_genes),
            'Term_Size': pathway_size,
            'Overlap': overlap_count,
            'Gene_list_size': len(gene_list)
        })
    
    # Create DataFrame from results
    if results:
        results_df = pd.DataFrame(results)
        results_df.sort_values('P-value', inplace=True)
        print(f"Simple enrichment found {len(results_df)} significant pathways")
        return results_df
    else:
        print("No enrichments found in simple test")
        return pd.DataFrame()

def run_pathway_enrichment(gene_list, background_genes=None, organism='hsapiens', use_r=False):
    """
    Runs pathway enrichment analysis using Python implementation.
    
    Args:
        gene_list (list): List of gene symbols to test for enrichment.
        background_genes (list, optional): Background gene list. If None, uses all genes. Defaults to None.
        organism (str, optional): Organism identifier. Defaults to 'hsapiens'.
        use_r (bool, optional): Ignored parameter (kept for backward compatibility). Python implementation is always used.
    
    Returns:
        pd.DataFrame: Enrichment results with GO terms and statistics.
    """
    if use_r:
        warnings.warn("R implementation is not available. Using Python implementation instead.")
        
    return run_pathway_enrichment_python(gene_list, background_genes, organism)

def benchmark_dmr_methods(cpg_results, genes_df, cpg_ids, max_gap=500, min_cpgs=3, p_thresh=0.01, 
                         effect_direction_consistency=True, runs=3):
    """
    Benchmark and compare the performance of different DMR finding methods.
    
    Args:
        cpg_results (pd.DataFrame): Results from differential methylation testing.
        genes_df (pd.DataFrame): DataFrame mapping CpGs to genes and genomic positions.
        cpg_ids (list): List of CpG IDs in the same order as rows in cpg_results.
        max_gap (int, optional): Maximum gap between CpGs to be considered in the same region (in bp).
            Defaults to 500.
        min_cpgs (int, optional): Minimum number of CpGs needed to form a DMR. Defaults to 3.
        p_thresh (float, optional): P-value threshold for CpGs to be considered in DMRs. Defaults to 0.01.
        effect_direction_consistency (bool, optional): Whether to require consistent effect direction
            within a DMR. Defaults to True.
        runs (int, optional): Number of benchmark runs to perform. Defaults to 3.
    
    Returns:
        dict: Dictionary with benchmark results and comparison of output DMRs.
    """
    import time
    
    # Check if intervaltree is available
    try:
        from intervaltree import IntervalTree
        interval_tree_available = True
    except ImportError:
        print("Warning: 'intervaltree' package not found. Skipping interval tree method benchmark.")
        interval_tree_available = False
    
    # Function to compare DataFrames of DMRs
    def compare_dmr_dataframes(df1, df2):
        if df1.empty and df2.empty:
            return {"match": True, "reason": "Both empty"}
        
        if df1.empty != df2.empty:
            return {"match": False, "reason": "One DataFrame is empty, the other is not"}
        
        # Check if both have same number of DMRs
        if len(df1) != len(df2):
            return {"match": False, "reason": f"Different number of DMRs: {len(df1)} vs {len(df2)}"}
        
        # Sort both DataFrames to ensure consistent comparison
        df1_sorted = df1.sort_values(['chromosome', 'start']).reset_index(drop=True)
        df2_sorted = df2.sort_values(['chromosome', 'start']).reset_index(drop=True)
        
        # Compare key columns
        for col in ['chromosome', 'start', 'end', 'n_cpgs']:
            if not df1_sorted[col].equals(df2_sorted[col]):
                return {"match": False, "reason": f"Differences in column '{col}'"}
        
        # Check if same CpGs are included
        cpg_matches = all(
            set(df1_sorted.loc[i, 'cpg_ids'].split(',')) == 
            set(df2_sorted.loc[i, 'cpg_ids'].split(','))
            for i in range(len(df1_sorted))
        )
        
        if not cpg_matches:
            return {"match": False, "reason": "Different CpGs in DMRs"}
        
        return {"match": True, "reason": "All DMR regions match"}
    
    # Benchmark results
    results = {
        "sequential_method": {
            "times": [],
            "dmrs": None
        }
    }
    
    if interval_tree_available:
        results["interval_tree_method"] = {
            "times": [],
            "dmrs": None
        }
    
    # Run benchmarks
    print(f"Running benchmark with {runs} iterations...")
    
    for i in range(runs):
        print(f"Iteration {i+1}/{runs}")
        
        # Benchmark sequential method
        start_time = time.time()
        sequential_dmrs = find_dmr(
            cpg_results, genes_df, cpg_ids, max_gap, min_cpgs, p_thresh, effect_direction_consistency
        )
        seq_time = time.time() - start_time
        results["sequential_method"]["times"].append(seq_time)
        
        if i == 0:
            results["sequential_method"]["dmrs"] = sequential_dmrs
        
        print(f"  Sequential method: {seq_time:.4f} seconds, {len(sequential_dmrs)} DMRs found")
        
        # Benchmark interval tree method if available
        if interval_tree_available:
            start_time = time.time()
            tree_dmrs = find_dmr_interval_tree(
                cpg_results, genes_df, cpg_ids, max_gap, min_cpgs, p_thresh, effect_direction_consistency
            )
            tree_time = time.time() - start_time
            results["interval_tree_method"]["times"].append(tree_time)
            
            if i == 0:
                results["interval_tree_method"]["dmrs"] = tree_dmrs
            
            print(f"  Interval tree method: {tree_time:.4f} seconds, {len(tree_dmrs)} DMRs found")
    
    # Calculate average times
    results["sequential_method"]["avg_time"] = sum(results["sequential_method"]["times"]) / runs
    
    if interval_tree_available:
        results["interval_tree_method"]["avg_time"] = sum(results["interval_tree_method"]["times"]) / runs
        
        # Compare output DMRs from both methods
        results["comparison"] = compare_dmr_dataframes(
            results["sequential_method"]["dmrs"], 
            results["interval_tree_method"]["dmrs"]
        )
    
    # Print summary
    print("\nBenchmark Results:")
    print(f"Sequential method: {results['sequential_method']['avg_time']:.4f} seconds average")
    
    if interval_tree_available:
        print(f"Interval tree method: {results['interval_tree_method']['avg_time']:.4f} seconds average")
        speedup = results["sequential_method"]["avg_time"] / results["interval_tree_method"]["avg_time"]
        print(f"Speedup: {speedup:.2f}x")
        
        if results["comparison"]["match"]:
            print("DMR outputs match! Reason:", results["comparison"]["reason"])
        else:
            print("Warning: DMR outputs don't match exactly. Reason:", results["comparison"]["reason"])
    
    return results

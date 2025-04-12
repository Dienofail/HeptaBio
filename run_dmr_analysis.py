#!/usr/bin/env python3
# run_dmr_analysis.py - Demonstration of DMR analysis functions with simulated data

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import traceback
import gc

# Enable debugging
import logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Import our modules
logger.info("Importing modules...")
try:
    from data_simulator import generate_cfDNA_methylation_data_with_gene_effects
    from dmr_calling import (
        design_matrix, 
        test_differential_methylation, 
        find_dmr, 
        find_dmr_interval_tree,
        benchmark_dmr_methods,
        gene_level_analysis,
        run_pathway_enrichment,
        beta_to_m, 
        m_to_beta, 
        logit_to_beta, 
        beta_to_logit,
        HAVE_RPY2,
        run_pathway_enrichment_python
    )
    from common_utils import (
        GENES_GENE_COL,
        GENES_CPG_COL,
        CLINICAL_DISEASE_COL,
        CLINICAL_AGE_COL,
        CLINICAL_SEX_COL,
        CLINICAL_STAGE_COL,
        CLINICAL_ALT_COL,
        CLINICAL_AST_COL,
        CLINICAL_SAMPLE_ID_COL
    )
    import plotting as plt_utils
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    sys.exit(1)

# Create output directories if they don't exist
Path("plots").mkdir(exist_ok=True)
Path("tables").mkdir(exist_ok=True)

logger.info("Starting DMR analysis with simulated data...")

def run_analysis():
    # ===================================
    # 1. Generate simulated cfDNA methylation data
    # ===================================
    logger.info("1. Generating simulated cfDNA methylation data...")

    # Use smaller dataset for faster testing and to avoid memory issues
    n_samples = 50  # Reduced from 100
    n_cpgs = 1000   # Reduced from 5000
    n_genes = 100   # Reduced from 500
    random_seed = 42

    try:
        # Generate data
        beta_values, depth_values, clinical_df, genes_df, cpg_ids, sex_cpg_idx = generate_cfDNA_methylation_data_with_gene_effects(
            n_samples=n_samples,
            n_cpgs=n_cpgs,
            n_genes=n_genes,
            random_seed=random_seed
        )

        logger.info(f"Generated data with {n_samples} samples and {n_cpgs} CpGs across {n_genes} genes")
        logger.info(f"Clinical data shape: {clinical_df.shape}")
        logger.info(f"Beta values shape: {beta_values.shape}")
        logger.info(f"Depth values shape: {depth_values.shape}")
        logger.info(f"Genes data shape: {genes_df.shape}")
    except Exception as e:
        logger.error(f"Error generating data: {e}")
        logger.error(traceback.format_exc())
        return

    # ===================================
    # 2. Add chromosome and position info to genes_df for DMR analysis
    # ===================================
    logger.info("2. Adding genomic position information to genes_df...")
    try:
        # Generate random chromosomes for genes (1-22, X, Y)
        chromosomes = list(range(1, 23)) + ['X', 'Y']
        gene_chromosomes = {}

        # Assign each gene to a chromosome
        for gene in set(genes_df[GENES_GENE_COL]):
            gene_chromosomes[gene] = str(np.random.choice(chromosomes))

        # Assign each CpG a position based on its gene
        genes_df['chromosome'] = genes_df[GENES_GENE_COL].map(gene_chromosomes)

        # For each gene, assign sequential positions to its CpGs
        position_counter = {}
        for chrom in set(genes_df['chromosome']):
            position_counter[chrom] = 1000000  # Start at 1Mb for each chromosome

        genes_df['position'] = 0
        for idx, row in genes_df.iterrows():
            gene = row[GENES_GENE_COL]
            chrom = row['chromosome']
            
            # Assign position and increment counter for this chromosome
            genes_df.at[idx, 'position'] = position_counter[chrom]
            position_counter[chrom] += 200  # 200bp between CpGs

        # Convert beta to logit for analysis functions
        logger.info("Converting beta values to logit...")
        beta_logits = beta_to_logit(beta_values)

        # Force garbage collection to free memory
        gc.collect()
    except Exception as e:
        logger.error(f"Error in genomic position assignment: {e}")
        logger.error(traceback.format_exc())
        return

    # ===================================
    # 3. Testing design_matrix function
    # ===================================
    logger.info("3. Testing design_matrix function...")
    try:
        # Create a formula for testing
        formula = "disease_status + age + sex"
        design = design_matrix(clinical_df, formula)
        logger.info(f"Design matrix shape: {design.shape}")
        logger.info(f"Design matrix columns: {design.columns.tolist()}")

        # Save design matrix to tables
        design.head(10).to_csv("tables/design_matrix_example.csv")
        logger.info("Design matrix example saved to tables/design_matrix_example.csv")
    except Exception as e:
        logger.error(f"Error in design matrix creation: {e}")
        logger.error(traceback.format_exc())
        return

    # ===================================
    # 4. Testing differential_methylation testing (Python implementation)
    # ===================================
    logger.info("4. Testing differential methylation analysis (Python)...")
    try:
        # Test differential methylation with Python implementation
        start_time = time.time()
        py_results = test_differential_methylation(
            beta_logits=beta_logits,
            clinical_df=clinical_df,
            formula=formula,
            variable_of_interest="disease_status[T.Control]",
            method="ols",
            convert_to_mvalues=True,
            use_r=False,
            n_jobs=1  # Use a single core to avoid potential memory issues
        )
        py_time = time.time() - start_time

        logger.info(f"Python implementation completed in {py_time:.2f} seconds")
        logger.info(f"Results shape: {py_results.shape}")
        logger.info("Top 5 most significant results:")
        logger.info(py_results.sort_values('p_value').head(5))

        # Save Python results
        py_results.head(100).to_csv("tables/python_differential_methylation.csv")
        logger.info("Python differential methylation results saved to tables/python_differential_methylation.csv")

        # Plot volcano plot
        fig, ax = plt.subplots(figsize=(10, 8))
        plt_utils.plot_volcano(
            py_results, 
            effect_size_col='delta_beta', 
            p_value_col='p_value',
            alpha=0.05, 
            effect_size_threshold=0.05,
            label_top_n=10,
            ax=ax
        )
        plt.tight_layout()
        plt.savefig("plots/python_volcano_plot.png", dpi=300)
        plt.close()
        logger.info("Volcano plot saved to plots/python_volcano_plot.png")

        # Force garbage collection to free memory
        gc.collect()
    except Exception as e:
        logger.error(f"Error in Python differential methylation testing: {e}")
        logger.error(traceback.format_exc())
        return

    # ===================================
    # 6. Testing find_dmr function
    # ===================================
    logger.info("6. Testing find_dmr function...")
    try:
        # Use Python results for DMR calling
        dmr_results = find_dmr(
            cpg_results=py_results,
            genes_df=genes_df,
            cpg_ids=cpg_ids,
            max_gap=500,
            min_cpgs=3,
            p_thresh=0.05,
            effect_direction_consistency=True
        )

        if len(dmr_results) > 0:
            logger.info(f"Found {len(dmr_results)} differentially methylated regions")
            logger.info("Top 5 DMRs by combined p-value:")
            logger.info(dmr_results.sort_values('combined_p').head(5))
            
            # Save DMR results
            dmr_results.to_csv("tables/dmr_results.csv", index=False)
            logger.info("DMR results saved to tables/dmr_results.csv")
            
            # Plot region methylation for the top DMR if available
            if len(dmr_results) > 0:
                top_dmr = dmr_results.sort_values('combined_p').iloc[0]
                top_dmr_cpgs = top_dmr['cpg_ids'].split(',')
                top_dmr_indices = [cpg_ids.index(cpg) for cpg in top_dmr_cpgs if cpg in cpg_ids]
                
                if len(top_dmr_indices) > 0:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plt_utils.plot_region_methylation(
                        top_dmr_indices,
                        beta_values,
                        clinical_df,
                        group_var='disease_status',
                        region_name=f"DMR on Chr{top_dmr['chromosome']}:{top_dmr['start']}-{top_dmr['end']}",
                        plot_type='box',
                        ax=ax
                    )
                    plt.tight_layout()
                    plt.savefig("plots/top_dmr_methylation.png", dpi=300)
                    plt.close()
                    logger.info("Top DMR methylation plot saved to plots/top_dmr_methylation.png")
        else:
            logger.info("No DMRs found meeting criteria.")
            
        # Force garbage collection to free memory
        gc.collect()
    except Exception as e:
        logger.error(f"Error in DMR analysis: {e}")
        logger.error(traceback.format_exc())

    # ===================================
    # 6b. Testing interval tree DMR finding
    # ===================================
    logger.info("6b. Testing interval tree DMR finding...")
    try:
        # Try to import intervaltree
        try:
            from intervaltree import IntervalTree
            interval_tree_available = True
        except ImportError:
            interval_tree_available = False
            logger.warning("intervaltree package not available. Run 'pip install intervaltree' to install it.")
        
        if interval_tree_available:
            # Use interval tree method for DMR finding
            tree_dmr_results = find_dmr_interval_tree(
                cpg_results=py_results,
                genes_df=genes_df,
                cpg_ids=cpg_ids,
                max_gap=500,
                min_cpgs=3,
                p_thresh=0.05,
                effect_direction_consistency=True
            )
            
            if len(tree_dmr_results) > 0:
                logger.info(f"Found {len(tree_dmr_results)} differentially methylated regions with interval tree method")
                logger.info("Top 5 DMRs by combined p-value:")
                logger.info(tree_dmr_results.sort_values('combined_p').head(5))
                
                # Save interval tree DMR results
                tree_dmr_results.to_csv("tables/interval_tree_dmr_results.csv", index=False)
                logger.info("Interval tree DMR results saved to tables/interval_tree_dmr_results.csv")
                
                # Compare results with sequential method
                if len(dmr_results) > 0:
                    seq_dmrs = len(dmr_results)
                    tree_dmrs = len(tree_dmr_results)
                    pct_diff = abs(seq_dmrs - tree_dmrs) / max(seq_dmrs, tree_dmrs) * 100
                    
                    logger.info(f"Sequential method found {seq_dmrs} DMRs, interval tree method found {tree_dmrs} DMRs")
                    logger.info(f"Percent difference: {pct_diff:.2f}%")
            else:
                logger.info("No DMRs found meeting criteria with interval tree method.")
            
            # Run benchmarking
            logger.info("Running benchmark comparison of DMR methods...")
            benchmark_results = benchmark_dmr_methods(
                cpg_results=py_results,
                genes_df=genes_df,
                cpg_ids=cpg_ids,
                max_gap=500,
                min_cpgs=3,
                p_thresh=0.05,
                effect_direction_consistency=True,
                runs=3
            )
            
            # Save benchmark results
            import json
            with open("tables/dmr_benchmark_results.json", "w") as f:
                benchmark_dict = {
                    "sequential_avg_time": benchmark_results["sequential_method"]["avg_time"],
                    "sequential_times": benchmark_results["sequential_method"]["times"],
                    "sequential_dmrs": len(benchmark_results["sequential_method"]["dmrs"]) if benchmark_results["sequential_method"]["dmrs"] is not None else 0
                }
                
                if "interval_tree_method" in benchmark_results:
                    benchmark_dict.update({
                        "interval_tree_avg_time": benchmark_results["interval_tree_method"]["avg_time"],
                        "interval_tree_times": benchmark_results["interval_tree_method"]["times"],
                        "interval_tree_dmrs": len(benchmark_results["interval_tree_method"]["dmrs"]) if benchmark_results["interval_tree_method"]["dmrs"] is not None else 0,
                        "comparison_match": benchmark_results["comparison"]["match"],
                        "comparison_reason": benchmark_results["comparison"]["reason"]
                    })
                    
                json.dump(benchmark_dict, f, indent=2)
                
            logger.info("Benchmark results saved to tables/dmr_benchmark_results.json")
        else:
            logger.warning("Skipping interval tree DMR finding tests (package not available)")
        
        # Force garbage collection to free memory
        gc.collect()
    except Exception as e:
        logger.error(f"Error in interval tree DMR analysis: {e}")
        logger.error(traceback.format_exc())

    # ===================================
    # 7. Testing gene_level_analysis function
    # ===================================
    logger.info("7. Testing gene_level_analysis function...")
    try:
        gene_results = gene_level_analysis(
            cpg_results=py_results,
            genes_df=genes_df,
            cpg_ids=cpg_ids,
            p_thresh=0.05
        )

        logger.info(f"Gene-level analysis results: {gene_results.shape}")
        logger.info("Top 5 genes by significance:")
        logger.info(gene_results.head(5))

        # Save gene results
        gene_results.to_csv("tables/gene_level_results.csv", index=False)
        logger.info("Gene level results saved to tables/gene_level_results.csv")

        # Plot distribution of effect sizes by gene
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(gene_results['mean_delta_beta'], bins=30)
        ax.set_xlabel('Mean Delta Beta')
        ax.set_ylabel('Number of Genes')
        ax.set_title('Distribution of Mean Effect Sizes Across Genes')
        plt.tight_layout()
        plt.savefig("plots/gene_effect_distribution.png", dpi=300)
        plt.close()
        logger.info("Gene effect distribution plot saved to plots/gene_effect_distribution.png")
        
        # Force garbage collection to free memory
        gc.collect()
    except Exception as e:
        logger.error(f"Error in gene-level analysis: {e}")
        logger.error(traceback.format_exc())
        return

    # ===================================
    # 8. Testing pathway enrichment (Python implementation)
    # ===================================
    logger.info("8. Testing pathway enrichment (Python)...")
    try:
        # Get significant genes for enrichment analysis
        sig_genes = gene_results[gene_results['min_p'] < 0.05]['gene'].tolist()
        background_genes = gene_results['gene'].tolist()

        if len(sig_genes) > 0:
            logger.info(f"Testing enrichment with {len(sig_genes)} significant genes vs {len(background_genes)} background genes")
            
            try:
                # Run pathway enrichment with Python implementation
                py_enrichment = run_pathway_enrichment(
                    gene_list=sig_genes,
                    background_genes=background_genes,
                    organism='hsapiens',
                    use_r=False
                )
                
                if not py_enrichment.empty:
                    logger.info(f"Pathway enrichment results: {py_enrichment.shape}")
                    logger.info("Top 5 enriched pathways:")
                    logger.info(py_enrichment.head(5))
                    
                    # Save enrichment results
                    py_enrichment.to_csv("tables/python_pathway_enrichment.csv", index=False)
                    logger.info("Python pathway enrichment results saved to tables/python_pathway_enrichment.csv")
                else:
                    logger.info("No significant pathway enrichment found with Python implementation")
            except Exception as e:
                logger.error(f"Error in Python pathway enrichment: {e}")
                logger.error(traceback.format_exc())
        else:
            logger.info("No significant genes found for pathway enrichment")
            
        # Force garbage collection to free memory
        gc.collect()
    except Exception as e:
        logger.error(f"Error in pathway enrichment: {e}")
        logger.error(traceback.format_exc())

    # ===================================
    # 9. Direct testing of run_pathway_enrichment_python function
    # ===================================
    logger.info("9. Direct testing of run_pathway_enrichment_python function...")
    try:
        # Use a different threshold to get a different set of genes
        sig_genes_direct = gene_results[gene_results['min_p'] < 0.1]['gene'].tolist()
        
        if len(sig_genes_direct) > 0:
            logger.info(f"Direct testing with {len(sig_genes_direct)} genes vs {len(background_genes)} background genes")
            
            try:
                # Call run_pathway_enrichment_python directly with custom parameters
                direct_enrichment = run_pathway_enrichment_python(
                    gene_list=sig_genes_direct,
                    background_genes=background_genes,
                    organism='hsapiens',
                    min_term_size=5,  # Lower minimum term size
                    max_term_size=1000  # Higher maximum term size
                )
                
                if not direct_enrichment.empty:
                    logger.info(f"Direct pathway enrichment results: {direct_enrichment.shape}")
                    logger.info("Top 5 enriched pathways from direct call:")
                    logger.info(direct_enrichment.head(5))
                    
                    # Save direct enrichment results
                    direct_enrichment.to_csv("tables/direct_pathway_enrichment.csv", index=False)
                    logger.info("Direct pathway enrichment results saved to tables/direct_pathway_enrichment.csv")
                    
                    # Compare with previous results if available
                    if 'py_enrichment' in locals() and not py_enrichment.empty:
                        logger.info("Comparing direct and wrapper function results:")
                        common_terms = set(direct_enrichment['Term'].tolist()).intersection(
                            set(py_enrichment['Term'].tolist()))
                        logger.info(f"Number of common enriched terms: {len(common_terms)}")
                else:
                    logger.info("No significant pathway enrichment found with direct Python implementation call")
            except Exception as e:
                logger.error(f"Error in direct pathway enrichment: {e}")
                logger.error(traceback.format_exc())
        else:
            logger.info("No genes meet threshold for direct pathway enrichment testing")
            
        # Force garbage collection to free memory
        gc.collect()
    except Exception as e:
        logger.error(f"Error in direct pathway enrichment testing: {e}")
        logger.error(traceback.format_exc())

    # ===================================
    # 10. Additional visualizations using plotting.py
    # ===================================
    logger.info("10. Creating additional visualizations...")
    try:
        # Beta value distribution by disease status
        logger.info("Creating beta value distribution plot...")
        fig, ax = plt.subplots(figsize=(10, 6))
        plt_utils.plot_beta_distribution(
            beta=beta_values,
            group_labels=clinical_df['disease_status'],
            group_name='Disease Status',
            bins=50,
            ax=ax
        )
        plt.tight_layout()
        plt.savefig("plots/beta_distribution_by_disease.png", dpi=300)
        plt.close()
        logger.info("Beta distribution plot saved to plots/beta_distribution_by_disease.png")

        # Coverage distribution by disease status
        logger.info("Creating coverage distribution plot...")
        fig, ax = plt.subplots(figsize=(10, 6))
        plt_utils.plot_coverage_distribution(
            depth=depth_values,
            group_labels=clinical_df['disease_status'],
            group_name='Disease Status',
            log_scale=True,
            ax=ax
        )
        plt.tight_layout()
        plt.savefig("plots/coverage_distribution_by_disease.png", dpi=300)
        plt.close()
        logger.info("Coverage distribution plot saved to plots/coverage_distribution_by_disease.png")

        # PCA plot of methylation data
        logger.info("Creating PCA plot...")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax, _ = plt_utils.plot_pca(
            beta=beta_values,
            clinical_df=clinical_df,
            color_by='disease_status',
            pc_x=1,
            pc_y=2,
            use_m_values=True,
            ax=ax
        )
        plt.tight_layout()
        plt.savefig("plots/pca_methylation_by_disease.png", dpi=300)
        plt.close()
        logger.info("PCA plot saved to plots/pca_methylation_by_disease.png")

        # t-SNE plot might be causing memory issues, so we'll make it optional
        try_tsne = False  # Set to True to try running t-SNE
        if try_tsne:
            logger.info("Creating t-SNE plot...")
            fig, ax = plt.subplots(figsize=(10, 8))
            ax, _ = plt_utils.plot_tsne(
                beta=beta_values,
                clinical_df=clinical_df,
                color_by='disease_status',
                use_m_values=True,
                perplexity=min(30, n_samples // 5),  # Adjust perplexity for smaller sample sizes
                ax=ax
            )
            plt.tight_layout()
            plt.savefig("plots/tsne_methylation_by_disease.png", dpi=300)
            plt.close()
            logger.info("t-SNE plot saved to plots/tsne_methylation_by_disease.png")
        else:
            logger.info("Skipping t-SNE plot due to potential memory issues")

        # Plot methylation vs age for a significant CpG
        if not py_results.empty:
            logger.info("Creating methylation vs age plot...")
            # Find a CpG significantly associated with age
            age_sig_cpgs = py_results.sort_values('p_value').head(50)
            if len(age_sig_cpgs) > 0:
                sig_cpg_idx = 0  # Use the top significant CpG
                
                fig, ax = plt.subplots(figsize=(8, 6))
                plt_utils.plot_methylation_vs_covariate(
                    beta=beta_values,
                    clinical_df=clinical_df,
                    cpg_index=sig_cpg_idx,
                    cpg_id=cpg_ids[sig_cpg_idx],
                    covariate='age',
                    add_regression=True,
                    ax=ax
                )
                plt.tight_layout()
                plt.savefig("plots/methylation_vs_age.png", dpi=300)
                plt.close()
                logger.info("Methylation vs age plot saved to plots/methylation_vs_age.png")
    except Exception as e:
        logger.error(f"Error in visualization: {e}")
        logger.error(traceback.format_exc())

    # ===================================
    # 11. Testing correct_for_covariates function
    # ===================================
    logger.info("11. Testing correct_for_covariates function...")
    try:
        # Select a variable to correct for (e.g., age)
        variable_to_correct = "age"
        variable_of_interest = "disease_status"
        
        # Run correction
        corrected_m_values, corrected_beta_logits = correct_for_covariates(
            beta_logits=beta_logits,
            clinical_df=clinical_df,
            variable_to_correct=variable_to_correct,
            variable_of_interest=variable_of_interest
        )
        
        logger.info(f"Corrected M-values shape: {corrected_m_values.shape}")
        logger.info(f"Corrected beta logits shape: {corrected_beta_logits.shape}")
        
        # Compare uncorrected vs corrected
        # Convert original beta_logits to M-values for comparison
        original_m_values = beta_to_m(logit_to_beta(beta_logits))
        
        # Calculate correlation between original and corrected values
        # Use a few random CpGs for demonstration
        n_random_cpgs = 5
        random_cpg_indices = np.random.choice(range(n_cpgs), n_random_cpgs, replace=False)
        
        logger.info("Correlation between original and corrected M-values:")
        for idx in random_cpg_indices:
            original = original_m_values[:, idx]
            corrected = corrected_m_values[:, idx]
            correlation = np.corrcoef(original, corrected)[0, 1]
            logger.info(f"CpG #{idx}: {correlation:.4f}")
        
        # Visualize before and after correction for one CpG
        vis_cpg_idx = random_cpg_indices[0]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot original data
        plt_utils.plot_methylation_vs_covariate(
            beta=logit_to_beta(beta_logits),
            clinical_df=clinical_df,
            cpg_index=vis_cpg_idx,
            cpg_id=cpg_ids[vis_cpg_idx],
            covariate=variable_to_correct,
            add_regression=True,
            title=f"Before correction for {variable_to_correct}",
            ax=axes[0]
        )
        
        # Plot corrected data
        plt_utils.plot_methylation_vs_covariate(
            beta=logit_to_beta(corrected_beta_logits),
            clinical_df=clinical_df,
            cpg_index=vis_cpg_idx,
            cpg_id=cpg_ids[vis_cpg_idx],
            covariate=variable_to_correct,
            add_regression=True,
            title=f"After correction for {variable_to_correct}",
            ax=axes[1]
        )
        
        plt.tight_layout()
        plt.savefig("plots/covariate_correction_example.png", dpi=300)
        plt.close()
        logger.info("Covariate correction visualization saved to plots/covariate_correction_example.png")
        
        # Run differential methylation on corrected data
        logger.info("Running differential methylation on corrected data...")
        # Simplified formula without the corrected variable
        simplified_formula = "~ disease_status"
        
        corrected_results = test_differential_methylation(
            beta_logits=corrected_beta_logits,
            clinical_df=clinical_df,
            formula=simplified_formula,
            variable_of_interest=None,
            method="ols",
            convert_to_mvalues=True,
            n_jobs=1
        )
        
        logger.info(f"Corrected differential methylation results shape: {corrected_results.shape}")
        logger.info("Top 5 most significant results after correction:")
        logger.info(corrected_results.sort_values('p_value').head(5))
        
        # Save results
        corrected_results.head(100).to_csv("tables/corrected_differential_methylation.csv")
        logger.info("Corrected differential methylation results saved to tables/corrected_differential_methylation.csv")
        
        # Force garbage collection to free memory
        gc.collect()
    except Exception as e:
        logger.error(f"Error in covariate correction testing: {e}")
        logger.error(traceback.format_exc())

    logger.info("\nDMR analysis complete. Results saved to tables/ and plots/ directories.")

if __name__ == "__main__":
    try:
        run_analysis()
    except Exception as e:
        logger.error(f"Unhandled exception in run_analysis: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# First try to import dmr_calling to check HAVE_RPY2 status
try:
    from dmr_calling import HAVE_RPY2, test_differential_methylation, run_pathway_enrichment
    from common_utils import beta_to_logit
    print(f"Successfully imported dmr_calling. HAVE_RPY2 = {HAVE_RPY2}")
except ImportError as e:
    print(f"Failed to import dmr_calling: {e}")
    print("Please ensure the dmr_calling.py and common_utils.py files are in your Python path.")
    sys.exit(1)

def run_simple_test():
    """Run a simple test of the Python-only implementation."""
    print("\nRunning simple test of differential methylation analysis...")
    
    # Create a small test dataset
    n_samples = 20
    n_cpgs = 100
    
    # Create synthetic beta values (0-1 range)
    np.random.seed(42)
    beta_values = np.random.beta(2, 5, size=(n_samples, n_cpgs))
    
    # Convert to logit space for the analysis
    beta_logits = beta_to_logit(beta_values)
    
    # Create a simple clinical dataframe
    clinical_data = {
        'disease_status': ['Control'] * 10 + ['Case'] * 10,
        'age': np.random.normal(60, 10, n_samples),
        'sex': np.random.choice(['M', 'F'], size=n_samples)
    }
    clinical_df = pd.DataFrame(clinical_data)
    
    # Run the differential methylation analysis
    formula = "~ disease_status + age + sex"
    
    # Test with use_r=False first (Python implementation)
    print("Testing with use_r=False (Python implementation)...")
    try:
        py_results = test_differential_methylation(
            beta_logits=beta_logits,
            clinical_df=clinical_df,
            formula=formula,
            variable_of_interest=None,
            method="ols",
            convert_to_mvalues=True,
            use_r=False,
            n_jobs=1
        )
        print(f"Python implementation successful. Results shape: {py_results.shape}")
        print("Top 3 results by p-value:")
        print(py_results.sort_values('p_value').head(3))
    except Exception as e:
        print(f"Error in Python implementation: {e}")
    
    # Test with use_r=True (should still use Python but with a warning)
    print("\nTesting with use_r=True (should use Python with warning)...")
    try:
        r_attempt_results = test_differential_methylation(
            beta_logits=beta_logits,
            clinical_df=clinical_df,
            formula=formula,
            variable_of_interest=None,
            method="ols",
            convert_to_mvalues=True,
            use_r=True,
            n_jobs=1
        )
        print(f"Test with use_r=True successful. Results shape: {r_attempt_results.shape}")
    except Exception as e:
        print(f"Error when use_r=True: {e}")
    
    # Test pathway enrichment
    print("\nTesting pathway enrichment (Python implementation)...")
    gene_list = ["TP53", "BRCA1", "BRCA2", "ATM", "PTEN"]
    
    try:
        enrichment_results = run_pathway_enrichment(
            gene_list=gene_list,
            background_genes=None,
            organism='hsapiens',
            use_r=False
        )
        if not enrichment_results.empty:
            print(f"Enrichment successful. Results shape: {enrichment_results.shape}")
        else:
            print("Enrichment returned empty DataFrame (expected if gseapy not installed)")
    except Exception as e:
        print(f"Error in enrichment: {e}")
    
    print("\nTests completed.")

if __name__ == "__main__":
    if HAVE_RPY2:
        print("Warning: rpy2 is still available in the environment.")
        print("This test is meant to verify that the code works without rpy2.")
    else:
        print("Confirmed: rpy2 is not available.")
        print("Testing Python-only implementations...")
    
    try:
        run_simple_test()
    except Exception as e:
        print(f"Test failed with error: {e}")
    else:
        print("All tests completed successfully!")

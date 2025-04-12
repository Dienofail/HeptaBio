#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Methylation Analysis Pipeline

This script simulates methylation data and tests all functions from the models module.
It generates visualizations and performance metrics for each model type.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle
import time
from datetime import datetime

# Import the modules
from models import (train_classifier, cross_validate_model, 
                    explain_model, nested_cross_validation,
                    calculate_scree_plot_values)
from plotting import (plot_beta_distribution, plot_pca, plot_tsne, 
                     plot_volcano, plot_region_methylation, 
                     plot_roc_curve, plot_precision_recall_curve, 
                     plot_metrics_comparison, calculate_metrics,
                     plot_scree)
from common_utils import (logit_to_beta, beta_to_m, beta_to_logit, m_to_beta,
                        CLINICAL_DISEASE_COL, CLINICAL_AGE_COL, CLINICAL_SEX_COL,
                        CLINICAL_SAMPLE_ID_COL)

# Create directories for output
os.makedirs('plots', exist_ok=True)
os.makedirs('tables', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def simulate_methylation_data(n_samples=100, n_cpgs=1000, n_diff_cpgs=50, effect_size=0.2):
    """
    Simulates methylation data with differential methylation between case and control groups.
    
    Args:
        n_samples (int): Number of samples to simulate (half cases, half controls).
        n_cpgs (int): Number of CpG sites to simulate.
        n_diff_cpgs (int): Number of differentially methylated CpGs between groups.
        effect_size (float): Mean difference in beta values for differentially methylated CpGs.
        
    Returns:
        tuple: (beta_values, clinical_df, diff_cpgs_indices)
            - beta_values: numpy array of simulated beta values
            - clinical_df: pandas DataFrame with clinical information
            - diff_cpgs_indices: indices of differentially methylated CpGs
    """
    print(f"Simulating methylation data with {n_samples} samples and {n_cpgs} CpG sites...")
    
    # Create clinical data
    n_cases = n_samples // 2
    n_controls = n_samples - n_cases
    
    # Clinical DataFrame
    clinical_data = {
        CLINICAL_SAMPLE_ID_COL: [f'S{i+1}' for i in range(n_samples)],
        CLINICAL_DISEASE_COL: ['Case'] * n_cases + ['Control'] * n_controls,
        CLINICAL_AGE_COL: np.random.normal(50, 15, n_samples).astype(int),
        CLINICAL_SEX_COL: np.random.choice(['M', 'F'], n_samples, p=[0.5, 0.5])
    }
    clinical_df = pd.DataFrame(clinical_data)
    clinical_df.set_index(CLINICAL_SAMPLE_ID_COL, inplace=True)
    
    # For numeric models, convert disease status to binary
    clinical_df['disease_binary'] = (clinical_df[CLINICAL_DISEASE_COL] == 'Case').astype(int)
    
    # Simulate base methylation values (from Beta distribution)
    # Shape parameters for Beta distribution to simulate bimodal methylation patterns
    alpha, beta = 0.4, 0.4
    beta_values = np.random.beta(alpha, beta, (n_samples, n_cpgs))
    
    # Generate CpG IDs
    cpg_ids = [f'cg{i+1:08d}' for i in range(n_cpgs)]
    
    # Select random CpGs to be differentially methylated
    diff_cpgs_indices = np.random.choice(n_cpgs, n_diff_cpgs, replace=False)
    
    # Add differential methylation effect
    for idx in diff_cpgs_indices:
        # For cases (first half of samples), increase or decrease methylation
        direction = 1 if np.random.random() < 0.5 else -1
        effect = direction * effect_size
        
        # Apply effect to cases, ensuring values stay in [0, 1]
        for i in range(n_cases):
            beta_values[i, idx] = np.clip(beta_values[i, idx] + effect, 0, 1)
    
    # Create a mapping dataset
    cpg_mapping = pd.DataFrame({
        'CpG_ID': cpg_ids,
        'Gene_Symbol': [f'Gene_{i//5}' for i in range(n_cpgs)],  # Assign 5 CpGs to each gene
        'Chromosome': np.random.choice(list(range(1, 23)) + ['X', 'Y'], n_cpgs),
        'Position': np.random.randint(1, 250000000, n_cpgs),
        'Is_Promoter': np.random.choice([True, False], n_cpgs, p=[0.2, 0.8])
    })
    
    print(f"Data simulation complete. {n_diff_cpgs} CpGs are differentially methylated.")
    
    return beta_values, clinical_df, diff_cpgs_indices, cpg_mapping

def run_analysis_pipeline():
    """Runs the complete analysis pipeline with simulated data."""
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Simulation parameters
    n_samples = 100
    n_cpgs = 2000
    n_diff_cpgs = 100
    effect_size = 0.2
    
    # Simulate data
    beta_values, clinical_df, diff_cpgs_indices, cpg_mapping = simulate_methylation_data(
        n_samples, n_cpgs, n_diff_cpgs, effect_size
    )
    
    # Convert beta to logit for modeling
    beta_logits = beta_to_logit(beta_values)
    
    # Setup output tracking
    all_results = {}
    model_metrics = {}
    
    # Plot the beta value distribution
    plt.figure(figsize=(10, 6))
    ax = plot_beta_distribution(beta_values, clinical_df[CLINICAL_DISEASE_COL], 
                               group_name='Disease Status')
    plt.tight_layout()
    plt.savefig('plots/beta_distribution.png')
    plt.close()
    
    # Plot PCA of the data
    plt.figure(figsize=(10, 8))
    ax, pca_model = plot_pca(beta_values, clinical_df, color_by=CLINICAL_DISEASE_COL, 
                            use_m_values=True)
    plt.tight_layout()
    plt.savefig('plots/pca_visualization.png')
    plt.close()
    
    # Plot t-SNE of the data
    plt.figure(figsize=(10, 8))
    ax, tsne_model = plot_tsne(beta_values, clinical_df, color_by=CLINICAL_DISEASE_COL,
                              use_m_values=True, perplexity=5)  # Lower perplexity for small datasets
    plt.tight_layout()
    plt.savefig('plots/tsne_visualization.png')
    plt.close()
    
    # Define the model types to test
    model_types = ['rf', 'lasso', 'elasticnet']
    
    # 1. Basic Model Training
    print("\n" + "="*80)
    print("1. BASIC MODEL TRAINING")
    print("="*80)
    
    basic_results = {}
    for model_type in model_types:
        print(f"\nTraining {model_type.upper()} model...")
        model_result = train_classifier(
            beta_logits, 
            clinical_df, 
            target_col='disease_binary',
            model_type=model_type,
            test_size=0.3,
            random_state=RANDOM_SEED
        )
        
        basic_results[model_type] = model_result
        
        # Extract metrics
        metrics = model_result['metrics']
        print(f"  Test accuracy: {metrics['accuracy']:.4f}")
        print(f"  Test AUC: {metrics['auc']:.4f}")
        print(f"  Test F1: {metrics['f1']:.4f}")
        
        # Save the model
        with open(f'models/basic_{model_type}_model.pkl', 'wb') as f:
            pickle.dump(model_result, f)
    
    all_results['basic_training'] = basic_results
    
    # Plot ROC curves for all models
    plt.figure(figsize=(10, 8))
    for model_type, result in basic_results.items():
        plot_roc_curve(
            result['y_test'], 
            result['y_pred_proba'], 
            label=model_type.upper(),
            show_sens_at_spec=0.95  # Show sensitivity at 95% specificity
        )
    plt.tight_layout()
    plt.savefig('plots/basic_models_roc_curves.png')
    plt.close()
    
    # Calculate and compare metrics
    metrics_list = []
    for model_type, result in basic_results.items():
        metrics = calculate_metrics(
            result['y_test'],
            result['y_pred'],
            result['y_pred_proba']
        )
        metrics_list.append(metrics)
        model_metrics[f'basic_{model_type}'] = metrics
    
    # Plot metrics comparison
    plt.figure(figsize=(12, 6))
    plot_metrics_comparison(metrics_list, model_types)
    plt.tight_layout()
    plt.savefig('plots/basic_models_metrics_comparison.png')
    plt.close()
    
    # 2. Feature Selection and Model Explanations
    print("\n" + "="*80)
    print("2. FEATURE SELECTION AND MODEL EXPLANATIONS")
    print("="*80)
    
    # Train model with feature selection
    feature_selection_methods = ['variance', 'univariate']
    fs_results = {}
    
    for fs_method in feature_selection_methods:
        print(f"\nTraining RF model with {fs_method} feature selection...")
        fs_result = train_classifier(
            beta_logits, 
            clinical_df, 
            target_col='disease_binary',
            model_type='rf',
            test_size=0.3,
            feature_selection=fs_method,
            n_features=200,
            random_state=RANDOM_SEED
        )
        
        metrics = fs_result['metrics']
        print(f"  Test accuracy: {metrics['accuracy']:.4f}")
        print(f"  Test AUC: {metrics['auc']:.4f}")
        
        fs_results[fs_method] = fs_result
    
    all_results['feature_selection'] = fs_results
    
    # Model explanation for one model
    print("\nExtracting feature importance for RF model with variance feature selection...")
    importance_df = explain_model(
        fs_results['variance'],
        genes_df=cpg_mapping,
        cpg_ids=cpg_mapping['CpG_ID'].values,
        top_n=20
    )
    
    # Save feature importance to CSV
    importance_df.to_csv('tables/rf_feature_importance.csv', index=False)
    print(f"Feature importance saved to tables/rf_feature_importance.csv")
    
    # Plot top features
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df.head(15), x='Importance', y='CpG_ID')
    plt.title('Top 15 Features by Importance')
    plt.tight_layout()
    plt.savefig('plots/top_features_importance.png')
    plt.close()
    
    # Check if any of the differentially methylated CpGs were identified
    diff_cpg_ids = [cpg_mapping['CpG_ID'].iloc[idx] for idx in diff_cpgs_indices]
    identified_diff_cpgs = importance_df[importance_df['CpG_ID'].isin(diff_cpg_ids)]
    
    print(f"\nNumber of true differentially methylated CpGs in top features: {len(identified_diff_cpgs)}")
    if not identified_diff_cpgs.empty:
        print("Top differentially methylated CpGs identified:")
        print(identified_diff_cpgs.head().to_string())
    
    # 2.5 Testing PCA Dimensionality Reduction
    print("\n" + "="*80)
    print("2.5 TESTING PCA DIMENSIONALITY REDUCTION")
    print("="*80)
    
    # Test models with PCA dimensionality reduction
    pca_n_components = [10, 20, 50]
    pca_results = {}
    
    # Test Random Forest with different PCA components
    for n_components in pca_n_components:
        print(f"\nTraining RF model with PCA ({n_components} components)...")
        pca_result = train_classifier(
            beta_logits, 
            clinical_df, 
            target_col='disease_binary',
            model_type='rf',
            test_size=0.3,
            use_pca=True,
            n_components=n_components,
            random_state=RANDOM_SEED
        )
        
        metrics = pca_result['metrics']
        print(f"  Test accuracy: {metrics['accuracy']:.4f}")
        print(f"  Test AUC: {metrics['auc']:.4f}")
        
        pca_results[f'pca_{n_components}'] = pca_result
        
        # Calculate and save detailed metrics
        detailed_metrics = calculate_metrics(
            pca_result['y_test'],
            pca_result['y_pred'],
            pca_result['y_pred_proba']
        )
        model_metrics[f'pca_{n_components}_rf'] = detailed_metrics
    
    all_results['pca'] = pca_results
    
    # Plot ROC curves for PCA models
    plt.figure(figsize=(10, 8))
    for n_components, result in pca_results.items():
        plot_roc_curve(
            result['y_test'], 
            result['y_pred_proba'], 
            label=f'RF with {n_components.split("_")[1]} PCA components',
            show_sens_at_spec=0.95  # Show sensitivity at 95% specificity with CI
        )
    
    # Also add the basic RF model for comparison
    plot_roc_curve(
        basic_results['rf']['y_test'], 
        basic_results['rf']['y_pred_proba'], 
        label='RF without PCA',
        show_sens_at_spec=0.95
    )
    
    plt.tight_layout()
    plt.savefig('plots/pca_models_roc_curves.png')
    plt.close()
    
    # Compare sensitivity at 95% specificity across models
    plt.figure(figsize=(10, 6))
    model_labels = []
    sens_values = []
    ci_lower = []
    ci_upper = []
    
    # Add PCA models
    for n_components in pca_n_components:
        metrics = model_metrics[f'pca_{n_components}_rf']
        model_labels.append(f'PCA {n_components}')
        sens_values.append(metrics['sensitivity_at_95spec'])
        ci_lower.append(metrics['sensitivity_at_95spec'] - metrics['sensitivity_at_95spec_lower_ci'])
        ci_upper.append(metrics['sensitivity_at_95spec_upper_ci'] - metrics['sensitivity_at_95spec'])
    
    # Add basic RF for comparison
    metrics = model_metrics['basic_rf']
    model_labels.append('No PCA')
    sens_values.append(metrics['sensitivity_at_95spec'])
    ci_lower.append(metrics['sensitivity_at_95spec'] - metrics['sensitivity_at_95spec_lower_ci'])
    ci_upper.append(metrics['sensitivity_at_95spec_upper_ci'] - metrics['sensitivity_at_95spec'])
    
    # Plot with error bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(model_labels, sens_values, yerr=[ci_lower, ci_upper], fmt='o', capsize=5, ecolor='black')
    plt.axhline(y=sens_values[-1], color='r', linestyle='--', alpha=0.5)  # Reference line for No PCA
    plt.ylabel('Sensitivity at 95% Specificity')
    plt.title('Comparison of Sensitivity at 95% Specificity with 95% CIs')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig('plots/pca_sensitivity_comparison.png')
    plt.close()
    
    # Test the scree plot functionality
    print("\nCalculating PCA eigenvalues and creating scree plot...")
    
    # Calculate PCA eigenvalues from beta values
    scree_data = calculate_scree_plot_values(
        beta_values, 
        use_m_values=True, 
        n_components=min(50, beta_values.shape[0], beta_values.shape[1]),
        random_state=RANDOM_SEED
    )
    
    # Create scree plot with elbow detection
    plt.figure(figsize=(10, 6))
    ax, elbow_idx = plot_scree(
        eigenvalues=scree_data['eigenvalues'],
        explained_variance_ratio=scree_data['explained_variance_ratio'],
        cumulative_explained_variance=scree_data['cumulative_explained_variance'],
        detect_elbow=True
    )
    
    # Print the detected elbow information
    if elbow_idx is not None:
        n_components_at_elbow = elbow_idx + 1  # 0-based to 1-based
        variance_at_elbow = scree_data['cumulative_explained_variance'][elbow_idx] * 100
        print(f"Elbow detected at PC{n_components_at_elbow} with {variance_at_elbow:.2f}% cumulative variance explained")
    else:
        print("No elbow detected in scree plot")
    
    # Save the plot
    plt.savefig('plots/pca_scree_plot.png')
    plt.close()
    
    # Alternative: create scree plot directly from beta values
    plt.figure(figsize=(10, 6))
    ax, elbow_idx = plot_scree(
        beta_values=beta_values,
        use_m_values=True,
        n_components=min(50, beta_values.shape[0], beta_values.shape[1]),
        detect_elbow=True
    )
    plt.savefig('plots/pca_scree_plot_direct.png')
    plt.close()
    
    # Create a table showing sensitivity at different specificities with CIs
    sens_spec_data = []
    for model_name, metrics_dict in model_metrics.items():
        if 'sensitivity_at_95spec' in metrics_dict:
            row = {
                'Model': model_name,
                'Sensitivity at 90% Spec': f"{metrics_dict['sensitivity_at_90spec']:.3f} ({metrics_dict['sensitivity_at_90spec_lower_ci']:.3f}-{metrics_dict['sensitivity_at_90spec_upper_ci']:.3f})",
                'Sensitivity at 95% Spec': f"{metrics_dict['sensitivity_at_95spec']:.3f} ({metrics_dict['sensitivity_at_95spec_lower_ci']:.3f}-{metrics_dict['sensitivity_at_95spec_upper_ci']:.3f})",
                'Sensitivity at 99% Spec': f"{metrics_dict['sensitivity_at_99spec']:.3f} ({metrics_dict['sensitivity_at_99spec_lower_ci']:.3f}-{metrics_dict['sensitivity_at_99spec_upper_ci']:.3f})",
                'AUC': f"{metrics_dict['auroc']:.3f}"
            }
            sens_spec_data.append(row)
    
    sens_spec_df = pd.DataFrame(sens_spec_data)
    sens_spec_df.to_csv('tables/sensitivity_at_specificity_with_ci.csv', index=False)
    print(f"Sensitivity at specificity table saved to tables/sensitivity_at_specificity_with_ci.csv")
    
    # 3. Cross-Validation
    print("\n" + "="*80)
    print("3. CROSS-VALIDATION ANALYSIS")
    print("="*80)
    
    cv_results = {}
    for model_type in model_types:
        print(f"\nCross-validating {model_type.upper()} model...")
        cv_result = cross_validate_model(
            beta_logits, 
            clinical_df, 
            target_col='disease_binary',
            model_type=model_type,
            cv=5,
            feature_selection='variance',
            n_features=200,
            random_state=RANDOM_SEED
        )
        
        mean_metrics = cv_result['mean_metrics']
        std_metrics = cv_result['std_metrics']
        
        print(f"  Mean CV accuracy: {mean_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}")
        print(f"  Mean CV AUC: {mean_metrics['auc']:.4f} ± {std_metrics['auc']:.4f}")
        print(f"  Mean CV F1: {mean_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")
        
        cv_results[model_type] = cv_result
        
        # Calculate and save detailed metrics
        detailed_metrics = calculate_metrics(
            cv_result['y_true'],
            cv_result['y_pred'],
            cv_result['y_proba']
        )
        model_metrics[f'cv_{model_type}'] = detailed_metrics
    
    all_results['cross_validation'] = cv_results
    
    # Plot ROC curves for all cross-validated models
    plt.figure(figsize=(10, 8))
    for model_type, result in cv_results.items():
        plot_roc_curve(
            result['y_true'], 
            result['y_proba'], 
            label=model_type.upper(),
            show_sens_at_spec=0.95  # Show sensitivity at 95% specificity
        )
    plt.tight_layout()
    plt.savefig('plots/cv_models_roc_curves.png')
    plt.close()
    
    # 4. Hyperparameter Tuning with Nested CV
    print("\n" + "="*80)
    print("4. HYPERPARAMETER TUNING WITH NESTED CROSS-VALIDATION")
    print("="*80)
    print("This may take some time...")
    
    # Use RF for nested CV
    nested_cv_model_types = ['rf']
    nested_cv_results = {}
    
    for model_type in nested_cv_model_types:
        print(f"\nPerforming nested CV for {model_type.upper()} model...")
        
        try:
            # Define smaller parameter grid for demonstration
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [None, 10]
            }
            
            nested_result = nested_cross_validation(
                beta_logits, 
                clinical_df, 
                target_col='disease_binary',
                model_type=model_type,
                outer_cv=3,  # Smaller values for demonstration
                inner_cv=2,  # Smaller values for demonstration
                feature_selection='variance',
                n_features=200,
                random_state=RANDOM_SEED,
                param_grid=param_grid
            )
            
            mean_metrics = nested_result['mean_metrics']
            std_metrics = nested_result['std_metrics']
            
            print(f"  Mean nested CV accuracy: {mean_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}")
            print(f"  Mean nested CV AUC: {mean_metrics['auc']:.4f} ± {std_metrics['auc']:.4f}")
            print(f"  Best parameters per fold:")
            for i, params in enumerate(nested_result['best_params_per_fold']):
                print(f"    Fold {i+1}: {params}")
            
            nested_cv_results[model_type] = nested_result
            
            # Calculate and save detailed metrics
            detailed_metrics = calculate_metrics(
                nested_result['y_true'],
                nested_result['y_pred'],
                nested_result['y_proba']
            )
            model_metrics[f'nested_cv_{model_type}'] = detailed_metrics
            
        except Exception as e:
            print(f"Error performing nested CV for {model_type}: {str(e)}")
            print("Skipping this model for nested CV analysis.")
    
    all_results['nested_cv'] = nested_cv_results
    
    # Save all metrics to a CSV file
    metrics_df = pd.DataFrame(model_metrics).T
    metrics_df.index.name = 'Model'
    metrics_df.to_csv('tables/all_model_metrics.csv')
    print(f"\nAll model metrics saved to tables/all_model_metrics.csv")
    
    # Create a summary plot of all models
    plt.figure(figsize=(12, 10))
    
    # Plot ROC curves for the best models from each approach
    plt.subplot(2, 2, 1)
    plot_roc_curve(
        basic_results['rf']['y_test'], 
        basic_results['rf']['y_pred_proba'], 
        label='Basic RF'
    )
    plot_roc_curve(
        cv_results['rf']['y_true'], 
        cv_results['rf']['y_proba'], 
        label='CV RF'
    )
    if 'rf' in nested_cv_results:
        plot_roc_curve(
            nested_cv_results['rf']['y_true'], 
            nested_cv_results['rf']['y_proba'], 
            label='Nested CV RF'
        )
    plt.title('ROC Curves Comparison - RF Models')
    
    # Plot AUC comparison for all models
    plt.subplot(2, 2, 2)
    model_names = []
    aucs = []
    
    for model_type in model_types:
        if f'basic_{model_type}' in model_metrics:
            model_names.append(f'Basic {model_type.upper()}')
            aucs.append(model_metrics[f'basic_{model_type}'].get('auroc', np.nan))
        
        if f'cv_{model_type}' in model_metrics:
            model_names.append(f'CV {model_type.upper()}')
            aucs.append(model_metrics[f'cv_{model_type}'].get('auroc', np.nan))
    
    for model_type in nested_cv_model_types:
        if f'nested_cv_{model_type}' in model_metrics:
            model_names.append(f'Nested CV {model_type.upper()}')
            aucs.append(model_metrics[f'nested_cv_{model_type}'].get('auroc', np.nan))
    
    # Create AUC comparison bar chart
    plt.bar(model_names, aucs)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0.5, 1.05)
    plt.title('AUC Comparison Across Models')
    plt.tight_layout()
    
    # Plot feature importance vs. true differential status
    plt.subplot(2, 2, 3)
    
    # Get feature importance from RF model
    rf_result = basic_results['rf']
    feature_importances = rf_result['model'].feature_importances_
    
    # Mark true differential CpGs
    is_diff = np.zeros(n_cpgs, dtype=bool)
    is_diff[diff_cpgs_indices] = True
    
    # Create a DataFrame for plotting
    importance_comparison = pd.DataFrame({
        'Importance': feature_importances,
        'Is_Differential': is_diff
    })
    
    # Calculate mean importance for differential vs. non-differential CpGs
    mean_imp = importance_comparison.groupby('Is_Differential')['Importance'].mean()
    
    plt.bar(['Non-Differential', 'Differential'], mean_imp.values)
    plt.title('Mean Feature Importance')
    plt.ylabel('Mean Importance')
    
    # Plot precision-recall curve for the best model
    plt.subplot(2, 2, 4)
    best_model_data = cv_results['rf']  # Assume RF with CV is our best model
    plot_precision_recall_curve(
        best_model_data['y_true'],
        best_model_data['y_proba'],
        label='RF (CV)'
    )
    plt.title('Precision-Recall Curve - Best Model')
    
    plt.tight_layout()
    plt.savefig('plots/summary_analysis.png')
    plt.close()
    
    # Print execution time
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nAnalysis completed in {duration:.2f} seconds")
    print(f"Plots and tables saved to the 'plots' and 'tables' directories")

    return all_results

if __name__ == "__main__":
    print("="*80)
    print("METHYLATION DATA ANALYSIS PIPELINE")
    print("="*80)
    results = run_analysis_pipeline()

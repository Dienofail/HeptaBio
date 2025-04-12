import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# Import utility functions from common_utils
try:
    from common_utils import beta_to_m, logit_to_beta
except ImportError:
    print("Warning: common_utils not found. Using local definitions of conversion functions.")
    # Define local versions if common_utils is not available
    def beta_to_m(beta, epsilon=1e-6):
        beta_clipped = np.clip(beta, epsilon, 1 - epsilon)
        return np.log2(beta_clipped / (1 - beta_clipped))
    
    def logit_to_beta(logits):
        odds = np.exp(logits)
        return odds / (1 + odds)

def train_classifier(beta_logits, 
                     clinical_df, 
                     target_col="disease_status", 
                     model_type="rf", 
                     test_size=0.2,
                     feature_selection=None,
                     n_features=1000,
                     use_pca=False,
                     n_components=50,
                     random_state=42,
                     model_params=None):
    """
    Trains a classifier to predict a target column from methylation data.
    
    Args:
        beta_logits (np.ndarray): Methylation data in logit form (samples x CpGs).
        clinical_df (pd.DataFrame): Clinical data with target column.
        target_col (str): Column name in clinical_df to predict. Default is "disease_status".
        model_type (str): Type of model to train - "rf" (Random Forest), "lasso" (Logistic Regression with L1), 
                          or "elasticnet" (Logistic Regression with ElasticNet). Default is "rf".
        test_size (float): Proportion of data to use for testing. Default is 0.2.
        feature_selection (str or None): Method for feature selection - "variance" (most variable CpGs) or 
                                        "univariate" (top CpGs by univariate p-value) or None (no selection). Default is None.
        n_features (int): Number of features to select if feature_selection is not None. Default is 1000.
        use_pca (bool): Whether to use PCA for dimensionality reduction. Default is False.
        n_components (int): Number of PCA components to keep if use_pca is True. Default is 50.
        random_state (int): Random seed for reproducibility. Default is 42.
        model_params (dict or None): Parameters to pass to the model constructor. Default is None (use defaults).
    
    Returns:
        dict: Dictionary containing the trained model, feature transformer, and other relevant information.
    """
    # Convert logits to beta values if necessary and then to M-values
    if np.min(beta_logits) < 0 or np.max(beta_logits) > 1:
        # Assume these are already logits, convert to beta first
        beta_values = logit_to_beta(beta_logits)
    else:
        # Assume these are already beta values
        beta_values = beta_logits
    
    # Convert to M-values for better analysis
    m_values = beta_to_m(beta_values)
    
    # Get target variable
    y = clinical_df[target_col].values
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        m_values, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Feature selection
    feature_selector = None
    if feature_selection == "variance":
        # Select top n_features by variance
        variances = np.var(X_train, axis=0)
        top_indices = np.argsort(variances)[-n_features:]
        X_train = X_train[:, top_indices]
        X_test = X_test[:, top_indices]
        
        # Create a mask of selected features for later reference
        feature_mask = np.zeros(m_values.shape[1], dtype=bool)
        feature_mask[top_indices] = True
        feature_selector = {"type": "variance", "mask": feature_mask, "indices": top_indices}
        
    elif feature_selection == "univariate":
        # Select top n_features by univariate F-test
        selector = SelectKBest(f_classif, k=n_features)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)
        
        # Save the mask and selected indices
        feature_mask = selector.get_support()
        top_indices = np.where(feature_mask)[0]
        feature_selector = {"type": "univariate", "mask": feature_mask, "indices": top_indices, "selector": selector}
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Apply PCA if requested
    pca_transformer = None
    if use_pca:
        pca_transformer = PCA(n_components=n_components, random_state=random_state)
        X_train = pca_transformer.fit_transform(X_train)
        X_test = pca_transformer.transform(X_test)
    
    # Initialize and train model
    model = None
    if model_type == "rf":
        # Random Forest Classifier
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': random_state
        }
        if model_params:
            params.update(model_params)
        model = RandomForestClassifier(**params)
        
    elif model_type == "lasso":
        # Logistic Regression with L1 penalty
        params = {
            'penalty': 'l1',
            'C': 1.0,
            'solver': 'liblinear',
            'random_state': random_state
        }
        if model_params:
            params.update(model_params)
        model = LogisticRegression(**params)
        
    elif model_type == "elasticnet":
        # Logistic Regression with ElasticNet penalty
        params = {
            'penalty': 'elasticnet',
            'C': 1.0,
            'solver': 'saga',
            'l1_ratio': 0.5,
            'random_state': random_state
        }
        if model_params:
            params.update(model_params)
        model = LogisticRegression(**params)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else None,
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    
    return {
        'model': model,
        'model_type': model_type,
        'feature_selector': feature_selector,
        'scaler': scaler,
        'pca_transformer': pca_transformer,
        'metrics': metrics,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


def cross_validate_model(beta_logits, 
                         clinical_df, 
                         target_col="disease_status", 
                         model_type="rf",
                         cv=5,
                         feature_selection=None,
                         n_features=1000,
                         use_pca=False,
                         n_components=50,
                         random_state=42,
                         model_params=None):
    """
    Performs cross-validation on a classifier and returns performance metrics.
    
    Args:
        beta_logits (np.ndarray): Methylation data in logit form (samples x CpGs).
        clinical_df (pd.DataFrame): Clinical data with target column.
        target_col (str): Column name in clinical_df to predict. Default is "disease_status".
        model_type (str): Type of model to train - "rf", "lasso", "elasticnet". Default is "rf".
        cv (int): Number of cross-validation folds. Default is 5.
        feature_selection (str or None): Method for feature selection. Default is None.
        n_features (int): Number of features to select. Default is 1000.
        use_pca (bool): Whether to use PCA. Default is False.
        n_components (int): Number of PCA components. Default is 50.
        random_state (int): Random seed. Default is 42.
        model_params (dict or None): Parameters for the model. Default is None.
    
    Returns:
        dict: Dictionary with cross-validation metrics and feature importance information.
    """
    # Convert logits to beta values if necessary and then to M-values
    if np.min(beta_logits) < 0 or np.max(beta_logits) > 1:
        # Assume these are already logits, convert to beta first
        beta_values = logit_to_beta(beta_logits)
    else:
        # Assume these are already beta values
        beta_values = beta_logits
    
    # Convert to M-values for better analysis
    m_values = beta_to_m(beta_values)
    
    # Get target variable
    y = clinical_df[target_col].values
    
    # Initialize cross-validation
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Metrics to collect
    metrics = {
        'accuracy': [],
        'auc': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    # Feature importance across folds
    feature_importances = np.zeros(m_values.shape[1])
    selected_features_counts = np.zeros(m_values.shape[1])
    
    # Predictions for all samples
    y_pred_all = np.zeros_like(y, dtype=float)
    y_proba_all = np.zeros_like(y, dtype=float)
    
    # Cross-validate
    for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(m_values, y)):
        print(f"Processing fold {fold+1}/{cv}")
        X_train, X_test = m_values[train_idx], m_values[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Feature selection
        if feature_selection == "variance":
            # Select top n_features by variance
            variances = np.var(X_train, axis=0)
            top_indices = np.argsort(variances)[-n_features:]
            X_train = X_train[:, top_indices]
            X_test = X_test[:, top_indices]
            
            # Update feature counts
            selected_features_counts[top_indices] += 1
            
        elif feature_selection == "univariate":
            # Select top n_features by univariate F-test
            selector = SelectKBest(f_classif, k=n_features)
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)
            
            # Update feature counts
            selected_mask = selector.get_support()
            selected_features_counts[selected_mask] += 1
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Apply PCA if requested
        if use_pca:
            pca_transformer = PCA(n_components=n_components, random_state=random_state)
            X_train = pca_transformer.fit_transform(X_train)
            X_test = pca_transformer.transform(X_test)
        
        # Initialize and train model
        model = None
        if model_type == "rf":
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': random_state
            }
            if model_params:
                params.update(model_params)
            model = RandomForestClassifier(**params)
            
        elif model_type == "lasso":
            params = {
                'penalty': 'l1',
                'C': 1.0,
                'solver': 'liblinear',
                'random_state': random_state
            }
            if model_params:
                params.update(model_params)
            model = LogisticRegression(**params)
            
        elif model_type == "elasticnet":
            params = {
                'penalty': 'elasticnet',
                'C': 1.0,
                'solver': 'saga',
                'l1_ratio': 0.5,
                'random_state': random_state
            }
            if model_params:
                params.update(model_params)
            model = LogisticRegression(**params)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Fit the model
        model.fit(X_train, y_train)
        
        # Store feature importances
        if model_type == "rf":
            # For tree-based models, use feature_importances_
            if feature_selection:
                # If feature selection was used, we need to map back to original indices
                if feature_selection == "variance":
                    for i, idx in enumerate(top_indices):
                        feature_importances[idx] += model.feature_importances_[i]
                elif feature_selection == "univariate":
                    original_indices = np.where(selected_mask)[0]
                    for i, idx in enumerate(original_indices):
                        feature_importances[idx] += model.feature_importances_[i]
            else:
                feature_importances += model.feature_importances_
                
        elif model_type in ["lasso", "elasticnet"]:
            # For linear models, use coefficients
            if feature_selection:
                if feature_selection == "variance":
                    for i, idx in enumerate(top_indices):
                        feature_importances[idx] += abs(model.coef_[0][i])
                elif feature_selection == "univariate":
                    original_indices = np.where(selected_mask)[0]
                    for i, idx in enumerate(original_indices):
                        feature_importances[idx] += abs(model.coef_[0][i])
            else:
                feature_importances += abs(model.coef_[0])
        
        # Evaluate on test set
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
        y_pred = model.predict(X_test)
        
        # Store predictions
        y_pred_all[test_idx] = y_pred
        y_proba_all[test_idx] = y_pred_proba
        
        # Calculate metrics
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        if len(np.unique(y_test)) > 1:
            metrics['auc'].append(roc_auc_score(y_test, y_pred_proba))
        metrics['precision'].append(precision_score(y_test, y_pred, average='weighted'))
        metrics['recall'].append(recall_score(y_test, y_pred, average='weighted'))
        metrics['f1'].append(f1_score(y_test, y_pred, average='weighted'))
    
    # Average metrics across folds
    mean_metrics = {k: np.mean(v) for k, v in metrics.items() if v}
    std_metrics = {k: np.std(v) for k, v in metrics.items() if v}
    
    # Normalize feature importances by number of folds
    feature_importances /= cv
    
    # Calculate frequency of selection for each feature
    feature_selection_frequency = selected_features_counts / cv if feature_selection else None
    
    # Get indices of top features by importance
    top_features_idx = np.argsort(feature_importances)[::-1]
    
    return {
        'mean_metrics': mean_metrics,
        'std_metrics': std_metrics,
        'feature_importances': feature_importances,
        'top_features_idx': top_features_idx,
        'feature_selection_frequency': feature_selection_frequency,
        'y_true': y,
        'y_pred': y_pred_all,
        'y_proba': y_proba_all
    }


def explain_model(model_result, genes_df=None, cpg_ids=None, top_n=20):
    """
    Extracts feature importance from a trained model and creates an interpretable summary.
    
    Args:
        model_result (dict): Result dictionary from train_classifier.
        genes_df (pd.DataFrame or None): DataFrame mapping CpG indices to genes. Default is None.
        cpg_ids (list or None): List of CpG identifiers. Default is None.
        top_n (int): Number of top features to return. Default is 20.
    
    Returns:
        pd.DataFrame: DataFrame of top features with importance scores and interpretations.
    """
    model = model_result['model']
    model_type = model_result['model_type']
    feature_selector = model_result.get('feature_selector', None)
    
    # Extract feature importances or coefficients
    importances = None
    if model_type in ["rf"]:
        # For tree-based models
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            raise ValueError(f"Model of type {model_type} does not have feature_importances_ attribute")
    
    elif model_type in ["lasso", "elasticnet"]:
        # For linear models
        if hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])  # Take absolute values for importance
        else:
            raise ValueError(f"Model of type {model_type} does not have coef_ attribute")
    
    # If no importances found
    if importances is None:
        raise ValueError("Could not extract feature importances from the model")
    
    # Map back to original feature indices if feature selection was used
    original_importances = None
    if feature_selector:
        sel_type = feature_selector.get('type')
        if sel_type in ["variance", "univariate"]:
            mask = feature_selector.get('mask')
            indices = feature_selector.get('indices')
            
            # Initialize array of zeros for all original features
            original_importances = np.zeros(len(mask))
            
            # Fill in the values for selected features
            for i, idx in enumerate(indices):
                original_importances[idx] = importances[i]
        else:
            # If feature selection method is not recognized
            original_importances = importances
    else:
        # If no feature selection was used
        original_importances = importances
    
    # Sort features by importance
    sorted_idx = np.argsort(original_importances)[::-1]
    
    # Create a DataFrame with the results
    importance_df = pd.DataFrame({
        'Feature_Index': sorted_idx,
        'Importance': original_importances[sorted_idx]
    })
    
    # Limit to top_n features
    importance_df = importance_df.head(top_n)
    
    # Add CpG IDs if available
    if cpg_ids is not None:
        importance_df['CpG_ID'] = [cpg_ids[idx] for idx in importance_df['Feature_Index']]
    
    # Add gene information if available
    if genes_df is not None and cpg_ids is not None:
        # Assuming genes_df has columns 'CpG_ID' and 'Gene_Symbol'
        cpg_col = 'CpG_ID'  # Adjust based on actual column name
        gene_col = 'Gene_Symbol'  # Adjust based on actual column name
        
        # Create a mapping of CpG ID to gene symbol
        cpg_to_gene = dict(zip(genes_df[cpg_col], genes_df[gene_col]))
        
        # Add the gene information
        importance_df['Gene'] = importance_df['CpG_ID'].map(cpg_to_gene)
    
    # Add direction of effect for linear models
    if model_type in ["lasso", "elasticnet"] and hasattr(model, 'coef_'):
        # Map feature indices to their coefficients' signs
        coef_signs = {idx: np.sign(model.coef_[0][i]) for i, idx in enumerate(feature_selector.get('indices'))} if feature_selector else {i: np.sign(coef) for i, coef in enumerate(model.coef_[0])}
        
        # Add the direction to the DataFrame
        importance_df['Direction'] = importance_df['Feature_Index'].map(lambda idx: "Positive (higher in cases)" if coef_signs.get(idx, 0) > 0 else "Negative (lower in cases)")
    
    return importance_df


def nested_cross_validation(beta_logits, 
                           clinical_df, 
                           target_col="disease_status", 
                           model_type="rf",
                           outer_cv=5,
                           inner_cv=3,
                           feature_selection=None,
                           n_features=1000,
                           use_pca=False,
                           n_components=50,
                           random_state=42,
                           param_grid=None):
    """
    Performs nested cross-validation with hyperparameter tuning.
    
    Args:
        beta_logits (np.ndarray): Methylation data in logit form (samples x CpGs).
        clinical_df (pd.DataFrame): Clinical data with target column.
        target_col (str): Column name in clinical_df to predict. Default is "disease_status".
        model_type (str): Type of model to train - "rf", "lasso", "elasticnet". Default is "rf".
        outer_cv (int): Number of outer cross-validation folds. Default is 5.
        inner_cv (int): Number of inner cross-validation folds for hyperparameter tuning. Default is 3.
        feature_selection (str or None): Method for feature selection. Default is None.
        n_features (int): Number of features to select. Default is 1000.
        use_pca (bool): Whether to use PCA. Default is False.
        n_components (int): Number of PCA components. Default is 50.
        random_state (int): Random seed. Default is 42.
        param_grid (dict or None): Parameter grid for hyperparameter tuning. Default is None (use default grid).
    
    Returns:
        dict: Dictionary with nested cross-validation results, best parameters, and metrics.
    """
    # Convert logits to beta values if necessary and then to M-values
    if np.min(beta_logits) < 0 or np.max(beta_logits) > 1:
        # Assume these are already logits, convert to beta first
        beta_values = logit_to_beta(beta_logits)
    else:
        # Assume these are already beta values
        beta_values = beta_logits
    
    # Convert to M-values for better analysis
    m_values = beta_to_m(beta_values)
    
    # Get target variable
    y = clinical_df[target_col].values
    
    # Initialize outer cross-validation
    outer_cv_splitter = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=random_state)
    
    # Default parameter grids for each model type
    default_param_grids = {
        "rf": {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        },
        "lasso": {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0]
        },
        "elasticnet": {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.5, 0.7, 0.9]
        }
    }
    
    # Use default parameter grid if none provided
    if param_grid is None:
        param_grid = default_param_grids.get(model_type)
        if param_grid is None:
            raise ValueError(f"No default parameter grid for model type: {model_type}")
    
    # Metrics to collect
    metrics = {
        'accuracy': [],
        'auc': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    # Store best parameters for each fold
    best_params_list = []
    
    # Predictions for all samples
    y_pred_all = np.zeros_like(y, dtype=float)
    y_proba_all = np.zeros_like(y, dtype=float)
    
    # Nested cross-validation
    for fold, (train_idx, test_idx) in enumerate(outer_cv_splitter.split(m_values, y)):
        print(f"Outer Fold {fold+1}/{outer_cv}")
        X_train_outer, X_test_outer = m_values[train_idx], m_values[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]
        
        # Feature selection on the outer training set
        selected_indices = None
        if feature_selection == "variance":
            # Select top n_features by variance
            variances = np.var(X_train_outer, axis=0)
            selected_indices = np.argsort(variances)[-n_features:]
            X_train_outer = X_train_outer[:, selected_indices]
            X_test_outer = X_test_outer[:, selected_indices]
            
        elif feature_selection == "univariate":
            # Select top n_features by univariate F-test
            selector = SelectKBest(f_classif, k=n_features)
            X_train_outer = selector.fit_transform(X_train_outer, y_train_outer)
            X_test_outer = selector.transform(X_test_outer)
            selected_indices = np.where(selector.get_support())[0]
        
        # Standardize features
        scaler_outer = StandardScaler()
        X_train_outer = scaler_outer.fit_transform(X_train_outer)
        X_test_outer = scaler_outer.transform(X_test_outer)
        
        # Apply PCA if requested
        if use_pca:
            pca_transformer = PCA(n_components=n_components, random_state=random_state)
            X_train_outer = pca_transformer.fit_transform(X_train_outer)
            X_test_outer = pca_transformer.transform(X_test_outer)
        
        # Initialize model based on type
        base_model = None
        if model_type == "rf":
            base_model = RandomForestClassifier(random_state=random_state)
        elif model_type == "lasso":
            base_model = LogisticRegression(penalty='l1', solver='liblinear', random_state=random_state)
        elif model_type == "elasticnet":
            base_model = LogisticRegression(penalty='elasticnet', solver='saga', random_state=random_state)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Inner cross-validation with GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='roc_auc',
            n_jobs=-1,
            return_train_score=True,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X_train_outer, y_train_outer)
        
        # Get the best model and parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_params_list.append(best_params)
        
        # Predict on the outer test set
        y_pred_proba = best_model.predict_proba(X_test_outer)[:, 1] if hasattr(best_model, 'predict_proba') else best_model.predict(X_test_outer)
        y_pred = best_model.predict(X_test_outer)
        
        # Store predictions
        y_pred_all[test_idx] = y_pred
        y_proba_all[test_idx] = y_pred_proba
        
        # Calculate metrics
        metrics['accuracy'].append(accuracy_score(y_test_outer, y_pred))
        if len(np.unique(y_test_outer)) > 1:
            metrics['auc'].append(roc_auc_score(y_test_outer, y_pred_proba))
        metrics['precision'].append(precision_score(y_test_outer, y_pred, average='weighted'))
        metrics['recall'].append(recall_score(y_test_outer, y_pred, average='weighted'))
        metrics['f1'].append(f1_score(y_test_outer, y_pred, average='weighted'))
        
        print(f"Fold {fold+1} Best Parameters: {best_params}")
        print(f"Fold {fold+1} Metrics: AUC={metrics['auc'][-1]:.4f}, Accuracy={metrics['accuracy'][-1]:.4f}")
    
    # Average metrics across folds
    mean_metrics = {k: np.mean(v) for k, v in metrics.items() if v}
    std_metrics = {k: np.std(v) for k, v in metrics.items() if v}
    
    print(f"Mean Metrics Across Folds: AUC={mean_metrics.get('auc', 'N/A')}, Accuracy={mean_metrics.get('accuracy', 'N/A')}")
    
    return {
        'mean_metrics': mean_metrics,
        'std_metrics': std_metrics,
        'best_params_per_fold': best_params_list,
        'y_true': y,
        'y_pred': y_pred_all,
        'y_proba': y_proba_all
    }


def calculate_scree_plot_values(beta_values, use_m_values=True, n_components=None, random_state=42):
    """
    Calculates the eigenvalues needed for a scree plot from beta-values or M-values.
    
    Args:
        beta_values (np.ndarray): Beta values matrix (samples x CpGs). Assumes values in [0, 1].
        use_m_values (bool): Whether to convert beta values to M-values before PCA. Default is True.
        n_components (int or None): Number of components to calculate. If None, uses min(n_samples, n_features).
        random_state (int): Random seed for reproducibility. Default is 42.
        
    Returns:
        dict: Dictionary containing eigenvalues, explained variance ratios, and the PCA model.
    """
    # Convert to M-values if needed
    data_for_pca = beta_values
    if use_m_values:
        # Convert beta to M-values
        data_for_pca = beta_to_m(beta_values)
        
        # Handle potential infinities
        data_for_pca[np.isinf(data_for_pca)] = np.nan
        col_means = np.nanmean(data_for_pca, axis=0)
        inds = np.where(np.isnan(data_for_pca))
        data_for_pca[inds] = np.take(col_means, inds[1])
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_for_pca)
    
    # Run PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_result = pca.fit_transform(scaled_data)
    
    # Calculate cumulative explained variance
    cum_var_explained = np.cumsum(pca.explained_variance_ratio_)
    
    # Detect elbow point - find the point of maximum curvature
    eigenvalues = pca.explained_variance_
    
    return {
        'eigenvalues': eigenvalues,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_explained_variance': cum_var_explained,
        'pca_model': pca
    }


# Example usage
if __name__ == "__main__":
    print("Machine learning functions for cfDNA methylation analysis.")
    print("Import this module to use the functions.")

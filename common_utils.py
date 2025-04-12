import numpy as np

# --- Global Variables for Clinical Data Columns ---
# These should be updated based on the actual column names in the clinical data DataFrame
CLINICAL_DISEASE_COL = 'disease_status'  # Example: 'Group', 'Diagnosis', etc.
CLINICAL_AGE_COL = 'age'
CLINICAL_SEX_COL = 'sex'
CLINICAL_STAGE_COL = 'fibrosis_stage' # Example
CLINICAL_ALT_COL = 'alt' # Example
CLINICAL_AST_COL = 'ast' # Example
# Add other clinical/demographic columns as needed
CLINICAL_SAMPLE_ID_COL = 'sample_id' # Assuming clinical_df has a sample ID column to match beta/depth rows if they are not aligned by index

# --- Data Conversion Functions ---

def logit_to_beta(logits: np.ndarray) -> np.ndarray:
    """Converts logit-transformed beta values back to beta values (0 to 1 scale)."""
    odds = np.exp(logits)
    beta = odds / (1 + odds)
    return beta

def beta_to_m(beta: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Converts beta values (proportion methylated, 0 to 1) to M-values (log2 ratio of methylated to unmethylated).
    An epsilon is added to avoid log(0) issues.

    Args:
        beta (np.ndarray): Array of beta values.
        epsilon (float): Small value to clip beta values, preventing log(0). Defaults to 1e-6.

    Returns:
        np.ndarray: Array of M-values.
    """
    beta_clipped = np.clip(beta, epsilon, 1 - epsilon)
    m_values = np.log2(beta_clipped / (1 - beta_clipped))
    return m_values

def beta_to_logit(beta: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Converts beta values (proportion methylated, 0 to 1) to logit values.
    An epsilon is added to avoid log(0) issues.

    Args:
        beta (np.ndarray): Array of beta values.
        epsilon (float): Small value to clip beta values, preventing log(0). Defaults to 1e-6.

    Returns:
        np.ndarray: Array of logit values.
    """
    beta_clipped = np.clip(beta, epsilon, 1 - epsilon)
    logit_values = np.log(beta_clipped / (1 - beta_clipped))
    return logit_values

def m_to_beta(m_values: np.ndarray) -> np.ndarray:
    """
    Converts M-values (log2 ratio of methylated to unmethylated) to beta values (proportion methylated, 0 to 1).

    Args:
        m_values (np.ndarray): Array of M-values.

    Returns:
        np.ndarray: Array of beta values.
    """
    beta = 2**m_values / (1 + 2**m_values)
    return beta

# --- Placeholder for CpG-to-Gene Mapping ---
# This might be used later in analysis modules
GENES_CPG_COL = 'CpG_ID' # Column name for CpG identifiers in the genes_df
GENES_GENE_COL = 'Gene_Symbol' # Column name for associated gene names

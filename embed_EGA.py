# import libraries
import os
import sys

import json
from idlelib.pyparse import trans

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import pandas as pd
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro
import subprocess

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import SpectralClustering

# numpy2ri.activate()

# check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")


class vectorizer:
    def __init__(self, model_name="BAAI/bge-m3", device=None):
        self.model = SentenceTransformer(model_name, device=device)
        print(f"Model loaded on device: {self.model.device}")

    def get_embedding(self, text, batch_size=32, show_progress=True):
        embeddings = self.model.encode(
            text,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization
        )
        return embeddings

    def process_json(self, file_path, batch_size=32):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        originals = [item['original'] for item in data['items']]
        translations = [item['translation'] for item in data['items']]
        item_numbers = [item['number'] for item in data['items']]

        orig_emb = self.get_embedding(originals, batch_size)
        trans_emb = self.get_embedding(translations, batch_size)

        # orig_df = pd.DataFrame(orig_emb, index=item_numbers, columns=item_numbers)
        # trans_df = pd.DataFrame(trans_emb, index=item_numbers, columns=item_numbers)
        print(f"original embedding: {orig_emb.shape}")
        print(f"translation embedding: {trans_emb.shape}")

        return {
            'original': {'embeddings': orig_emb},
            'translation': {'embeddings': trans_emb},
            'item_numbers': item_numbers
        }


# ===============================================================================
# ============================== TMFG Network ===================================
class TMFG:
    def __init__(self):
        print("Loading R packages for TMFG")
        self.eganet = importr('EGAnet')
        self.base = importr('base')
        self.grdevices = importr('grDevices')
        print("R packages loaded successfully")

    def compute_corr(self, embeddings, method='cosine'):
        if method == 'cosine':
            return embeddings @ embeddings.T
        elif method == 'pearson':
            return np.corrcoef(embeddings)
        else:
            raise ValueError("Method: 'cosine' or 'pearson'")

    def run_tmfg(self, corr_matrix):
        # Save correlation to file, more stable
        np.savetxt('_temp_corr.csv', corr_matrix, delimiter=',')

        # converting R and Python object is unstable
        # with localconverter(ro.default_converter + numpy2ri.converter):
        # ro.globalenv["corr_mat"] = corr_matrix

        r_script = '''library(EGAnet)

cat("=== Loading correlation ===\\n")
corr_mat <- as.matrix(read.csv('_temp_corr.csv', header=FALSE))
cat("Corr dim:", dim(corr_mat), "\\n")
cat("Corr range:", range(corr_mat), "\\n")

cat("\\n=== Running TMFG ===\\n")
result <- tryCatch({
    TMFG(corr_mat)
}, error = function(e) {
    cat("ERROR:", conditionMessage(e), "\\n")
    return(NULL)
})

if (is.null(result)) {
    stop("TMFG returned NULL or failed")
}

cat("Result type:", class(result), "\\n")
cat("Result typeof:", typeof(result), "\\n")


adj_mat <- NULL

if (is.matrix(result)) {
    cat("Direct matrix\\n")
    adj_mat <- result
} else if (is.list(result)) {
    cat("List with names:", paste(names(result), collapse=", "), "\\n")

    if ("adj" %in% names(result)) {
        adj_mat <- result$adj
        cat("Used result$adj\\n")
    } else if ("network" %in% names(result)) {
        adj_mat <- result$network
        cat("Used result$network\\n")
    } else if (length(result) > 0) {
        for (name in names(result)) {
            if (is.matrix(result[[name]])) {
                adj_mat <- result[[name]]
                cat("Used result$", name, "\\n", sep="")
                break
            }
        }
    }
}

if (is.null(adj_mat)) {
    cat("ERROR: Could not extract adjacency matrix\\n")
    cat("Result structure:\\n")
    str(result)
    stop("Failed to extract adjacency")
}

adj_mat <- as.matrix(adj_mat)
cat("Adjacency dim:", dim(adj_mat), "\\n")
cat("Adjacency range:", range(adj_mat), "\\n")

write.table(adj_mat, '_temp_tmfg_result.csv', sep=',', row.names=FALSE, col.names=FALSE)

if (file.exists('_temp_tmfg_result.csv')) {
    info <- file.info('_temp_tmfg_result.csv')
    cat("File created, size:", info$size, "bytes\\n")

    if (info$size == 0) {
        cat("ERROR: File is empty!\\n")
    }
} else {
    cat("ERROR: File not created\\n")
}
'''
        with open('_temp_run_tmfg.R', 'w') as f:
            f.write(r_script)
        result = subprocess.run(['Rscript', '_temp_run_tmfg.R'],
                                capture_output=True, text=True)
        if result.returncode != 0:
            print(f"R error: {result.stderr}")
            raise RuntimeError("TMFG failed in R")
        print(result.stdout)
        adj_mat = np.loadtxt('_temp_tmfg_result.csv', delimiter=',')

        for f in ['_temp_corr.csv', '_temp_tmfg_result.csv', '_temp_run_tmfg.R']:
            if os.path.exists(f):
                os.remove(f)

        n_edges = int(np.sum(adj_mat > 0) / 2)
        print(f"TMFG completed: {adj_mat.shape}, {n_edges} edges")

        return {
            "adjacency": adj_mat,
            "edges": n_edges
        }

    def visualize_network(self, corr_matrix, membership=None, item_labels=None,output_file='network_plot.png', title='TMFG Network',
                          node_size=8, label_size=0.8):
        print(f"Creating network visualization: {output_file}")

        # Save correlation matrix
        np.savetxt('_temp_vis_corr.csv', corr_matrix, delimiter=',')
        # Prepare membership if provided
        if membership is not None:
            np.savetxt('_temp_vis_membership.csv', membership, delimiter=',', fmt='%d')
            has_membership = 'TRUE'
        else:
            has_membership = 'FALSE'

        if item_labels is not None:
            with open('_temp_vis_labels.csv', 'w') as f:
                for label in item_labels:
                    f.write(f"{label}\n")
            has_labels = 'TRUE'
        else:
            has_labels = 'FALSE'

        r_script = f'''
    library(EGAnet)
    library(qgraph)

    # Load correlation matrix
    corr_mat <- as.matrix(read.csv('_temp_vis_corr.csv', header=FALSE))
    cat("Correlation matrix loaded:", dim(corr_mat), "\\n")

    # Run TMFG to get network
    tmfg_result <- TMFG(corr_mat)

    # Extract adjacency matrix
    if (is.matrix(tmfg_result)) {{
        adj_mat <- tmfg_result
    }} else if (is.list(tmfg_result)) {{
        if ("adj" %in% names(tmfg_result)) {{
            adj_mat <- tmfg_result$adj
        }} else if ("network" %in% names(tmfg_result)) {{
            adj_mat <- tmfg_result$network
        }} else {{
            adj_mat <- tmfg_result[[1]]
        }}
    }} else {{
        stop("Could not extract adjacency matrix")
    }}

    # Load membership if provided
    if ({has_membership}) {{
        membership <- scan('_temp_vis_membership.csv', quiet=TRUE)
        cat("Membership loaded, unique communities:", length(unique(membership)), "\\n")
        groups <- membership
    }} else {{
        groups <- rep(1, nrow(adj_mat))
    }}

    # Load labels if provided
    if ({has_labels}) {{
        labels <- readLines('_temp_vis_labels.csv')
        cat("Labels loaded:", length(labels), "\\n")
    }} else {{
        labels <- paste0("Item", 1:nrow(adj_mat))
    }}

    # Create weighted adjacency for visualization
    weighted_adj <- adj_mat * corr_mat

    # Open PNG device
    png('{output_file}', width=1200, height=1200, res=150)

    # Plot using qgraph
    qgraph(weighted_adj,
           layout = "spring",
           groups = as.factor(groups),
           labels = labels,
           label.cex = {label_size},
           vsize = {node_size},
           edge.width = 1.5,
           edge.color = "darkgray",
           posCol = "darkblue",
           negCol = "darkred",
           title = "{title}",
           legend = TRUE,
           legend.cex = 0.5,
           border.width = 1.5,
           border.color = "black")

    dev.off()
    cat("Plot saved to {output_file}\\n")
    '''

        with open('_temp_run_vis.R', 'w') as f:
            f.write(r_script)

        result = subprocess.run(['Rscript', '_temp_run_vis.R'],
                                capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"R error: {result.stderr}")
            raise RuntimeError("Network visualization failed")

        # Cleanup temp files
        temp_files = ['_temp_vis_corr.csv', '_temp_run_vis.R']
        if membership is not None:
            temp_files.append('_temp_vis_membership.csv')
        if item_labels is not None:
            temp_files.append('_temp_vis_labels.csv')

        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)

        return output_file

    def compare_networks(self, orig_adj, trans_adj, orig_mem, trans_mem):
        orig_edges = int(orig_adj.sum() / 2)
        trans_edges = int(trans_adj.sum() / 2)

        # NMI
        if orig_mem is not None and trans_mem is not None:
            nmi = normalized_mutual_info_score(orig_mem, trans_mem)
        else:
            nmi = float('nan')

        if orig_adj.shape == trans_adj.shape:
            n = orig_adj.shape[0]
            idx = np.triu_indices(n, k=1)
            orig_weights = orig_adj[idx]
            trans_weights = trans_adj[idx]

            # Jaccard similarity on binarised edge sets
            orig_edges_bin = orig_weights > 0
            trans_edges_bin = trans_weights > 0
            intersection = np.sum(orig_edges_bin & trans_edges_bin)
            union = np.sum(orig_edges_bin | trans_edges_bin)
            jaccard = float(intersection / union) if union > 0 else float('nan')

            # Pearson correlation between weights of common edges only
            common_edges = orig_edges_bin & trans_edges_bin
            if common_edges.sum() > 1:
                orig_common = orig_weights[common_edges]
                trans_common = trans_weights[common_edges]
                if orig_common.std() > 0 and trans_common.std() > 0:
                    weight_corr = float(np.corrcoef(orig_common, trans_common)[0, 1])
                else:
                    weight_corr = float('nan')
            else:
                weight_corr = float('nan')
        else:
            jaccard = float('nan')
            weight_corr = float('nan')

        return {
            'orig_edges': orig_edges,
            'trans_edges': trans_edges,
            'nmi': nmi,
            'jaccard': jaccard,
            'weight_corr': weight_corr,
        }

    def calculate_membership(self, adjacency_matrix, algorithm='walktrap'):
        """
        Detect communities using R's igraph/EGAnet
        """
        n_items = adjacency_matrix.shape[0]
        print(f"\nDetecting communities from network ({n_items} items)...")
        print(f"  Using algorithm: {algorithm}")

        # Save adjacency matrix for R
        np.savetxt('_temp_adjacency.csv', adjacency_matrix, delimiter=',')

        # R script for community detection
        r_script = f"""
library(igraph)

# Load adjacency matrix
adj_mat <- as.matrix(read.csv('_temp_adjacency.csv', header=FALSE))

# Create graph from adjacency matrix
g <- graph_from_adjacency_matrix(adj_mat, mode='undirected', weighted=TRUE)

# Detect communities
if ('{algorithm}' == 'walktrap') {{
    communities <- cluster_walktrap(g)
}} else if ('{algorithm}' == 'louvain') {{
    communities <- cluster_louvain(g)
}} else if ('{algorithm}' == 'leiden') {{
    communities <- cluster_leiden(g)
}} else {{
    stop("Unknown algorithm: {algorithm}")
}}

# Get membership
membership <- membership(communities)
n_communities <- max(membership)

cat("Detected", n_communities, "communities\\n")
cat("Modularity:", modularity(communities), "\\n")

# Print community sizes
for (i in 1:n_communities) {{
    size <- sum(membership == i)
    cat("  Community", i, ":", size, "items\\n")
}}

# Save results
write.table(data.frame(membership=membership), 
            '_temp_membership.csv',
            sep=',', row.names=FALSE, col.names=TRUE)

write.table(data.frame(n_communities=n_communities),
            '_temp_n_communities.csv',
            sep=',', row.names=FALSE, col.names=FALSE)
"""

        # Run R script
        with open('_temp_detect_communities.R', 'w') as f:
            f.write(r_script)

        result = subprocess.run(['Rscript', '_temp_detect_communities.R'],
                                capture_output=True, text=True)

        print(result.stdout)

        if result.returncode != 0:
            print("R Error:", result.stderr)
            raise RuntimeError("Community detection failed in R")

        # Read results
        import pandas as pd
        membership_df = pd.read_csv('_temp_membership.csv')
        memberships = membership_df['membership'].values

        n_communities = int(np.loadtxt('_temp_n_communities.csv'))

        # Cleanup
        for f in ['_temp_adjacency.csv', '_temp_membership.csv',
                  '_temp_n_communities.csv', '_temp_detect_communities.R']:
            if os.path.exists(f):
                os.remove(f)

        return memberships, n_communities


# ===============================================================================
# ======================== Unique Variable Analysis =============================
class UVA:
    def run_uva_iterative(self, data, threshold=0.25, max_iter=10, n_sample=None):
        # Detect if input is raw data or correlation matrix
        is_raw_data = (data.shape[0] != data.shape[1])  # Not square = raw data

        if is_raw_data:
            # Raw data: [n_observations × n_items]
            n_observations = data.shape[0]
            n_items = data.shape[1]  # COLUMNS are items!
            print(f"\nDetected raw data: {n_observations} observations × {n_items} items")
            print(f"R will automatically use n={n_observations} from data")
            if n_sample is not None:
                print(f"Warning: n_sample={n_sample} specified but will be ignored for raw data")
        else:
            # Square matrix: [n_items × n_items] correlation/network
            n_items = data.shape[0]
            print(f"\nDetected correlation/network matrix: {n_items} × {n_items}")
            if n_sample is None:
                n_sample = 1000  # Default for correlation matrix
                print(f"Using default n_sample={n_sample}")
            else:
                print(f"Using specified n_sample={n_sample}")

        all_items = list(range(n_items))  # 0-indexed
        removed_items = []
        iteration_history = []

        print(f"Running UVA iteratively (threshold={threshold}):")
        print(f"Starting with {n_items} items\n")

        current_data = data.copy()
        current_items = all_items.copy()

        for iteration in range(1, max_iter + 1):
            print(f"--- Iteration {iteration} ---")
            print(f"Current items: {len(current_items)}")

            # Run UVA on current subset
            result = self._run_uva_once(current_data, threshold, is_raw_data, n_sample)

            if not result['redundant_items']:
                print("No redundancies found, stopping")
                break

            # Map back to original indices
            redundant_original_idx = [current_items[i] for i in result['redundant_items']]

            print(f"Found {len(redundant_original_idx)} redundant items: {redundant_original_idx}")

            # Record this iteration
            iteration_history.append({
                'iteration': iteration,
                'n_items_before': len(current_items),
                'redundant_items': redundant_original_idx,
                'n_removed': len(redundant_original_idx)
            })

            # Update for next iteration
            removed_items.extend(redundant_original_idx)
            current_items = [item for item in current_items if item not in redundant_original_idx]

            # Extract reduced data for next iteration
            keep_indices = [i for i in range(len(current_items)) if i not in result['redundant_items']]

            if is_raw_data:
                # For raw data: keep columns (items), all rows (observations)
                current_data = current_data[:, keep_indices]
            else:
                # For correlation: keep both rows and columns
                current_data = current_data[np.ix_(keep_indices, keep_indices)]

            print(f"Items after removal: {len(current_items)}\n")

        print(f"UVA Completed after {len(iteration_history)} iterations")
        print(f"Total removed: {len(removed_items)} items")
        print(f"Final: {n_items} → {len(current_items)} items\n")

        return {
            'kept_items': current_items,
            'removed_items': removed_items,
            'n_initial': n_items,
            'n_final': len(current_items),
            'n_iterations': len(iteration_history),
            'iteration_history': iteration_history
        }

    def _run_uva_once(self, data_matrix, threshold, is_raw_data, n_sample=None):
        """Run UVA once (single iteration)"""
        # Save data matrix
        np.savetxt('_temp_uva_data.csv', data_matrix, delimiter=',')

        # Build R script based on data type
        if is_raw_data:
            # For raw data: don't specify n (R will infer from rows)
            r_script = f"""
library(EGAnet)

cat("=== UVA with Raw Data ===\\n")
data_mat <- as.matrix(read.csv('_temp_uva_data.csv', header=FALSE))
cat("Data dimensions:", dim(data_mat), "\\n")
cat("  Rows (observations):", nrow(data_mat), "\\n")
cat("  Cols (variables/items):", ncol(data_mat), "\\n")

uva_result <- UVA(
    data = data_mat,
    cut.off = {threshold},
    reduce = TRUE
)
"""
        else:
            # For correlation matrix: must specify n
            r_script = f"""
library(EGAnet)

cat("=== UVA with Correlation Matrix ===\\n")
corr_mat <- as.matrix(read.csv('_temp_uva_data.csv', header=FALSE))
cat("Correlation matrix dimensions:", dim(corr_mat), "\\n")
cat("Sample size specified: {n_sample}\\n")

uva_result <- UVA(
    data = corr_mat,
    n = {n_sample},
    cut.off = {threshold},
    reduce = TRUE
)
"""

        # Common R code for processing results
        r_script += """
redundant_0idx <- integer(0)
redundant <- NULL

if (!is.null(uva_result$redundancy)) {
    # the document writes "redudant" but lets try both "redudant" and "redundant"

    if (!is.null(uva_result$redundancy$redundant)) {
        redundant <- uva_result$redundancy$redundant
        cat("Found in $redundancy$redundant\\n")
    } else if (!is.null(uva_result$redundancy$redudant)) {
        redundant <- uva_result$redundancy$redudant
        cat("Found in $redundancy$redudant\\n")
    } else {
        cat("Checking redundancy structure:\\n")
        cat("Names:", paste(names(uva_result$redundancy), collapse=", "), "\\n")
    }
}

# output handling
if (!is.null(redundant) && is.list(redundant) && length(redundant) > 0) {
    all_redundant <- unique(unlist(redundant))
    cat("Redundant items (R-indexed):", paste(all_redundant, collapse=", "), "\\n")
    redundant_0idx <- all_redundant - 1
    write.table(redundant_0idx, '_temp_redundant.csv',
                sep=',', row.names=FALSE, col.names=FALSE)
} else {
    cat("No redundant items found\\n")
    write.table(matrix(ncol=1, nrow=0), '_temp_redundant.csv',
                sep=',', row.names=FALSE, col.names=FALSE)
}

write.table(redundant_0idx, '_temp_redundant.csv', 
            sep=',', row.names=FALSE, col.names=FALSE)
"""

        with open('_temp_run_uva.R', 'w') as f:
            f.write(r_script)

        result = subprocess.run(['Rscript', '_temp_run_uva.R'],
                                capture_output=True, text=True)

        print(result.stdout)

        if result.returncode != 0:
            print("R Error:")
            print(result.stderr)
            raise RuntimeError("UVA failed")

        # Read results
        try:
            redundant_items = np.loadtxt('_temp_redundant.csv', delimiter=',', dtype=int)
            if redundant_items.ndim == 0:
                redundant_items = [int(redundant_items)] if redundant_items.size > 0 else []
            else:
                redundant_items = redundant_items.tolist()
        except:
            redundant_items = []

        # Cleanup
        for f in ['_temp_uva_data.csv', '_temp_redundant.csv', '_temp_run_uva.R']:
            if os.path.exists(f):
                os.remove(f)

        return {'redundant_items': redundant_items}


class BootEGA:
    """Bootstrap EGA with iterative stability testing"""

    def run_bootega_iterative(self, data, n_boots=100,
                              stability_threshold=0.75, max_iter=10):
        # Detect if input is raw data or correlation matrix
        is_raw_data = (data.shape[0] != data.shape[1])

        if is_raw_data:
            n_observations = data.shape[0]
            n_items = data.shape[1]
            print(f"\nDetected raw data: {n_observations} observations × {n_items} items")
        else:
            n_items = data.shape[0]
            print(f"\nDetected correlation matrix: {n_items} × {n_items}")

        all_items = list(range(n_items))
        removed_items = []
        iteration_history = []

        print(f"Running bootEGA iteratively ({n_boots} boots/iter, threshold={stability_threshold})...")
        print(f"Starting with {n_items} items")
        print("=" * 60)

        current_data = data.copy()
        current_items = all_items.copy()

        for iteration in range(1, max_iter + 1):
            print(f"\n--- bootEGA Iteration {iteration} ---")
            print(f"Current items: {len(current_items)}")

            # Run bootEGA once
            result = self._run_bootega_once(current_data, n_boots, iteration, is_raw_data)

            # Check item stability
            item_stability = result['item_stability']

            if 'item.stability' in item_stability.columns:
                stability_col = 'item.stability'
            else:
                # Sometimes the column name might be different
                stability_col = item_stability.columns[0]

            # Find unstable items (< threshold)
            unstable_mask = item_stability[stability_col] < stability_threshold
            unstable_indices = item_stability.index[unstable_mask].tolist()

            if not unstable_indices:
                print(f"✓ All items stable (≥ {stability_threshold}) - stopping")
                iteration_history.append({
                    'iteration': iteration,
                    'n_items': len(current_items),
                    'structural_consistency': result['structural_consistency'],
                    'min_stability': item_stability[stability_col].min(),
                    'mean_stability': item_stability[stability_col].mean(),
                    'unstable_items': [],
                    'n_removed': 0
                })
                break

            # Map back to original indices
            unstable_original_idx = [current_items[i] for i in unstable_indices]

            print(f"Found {len(unstable_original_idx)} unstable items:")
            for idx in unstable_indices:
                orig_idx = current_items[idx]
                stab = item_stability.loc[idx, stability_col]
                print(f"  Item {orig_idx}: stability = {stab:.3f}")

            # Record this iteration
            iteration_history.append({
                'iteration': iteration,
                'n_items': len(current_items),
                'structural_consistency': result['structural_consistency'],
                'min_stability': item_stability[stability_col].min(),
                'mean_stability': item_stability[stability_col].mean(),
                'unstable_items': unstable_original_idx,
                'n_removed': len(unstable_original_idx)
            })

            # Update for next iteration
            removed_items.extend(unstable_original_idx)
            current_items = [item for item in current_items if item not in unstable_original_idx]

            # Extract reduced data for next iteration
            keep_indices = [i for i in range(len(current_items)) if i not in unstable_indices]

            if is_raw_data:
                # For raw data: keep columns (items), all rows (observations)
                current_data = current_data[:, keep_indices]
            else:
                # For correlation: keep both rows and columns
                current_data = current_data[np.ix_(keep_indices, keep_indices)]

            print(f"Removing {len(unstable_original_idx)} unstable items")
            print(f"Items after removal: {len(current_items)}")

        print("\n" + "=" * 60)
        print(f"bootEGA Completed after {len(iteration_history)} iterations")
        print(f"Final: {n_items} → {len(current_items)} items")
        print(f"Total removed: {len(removed_items)} items (unstable)")
        print("=" * 60)

        return {
            'kept_items': current_items,
            'removed_items': removed_items,
            'n_initial': n_items,
            'n_final': len(current_items),
            'n_iterations': len(iteration_history),
            'iteration_history': iteration_history
        }

    def _run_bootega_once(self, data_matrix, n_boots, iteration_num, is_raw_data):
        """Run bootEGA once (single iteration)"""

        print(f"  Running {n_boots} bootstrap samples...")

        np.savetxt('_temp_bootega_data.csv', data_matrix, delimiter=',')

        if is_raw_data:
            # For raw data: use resampling bootstrap
            r_script = f"""
library(EGAnet)

cat("=== bootEGA with Raw Data ===\\n")
data_mat <- as.matrix(read.csv('_temp_bootega_data.csv', header=FALSE))
cat("Data dimensions:", dim(data_mat), "\\n")
cat("  Rows (observations):", nrow(data_mat), "\\n")
cat("  Cols (variables/items):", ncol(data_mat), "\\n")

boot_result <- bootEGA(
    data = data_mat,
    iter = {n_boots},
    type = "resampling",
    seed = 42,
    model = "TMFG",
    ncores = 1
)
"""
        else:
            # For correlation: use parametric bootstrap (needs sample size)
            n_items = data_matrix.shape[0]
            n_sample = n_items * 10  # Default sample size for correlation
            r_script = f"""
library(EGAnet)

cat("=== bootEGA with Correlation Matrix ===\\n")
corr_mat <- as.matrix(read.csv('_temp_bootega_data.csv', header=FALSE))
cat("Correlation matrix dimensions:", dim(corr_mat), "\\n")
cat("Using n={n_sample} for parametric bootstrap\\n")

boot_result <- bootEGA(
    data = corr_mat,
    n = {n_sample},
    iter = {n_boots},
    type = "parametric",
    seed = 42,
    model = "TMFG",
    ncores = 1
)
"""

        # Common R code
        r_script += """
# Item stability
item_stability <- boot_result$item.stability$item.stability

# Structural consistency
struct_consist <- boot_result$frequency
consistency_value <- max(struct_consist)

cat("Structural consistency:", consistency_value, "\\n")
cat("Item stability range: [", min(item_stability), ",", max(item_stability), "]\\n")

# Save
write.table(data.frame(item.stability=item_stability), 
            '_temp_item_stability.csv',
            sep=',', row.names=FALSE, col.names=TRUE)

write.table(data.frame(consistency=consistency_value), 
            '_temp_struct_consistency.csv',
            sep=',', row.names=FALSE, col.names=TRUE)
"""

        with open('_temp_run_bootega.R', 'w') as f:
            f.write(r_script)

        result = subprocess.run(['Rscript', '_temp_run_bootega.R'],
                                capture_output=True, text=True)

        print(result.stdout)

        if result.returncode != 0:
            print("R Error:", result.stderr)
            raise RuntimeError("bootEGA failed")

        item_stability = pd.read_csv('_temp_item_stability.csv')
        struct_consistency = pd.read_csv('_temp_struct_consistency.csv')

        for f in ['_temp_bootega_data.csv', '_temp_item_stability.csv',
                  '_temp_struct_consistency.csv', '_temp_run_bootega.R']:
            if os.path.exists(f):
                os.remove(f)

        return {
            'item_stability': item_stability,
            'structural_consistency': float(struct_consistency['consistency'].values[0])
        }

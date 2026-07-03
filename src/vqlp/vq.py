import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from .utils import (
    exp_kernel_bw_perplexity,
    lp_norm_stable,
    lp_centroid_objective_p_root,
    lp_centroid_gradient_p_root,
)
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

try:
    import faiss as _faiss_module
    _FAISS_AVAILABLE = True
except ImportError:
    _faiss_module = None
    _FAISS_AVAILABLE = False

__all__ = [
    "VQRecaller",
    "VQFitter",
]

class VQRecaller:
    """
    Recall analysis for Vector Quantizer results.

    Computes and stores analysis products from applying a set of prototypes
    to data, including Best Matching Units, connectivity matrices, receptive
    fields, and reconstruction capabilities.

    Nearest-prototype search is performed via FAISS, which supports arbitrary
    Lp metrics. Both exact (flat) and approximate (IVF) search are available
    via the index_type parameter.

    Note: FAISS operates in float32 internally. Input data is cast to float32
    for the search step regardless of input dtype.

    Terminology
    -----------
    Prototype (W[m]) : One of M representative vectors in feature space,
        also called a codebook vector or neuron.  Shape (d,).
    BMU (Best Matching Unit) : The prototype closest to a given observation
        under the chosen Lp metric.  A 2nd BMU, 3rd BMU, etc. are the next
        closest prototypes.  Controlled by ``max_bmu``.
    QE (Quantization Error) : The Lp distance from an observation to its
        assigned BMU.  Lower is better; zero means the observation sits
        exactly on a prototype.
    RF (Receptive Field) : The set of observations for which a given
        prototype is the 1st BMU.  Analogous to a Voronoi cell.
    RFSize : The number of observations in each prototype's receptive field.
        A prototype with RFSize == 0 is "dead" (no data point maps to it).
    CADJ (Co-Adjacency matrix) : An M×M integer matrix where CADJ[i, j]
        counts the number of observations whose 1st BMU is prototype i and
        whose 2nd BMU is prototype j.  Asymmetric by construction.  Encodes
        the directed manifold neighborhood of the codebook.
    CONN (Connectivity matrix) : The symmetric version of CADJ:
        CONN = CADJ + CADJ^T.  Used for undirected neighborhood queries.
        CONN[i, j] > 0 means prototypes i and j are topological neighbors.
    """
    
    def __init__(self, p=2, max_bmu=2, index_type="flat", nlist=None, nprobe=None, verbose=True):
        """
        Initialize VQRecaller.
        
        Parameters
        ----------
        p : float, default=2
            Order of the Lp distance metric. Any positive value is supported
            for exact search (index_type="flat"). Approximate search
            (index_type="ivf") is restricted to p in {1, 2}.
        max_bmu : int, default=2
            Number of Best Matching Units to track for each observation
        index_type : str, default="flat"
            FAISS index type for nearest-neighbor search:
            - "flat": Exact brute-force search via IndexFlat. Correct for any
              p and recommended for codebooks up to a few thousand prototypes,
              and for any iterative fitting loop.
            - "ivf": Approximate search via IndexIVFFlat. Useful for very
              large codebooks where exact search is a bottleneck. Not suitable
              for iterative fitting (non-deterministic assignments interfere
              with convergence) and restricted to p in {1, 2}.
        nlist : int, optional
            (IVF only) Number of Voronoi cells. Defaults to sqrt(M).
        nprobe : int, optional
            (IVF only) Number of cells searched at query time. Higher values
            give better recall at the cost of speed. Defaults to nlist // 10.
        verbose : bool, default=True
            If True, print progress messages during recall analysis.
            Set to False to suppress all terminal output.
        """
        if index_type not in ("flat", "ivf"):
            raise ValueError(f"index_type must be 'flat' or 'ivf', got {index_type!r}")
        if index_type == "ivf" and p not in (1, 2):
            raise ValueError(
                f"Approximate (IVF) search supports only p in {{1, 2}}, got p={p}. "
                f"Use index_type='flat' for exotic Lp metrics."
            )

        # Model parameters
        self.p = p
        self.max_bmu = max_bmu
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.verbose = verbose
        
        # Recall products (computed by recall())
        self.BMU = None          # Best Matching Unit indices (N, max_bmu)
        self.QE = None           # Quantization Errors (N, max_bmu)
        self.AFF = None          # Affinities (N, max_bmu): Neuralware-form soft weights from QE
        self.RF = None           # Receptive Field: list of observation indices per prototype
        self.RFSize = None       # Size of Receptive Field per prototype
        self.CONN = None         # Connectivity matrix (sparse, symmetric)
        self.CONN_nhbs = None    # List of nonzero indices for each row of CONN
        self.CONN_nhbs_size = None  # Size of each CONN_nhbs[i] as numpy array
        self.CADJ = None         # Asymmetric co-adjacency matrix (sparse); CONN = CADJ + CADJ^T
        self.CADJ_nhbs = None    # List of nonzero indices for each row of CADJ
        self.CADJ_nhbs_size = None  # Size of each CADJ_nhbs[i] as numpy array

        # Label recall products (computed by recall_labels())
        self.WL = None           # Winning label per prototype, object array (M,); None for empty RFs
        self.WL_Dist = None      # Fuzzy label frequency table (M, n_unique_labels), row-normalized
        self.WL_Purity = None    # Hellinger-based purity score per prototype (M,)
        self.WL_unq = None       # Sorted unique label values; column j of WL_Dist corresponds to WL_unq[j]

        # Store shape info for validation (don't store X itself)
        self._M = None           # Number of prototypes
        self._N = None           # Number of observations
        self._d = None           # Dimensionality
    
    def recall(self, X=None, W=None, p=None, max_bmu=None, labels=None):
        """
        Perform recall analysis. Can operate in two modes:
        
        Mode 1 - Full analysis: Provide X and W to compute everything from scratch
        Mode 2 - Finalize: Provide no arguments to compute derived quantities 
                          from existing BMU/QE (assumes update_BMU() was called)
        
        Parameters
        ----------
        X : array-like, shape (N, d), optional
            Data matrix with N observations and d features
        W : array-like, shape (M, d), optional  
            Prototype matrix with M prototypes and d features
        p : float, optional
            Lp distance order. If None, uses self.p
        max_bmu : int, optional
            Number of BMUs to track. If None, uses self.max_bmu
        labels : array-like, shape (N,), optional
            Observation labels. Any hashable type is accepted (int, str, float,
            etc.). If provided, recall_labels() is called automatically after
            the main recall pipeline. Can also be called independently via
            recall_labels() on an already-recalled object.
            
        Returns
        -------
        self : VQRecaller
            Returns self for method chaining
        """
        # Update parameters if provided
        if p is not None:
            if self.index_type == "ivf" and p not in (1, 2):
                raise ValueError(
                    f"Approximate (IVF) search supports only p in {{1, 2}}, got p={p}."
                )
            self.p = p
        if max_bmu is not None:
            self.max_bmu = max_bmu
        
        # Mode 1: Full analysis from scratch
        if X is not None and W is not None:
            self.update_BMU(X, W)
            
        # Mode 2: Finalize existing BMU/QE
        elif X is None and W is None:
            # Validate that BMU and QE exist and are consistent
            if self.BMU is None or self.QE is None:
                raise RuntimeError("BMU and QE must be populated first. Call update_BMU() or provide X and W.")
            if self.BMU.shape != self.QE.shape:
                raise RuntimeError(f"BMU shape {self.BMU.shape} doesn't match QE shape {self.QE.shape}")
            if self._N is None or self._M is None:
                raise RuntimeError("Shape information missing. Call update_BMU() first.")
            
        # Invalid: partial arguments provided
        else:
            raise ValueError("Either provide both X and W, or neither (to finalize existing BMU/QE)")
        
        # Compute the derived recall products (cheap operations)
        self._compute_receptive_fields()
        self._compute_connectivity_matrix()
        
        if labels is not None:
            self.recall_labels(labels)
        
        if self.verbose:
            print(f"Recall analysis complete. MQE: {np.mean(self.QE[:, 0]):.6f}")
        return self
    
    def update_BMU(self, X, W):
        """
        Update Best Matching Units (BMU) and Quantization Errors (QE) via FAISS.
        
        This is the computationally expensive part of recall analysis.
        Can be called repeatedly during iterative fitting algorithms.
        
        Parameters
        ----------
        X : array-like, shape (N, d)
            Data matrix with N observations and d features
        W : array-like, shape (M, d)
            Prototype matrix with M prototypes and d features
        """
        X = np.array(X)
        W = np.array(W)
        
        # Store shape info for validation
        self._N, self._d = X.shape
        self._M = W.shape[0]
        
        if W.shape[1] != self._d:
            raise ValueError(f"W has {W.shape[1]} features but X has {self._d} features")
        if self.max_bmu > self._M:
            raise ValueError(f"max_bmu={self.max_bmu} exceeds number of prototypes M={self._M}")

        # FAISS requires float32 and contiguous arrays
        X_f32 = np.ascontiguousarray(X, dtype=np.float32)
        W_f32 = np.ascontiguousarray(W, dtype=np.float32)

        index = self._build_index(W_f32)
        distances, indices = index.search(X_f32, self.max_bmu)

        self.BMU = indices.astype(int)
        self.QE = distances.astype(X.dtype)
        self._compute_affinity()

    def _build_index(self, W_f32):
        """Build and populate the FAISS index for prototype set W."""
        if not _FAISS_AVAILABLE:
            raise ImportError(
                "FAISS is required for nearest-neighbor search but is not installed. "
                "Install it with: pip install faiss-cpu"
            )
        faiss = _faiss_module  # local alias for readability

        if self.index_type == "flat":
            index = faiss.IndexFlat(self._d, faiss.METRIC_Lp)
            index.metric_arg = float(self.p)
            index.add(W_f32)
            return index

        # IVF approximate search (p in {1, 2}, enforced at construction)
        nlist = self.nlist if self.nlist is not None else int(np.clip(int(np.sqrt(self._M)), 1, 65536))
        nlist = min(nlist, self._M)

        if self.p == 2:
            quantizer = faiss.IndexFlatL2(self._d)
            index = faiss.IndexIVFFlat(quantizer, self._d, nlist, faiss.METRIC_L2)
        else:  # p == 1
            quantizer = faiss.IndexFlat(self._d, faiss.METRIC_L1)
            index = faiss.IndexIVFFlat(quantizer, self._d, nlist, faiss.METRIC_L1)

        index.train(W_f32)
        index.add(W_f32)
        index.nprobe = self.nprobe if self.nprobe is not None else max(1, nlist // 10)
        return index

    def _compute_affinity(self):
        """
        Compute affinity matrix AFF from quantization errors QE.

        Uses the Neuralware form: the first BMU always gets affinity 1 before
        normalization, and each subsequent BMU k gets QE[:,0] / (QE[:,0] + QE[:,k]).
        Rows are then normalized to sum to 1. Division-by-zero entries (when both
        QE[:,0] and QE[:,k] are zero, i.e. the observation sits exactly on a
        prototype) are set to 0 before normalization, which concentrates all
        weight on the first BMU.

        This mirrors cpp_QE2Affinity() from AnnoyVQRecall.hpp.

        Result is stored as self.AFF with the same shape as self.QE.
        """
        AFF = np.ones_like(self.QE)  # column 0 = 1 for all rows
        q0 = self.QE[:, 0]
        for k in range(1, self.QE.shape[1]):
            denom = q0 + self.QE[:, k]
            # Where denom == 0, both prototypes coincide with the point;
            # set to 0 so row-normalization assigns all weight to BMU 0.
            with np.errstate(invalid="ignore", divide="ignore"):
                AFF[:, k] = np.where(denom > 0, q0 / denom, 0.0)
        # Row-normalize
        row_sums = AFF.sum(axis=1, keepdims=True)
        # row_sums should always be > 0 (col 0 is 1), but guard anyway
        AFF = np.where(row_sums > 0, AFF / row_sums, AFF)
        self.AFF = AFF

    def _compute_receptive_fields(self):
        """
        Compute receptive fields (RF) and their sizes (RFSize).

        Uses numpy argsort + searchsorted to avoid a Python-level loop over N
        observations. Observation indices are sorted by their first BMU, then
        sliced per prototype in O(N log N) total rather than O(N*M).
        """
        # Sort observation indices by their first BMU assignment
        order = np.argsort(self.BMU[:, 0], kind='stable')
        sorted_bmu = self.BMU[order, 0]

        # Find the start/end position of each prototype's block in the sorted array
        splits = np.searchsorted(sorted_bmu, np.arange(self._M + 1))

        # Slice: RF[i] is the array of observation indices assigned to prototype i
        self.RF = [order[splits[i]:splits[i + 1]] for i in range(self._M)]
        self.RFSize = np.diff(splits).astype(int)

        if self.verbose:
            print("Receptive fields computed.")
    
    def _compute_connectivity_matrix(self):
        """
        Compute the co-adjacency matrix (CADJ) and its symmetrised form (CONN),
        along with neighbor lists and sizes for both.

        CADJ[i, j] counts how many observations have prototype i as their 1st
        BMU and prototype j as their 2nd BMU (asymmetric). CONN = CADJ + CADJ^T
        is the symmetric version used for undirected connectivity.

        Builds both matrices from BMU column vectors using scipy sparse CSR
        construction, avoiding any Python-level loop over N observations.
        """
        if self.max_bmu < 2:
            if self.verbose:
                print("Warning: max_bmu < 2. Connectivity matrix will be empty.")
            self.CONN = None
            self.CONN_nhbs = None
            self.CONN_nhbs_size = None
            self.CADJ = None
            self.CADJ_nhbs = None
            self.CADJ_nhbs_size = None
            return

        # Each observation votes for an edge from its 1st BMU to its 2nd BMU.
        # csr_matrix((data, (rows, cols))) places data[k] at CADJ[rows[k], cols[k]],
        # so rows must carry the 1st BMU and cols the 2nd BMU to satisfy the
        # definition CADJ[i, j] = #{obs : 1st BMU = i, 2nd BMU = j}.
        rows = self.BMU[:, 0]          # 1st BMU → row of CADJ
        cols = self.BMU[:, 1]          # 2nd BMU → col of CADJ
        data = np.ones(self._N, dtype=int)
        self.CADJ = csr_matrix((data, (rows, cols)), shape=(self._M, self._M))

        # Extract CADJ neighbor lists and sizes directly from CSR structure
        self.CADJ_nhbs = [
            self.CADJ.indices[self.CADJ.indptr[i]:self.CADJ.indptr[i + 1]].tolist()
            for i in range(self._M)
        ]
        self.CADJ_nhbs_size = np.diff(self.CADJ.indptr).astype(int)

        # Symmetrise to get CONN
        self.CONN = self.CADJ + self.CADJ.T

        # Extract CONN neighbor lists and sizes
        conn_csr = self.CONN.tocsr()
        self.CONN_nhbs = [
            conn_csr.indices[conn_csr.indptr[i]:conn_csr.indptr[i + 1]].tolist()
            for i in range(self._M)
        ]
        self.CONN_nhbs_size = np.diff(conn_csr.indptr).astype(int)

        if self.verbose:
            print("Connectivity matrix computed.")

    def recall_labels(self, labels):
        """
        Compute per-prototype label summaries from observation labels.

        Must be called after update_BMU() (or recall()) so that self.BMU,
        self.AFF, self.RF, and self.RFSize are available.

        For each observation i, its label receives fractional credit distributed
        across all max_bmu prototypes in proportion to their affinity weights
        (self.AFF[i, :]). This produces a fuzzy frequency table (WL_Dist) whose
        rows are then normalized to probability distributions. The winning label
        per prototype is the argmax of that distribution, and purity is derived
        from the Hellinger distance between the distribution and the ideal
        one-hot encoding of the winner.

        This mirrors VQRecallLabels_worker / cpp_RecallLabels() from
        AnnoyVQRecall.hpp.

        Parameters
        ----------
        labels : array-like, shape (N,)
            Observation labels. Any hashable type is accepted (int, str,
            float, etc.). Length must equal the number of observations used
            in the most recent update_BMU() / recall() call.

        Returns
        -------
        self : VQRecaller
            Returns self for method chaining.

        Attributes set
        --------------
        WL : np.ndarray of object, shape (M,)
            Winning label for each prototype. None for prototypes whose
            receptive field is empty (RFSize == 0).
        WL_Dist : np.ndarray of float, shape (M, n_unique_labels)
            Row-normalized fuzzy label frequency table. Column j corresponds
            to WL_unq[j]. Rows for empty RFs are all-zero.
        WL_Purity : np.ndarray of float, shape (M,)
            Hellinger-based purity score in [0, 1]. 1 means all affinity mass
            falls on a single label; 0 for empty RFs.
        WL_unq : np.ndarray, shape (n_unique_labels,)
            Sorted unique label values. Use to interpret WL_Dist columns.
        """
        if self.BMU is None or self.AFF is None:
            raise RuntimeError(
                "BMU and AFF must be populated first. Call update_BMU() or recall() before recall_labels()."
            )

        labels = np.asarray(labels)
        if labels.shape[0] != self._N:
            raise ValueError(
                f"labels has {labels.shape[0]} entries but recall used {self._N} observations."
            )

        # Discover label universe and build index map {label -> col index}
        WL_unq, label_indices = np.unique(labels, return_inverse=True)
        n_labels = len(WL_unq)

        # Accumulate fuzzy frequency table via scatter-add.
        # For each of the max_bmu columns k, every observation i contributes
        # AFF[i, k] to the cell (BMU[i, k], label_col_of_i).
        WL_Dist = np.zeros((self._M, n_labels), dtype=float)
        for k in range(self.max_bmu):
            proto_indices = self.BMU[:, k]          # shape (N,)
            aff_weights   = self.AFF[:, k]          # shape (N,)
            # Scatter-add: WL_Dist[proto_indices[i], label_indices[i]] += aff_weights[i]
            np.add.at(WL_Dist, (proto_indices, label_indices), aff_weights)

        # Row-normalize to get probability distributions; leave empty-RF rows as zeros
        row_sums = WL_Dist.sum(axis=1, keepdims=True)
        active = (row_sums > 0).ravel()             # boolean mask of non-empty RFs
        WL_Dist[active] /= row_sums[active]

        # Winning label and Hellinger purity
        WL = np.empty(self._M, dtype=object)        # object dtype accepts any label type
        WL[:] = None                                 # default: no label for empty RFs
        WL_Purity = np.zeros(self._M, dtype=float)

        winner_cols = np.argmax(WL_Dist, axis=1)    # shape (M,); 0 for empty rows (harmless)
        for i in np.where(active)[0]:
            WL[i] = WL_unq[winner_cols[i]]
            p_winner = WL_Dist[i, winner_cols[i]]
            # Hellinger distance between the prototype's label distribution q
            # and the ideal one-hot e_winner is:
            #   H(q, e) = (1/sqrt(2)) * sqrt( sum_k (sqrt(q_k) - sqrt(e_k))^2 )
            # For a one-hot e_winner the sum collapses to:
            #   H^2 * 2 = (sqrt(p_winner) - 1)^2 + (1 - p_winner)
            #           = 2 - 2*sqrt(p_winner)
            # => H = sqrt(1 - sqrt(p_winner))
            # Purity is defined as 1 - H, so it equals 1 for a pure prototype
            # (all mass on one label) and 0 for an empty RF.
            hell_dist = np.sqrt(max(0.0, 1.0 - np.sqrt(p_winner)))
            WL_Purity[i] = 1.0 - hell_dist

        self.WL      = WL
        self.WL_Dist = WL_Dist
        self.WL_Purity = WL_Purity
        self.WL_unq  = WL_unq

        n_labeled = int(active.sum())
        if self.verbose:
            print(
                f"Label recall complete. {n_labeled}/{self._M} prototypes labeled "
                f"({n_labels} unique labels). Mean purity: {WL_Purity[active].mean():.4f}"
            )
        return self
    
    def reconstruct(self, W, X, method="hard"):
        """
        Reconstruct data using the quantizer.
        
        Parameters
        ----------
        W : array-like, shape (M, d)
            Prototype matrix with M prototypes and d features
        X : array-like, shape (N, d)
            Data matrix to reconstruct with N observations and d features
        method : str, default="hard"
            Reconstruction method:
            - "hard": Use closest prototype for each observation (respects self.p)
            - "soft": Weighted average using UMAP-style local connectivity.
              Always uses Euclidean (L2) distances internally; see
              _reconstruct_soft for rationale.
            
        Returns
        -------
        np.ndarray : Reconstructed data with shape (N, d)
        """
        if self.BMU is None:
            raise RuntimeError("Must call recall() before reconstruct()")
        
        W = np.array(W)
        X = np.array(X)
        
        # Validate dimensions
        if W.shape[0] != self._M:
            raise ValueError(f"W has {W.shape[0]} prototypes but recall used {self._M}")
        if X.shape[0] != self._N:
            raise ValueError(f"X has {X.shape[0]} observations but recall used {self._N}")
        if X.shape[1] != self._d or W.shape[1] != self._d:
            raise ValueError(f"Dimension mismatch: expected d={self._d}")
        
        if method == "hard":
            return self._reconstruct_hard(X, W)
        elif method == "soft":
            return self._reconstruct_soft(X, W)
        else:
            raise ValueError(f"Unknown reconstruction method: {method}")
    
    def _reconstruct_hard(self, X, W):
        """Hard reconstruction using closest prototypes."""
        first_bmu_indices = self.BMU[:, 0]
        return W[first_bmu_indices].astype(X.dtype)
    
    def _reconstruct_soft(self, X, W):
        """Soft reconstruction using UMAP-style weighted averaging.

        Note: distances to local prototypes are always computed with the
        Euclidean (L2) metric, regardless of self.p.  This is intentional:
        the UMAP perplexity-based bandwidth calibration (exp_kernel_bw_perplexity)
        is defined in terms of Euclidean distances, and the reconstruction is
        a geometric interpolation in feature space rather than a strict Lp
        operation.  If you need Lp-consistent reconstruction, use method='hard'.
        """
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        
        QX = np.zeros_like(X, dtype=np.float32)
        
        # Process each prototype's receptive field
        for i in range(self._M):
            if self.RFSize[i] == 0:
                continue
                
            obs_indices_in_rf = self.RF[i]
            nhbs_indices = np.array(list(set([i] + self.CONN_nhbs[i])), dtype=int)
            
            X_subset = X[obs_indices_in_rf, :]
            W_subset = W[nhbs_indices, :]
            
            # Compute distances from observations to local prototypes
            dist_matrix = cdist(X_subset, W_subset, metric='euclidean')
            
            # Reconstruct each observation in this receptive field
            for j, original_idx in enumerate(obs_indices_in_rf):
                dists = dist_matrix[j, :]
                
                # Calculate UMAP-style weights
                sigma = exp_kernel_bw_perplexity(dists)
                weights = np.exp(-(dists - np.min(dists)) / sigma)
                
                # Weighted average reconstruction
                sum_weights = np.sum(weights)
                if sum_weights == 0:
                    # Fallback to closest prototype
                    closest_idx = np.argmin(dists)
                    QX[original_idx, :] = W_subset[closest_idx, :]
                else:
                    QX[original_idx, :] = np.sum(weights[:, np.newaxis] * W_subset, axis=0) / sum_weights
        
        if self.verbose:
            print("Soft reconstruction complete.")
        return QX
    
    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path):
        """
        Save the VQRecaller to an HDF5 file.

        Parameters
        ----------
        path : str
            Destination file path.  Created or overwritten.

        See Also
        --------
        VQRecaller.load : Reconstruct a VQRecaller from a saved file.

        Examples
        --------
        >>> recaller.save("recaller.h5")
        >>> recaller2 = VQRecaller.load("recaller.h5")
        """
        from .io import VQRecallerIO
        VQRecallerIO(self).save(path)

    @classmethod
    def load(cls, path):
        """
        Load a VQRecaller from an HDF5 file created by :meth:`save`.

        Parameters
        ----------
        path : str
            Path to the HDF5 file.

        Returns
        -------
        VQRecaller

        Examples
        --------
        >>> recaller = VQRecaller.load("recaller.h5")
        """
        from .io import VQRecallerIO
        return VQRecallerIO.load(path)

    def get_summary(self):
        """
        Get a summary of the recall analysis results.

        Returns
        -------
        dict
            Summary statistics with the following keys:

            Always present after recall():
              - ``"M"`` : int — number of prototypes
              - ``"N"`` : int — number of observations
              - ``"d"`` : int — feature dimensionality
              - ``"p_norm"`` : float — Lp order used
              - ``"max_bmu"`` : int — number of BMUs tracked
              - ``"index_type"`` : str — FAISS index type used
              - ``"mean_quantization_error"`` : float — mean QE over 1st BMUs
              - ``"empty_prototypes"`` : int — number of prototypes with no observations
              - ``"largest_rf_size"`` : int — size of the largest receptive field
              - ``"connectivity_edges"`` : int — number of unique CONN edges (nnz // 2)
              - ``"mean_connectivity_degree"`` : float — mean number of CONN neighbors
              - ``"max_connectivity_degree"`` : int — max number of CONN neighbors

            Present only after recall_labels():
              - ``"n_unique_labels"`` : int
              - ``"labeled_prototypes"`` : int — prototypes with non-empty RF
              - ``"mean_label_purity"`` : float — Hellinger-based purity, mean over active prototypes
              - ``"min_label_purity"`` : float — worst-case purity over active prototypes
        """
        if self.BMU is None:
            return {"status": "No recall analysis performed"}
        
        summary = {
            "M": self._M,                    # Number of prototypes
            "N": self._N,                    # Number of observations  
            "d": self._d,                    # Dimensionality
            "p_norm": self.p,
            "max_bmu": self.max_bmu,
            "index_type": self.index_type,
            "mean_quantization_error": np.mean(self.QE[:, 0]),
            "empty_prototypes": np.sum(self.RFSize == 0),
            "largest_rf_size": np.max(self.RFSize) if self.RFSize is not None else 0,
            "connectivity_edges": self.CONN.nnz // 2 if self.CONN is not None else 0,
            "mean_connectivity_degree": np.mean(self.CONN_nhbs_size) if self.CONN_nhbs_size is not None else 0,
            "max_connectivity_degree": np.max(self.CONN_nhbs_size) if self.CONN_nhbs_size is not None else 0,
        }

        if self.WL is not None:
            active = self.RFSize > 0
            summary.update({
                "n_unique_labels":       len(self.WL_unq),
                "labeled_prototypes":    int(active.sum()),
                "mean_label_purity":     float(self.WL_Purity[active].mean()) if active.any() else float("nan"),
                "min_label_purity":      float(self.WL_Purity[active].min())  if active.any() else float("nan"),
            })

        return summary

    # ------------------------------------------------------------------
    # Serialization (to_dict / from_dict)
    # ------------------------------------------------------------------

    def to_dict(self):
        """
        Return a plain dict representation of this VQRecaller.

        All values are Python/numpy primitives (arrays, sparse matrices,
        ragged lists, scalars, or None) — no HDF5 types.  Intended for use
        with VQRecallerIO, which handles HDF5 encoding via _write_dict.

        Returns
        -------
        dict
        """
        d = {
            # Hyperparameters
            "p":          self.p,
            "max_bmu":    self.max_bmu,
            "index_type": self.index_type,
            "verbose":    self.verbose,
            "nlist":      self.nlist,
            "nprobe":     self.nprobe,
            # Shape metadata
            "_M": self._M,
            "_N": self._N,
            "_d": self._d,
            # Dense arrays (None if not yet recalled)
            "BMU":            self.BMU,
            "QE":             self.QE,
            "AFF":            self.AFF,
            "RFSize":         self.RFSize,
            "WL_Dist":        self.WL_Dist,
            "WL_Purity":      self.WL_Purity,
            "CADJ_nhbs_size": self.CADJ_nhbs_size,
            "CONN_nhbs_size": self.CONN_nhbs_size,
            # Object array with None entries (WL)
            "WL":     self.WL,
            "WL_unq": self.WL_unq,
            # Sparse matrices
            "CADJ": self.CADJ,
            "CONN": self.CONN,
            # Ragged lists
            "RF":        self.RF,
            "CADJ_nhbs": self.CADJ_nhbs,
            "CONN_nhbs": self.CONN_nhbs,
        }
        return d

    @classmethod
    def from_dict(cls, d):
        """
        Reconstruct a VQRecaller from a dict produced by :meth:`to_dict`.

        Bypasses ``__init__`` so that all saved attributes are restored
        directly without re-running any computation.

        Parameters
        ----------
        d : dict
            As returned by :meth:`to_dict`.

        Returns
        -------
        VQRecaller
        """
        obj = cls.__new__(cls)

        # Hyperparameters
        obj.p          = float(d["p"])
        obj.max_bmu    = int(d["max_bmu"])
        obj.index_type = str(d["index_type"])
        obj.verbose    = bool(d["verbose"])
        obj.nlist      = d.get("nlist")
        obj.nprobe     = d.get("nprobe")

        # Shape metadata (-1 sentinel → None for missing values)
        def _int_or_none(v):
            v = int(v) if v is not None else -1
            return None if v == -1 else v

        obj._M = _int_or_none(d.get("_M"))
        obj._N = _int_or_none(d.get("_N"))
        obj._d = _int_or_none(d.get("_d"))

        # Dense arrays
        for name in ("BMU", "QE", "AFF", "RFSize",
                     "WL_Dist", "WL_Purity", "CADJ_nhbs_size", "CONN_nhbs_size"):
            setattr(obj, name, d.get(name))

        # WL (object array with possible None entries)
        obj.WL = d.get("WL")

        # WL_unq: type is preserved by _write_label_array / _read_label_array
        # so no casting is needed here
        obj.WL_unq = d.get("WL_unq")

        # Sparse matrices
        obj.CADJ = d.get("CADJ")
        obj.CONN = d.get("CONN")

        # Ragged lists
        obj.RF        = d.get("RF")
        obj.CADJ_nhbs = d.get("CADJ_nhbs")
        obj.CONN_nhbs = d.get("CONN_nhbs")

        return obj


class VQFitter:
    """
    Vector Quantizer Fitter using arbitrary Lp distance metrics.

    Supports multiple fitting algorithms including random sampling,
    IRLS (Iteratively Reweighted Least Squares), gradient descent (GD),
    PAM (Partitioning Around Medoids), and FAISS k-means (p=2 only).

    Typical workflow
    ----------------
    1. Instantiate with the desired number of prototypes M and Lp order p::

           fitter = VQFitter(M=20, p=2, random_state=42)

    2. Fit prototypes to a data matrix X of shape (N, d)::

           fitter.fit(X)          # defaults to k-means for p=2, IRLS otherwise

    3. Run recall analysis to compute BMU assignments, receptive fields,
       connectivity, etc.  Call with no arguments to analyze the training
       data, or pass new X to analyze held-out data::

           fitter.recall()        # training data
           fitter.recall(X_test)  # new data

    4. Inspect results via the embedded VQRecaller object::

           recaller = fitter.recaller
           print(recaller.BMU)       # shape (N, max_bmu)
           print(recaller.QE)        # quantization errors
           print(recaller.RFSize)    # receptive field sizes
           print(recaller.CONN)      # sparse connectivity matrix

    5. Optionally reconstruct data::

           X_hard = recaller.reconstruct(fitter.W, X, method='hard')
           X_soft = recaller.reconstruct(fitter.W, X, method='soft')

    See VQRecaller for definitions of BMU, QE, RF, CADJ, and CONN.
    """
    
    AVAILABLE_METHODS = ["random", "IRLS", "GD", "PAM", "kmeans"]
    
    def __init__(self, M, p=2, max_bmu=2, random_state=None, verbose=True):
        """
        Initialize the VQ Fitter.
        
        Parameters
        ----------
        M : int
            Number of prototypes to create
        p : float, default=2
            Order of the Lp distance metric (e.g., 1 for Manhattan, 2 for Euclidean)
        max_bmu : int, default=2
            Number of Best Matching Units (closest prototypes) to track
        random_state : int or None, optional
            Seed for the instance-local random number generator. Reproducible
            results are guaranteed without touching the global NumPy random
            state, so multiple VQFitter instances with different seeds can
            coexist safely.
        verbose : bool, default=True
            If True, print progress messages during fitting and recall.
            Set to False to suppress all terminal output.
        """
        self.M = int(M)
        self.p = p
        self.max_bmu = max_bmu
        self.random_state = random_state
        self.verbose = verbose
        self.W = None  # Prototype matrix (M, d)

        # Use an instance-local Generator rather than np.random.seed(), which
        # would mutate the global NumPy random state and interfere with other
        # code (other VQFitter instances, user code, etc.).
        self._rng = np.random.default_rng(self.random_state)

        # VQRecaller instance for BMU tracking during fitting
        self.recaller = VQRecaller(p=self.p, max_bmu=self.max_bmu, verbose=self.verbose)
    
    def fit(self, X, method=None, distX=None, **kwargs):
        """
        Fit the vector quantizer using the specified method.
        
        Defaults to "kmeans" when p=2, and "IRLS" otherwise.
        
        Parameters
        ----------
        X : array-like, shape (N, d)
            Training data matrix
        method : str, optional
            Fitting method. One of: ["random", "IRLS", "GD", "PAM", "kmeans"].
            Defaults to "kmeans" if p=2, "IRLS" otherwise.
        distX : array, shape (N, N), optional
            Precomputed distance matrix. If provided, internal distance
            calculation is skipped (PAM only).
        **kwargs : additional arguments
            Method-specific parameters (see individual _fit_* methods).
            
        Returns
        -------
        self : VQFitter
            Returns self for method chaining
        """
        X = np.array(X)
        N, d = X.shape
        
        if self.M > N:
            raise ValueError(f"Cannot create {self.M} prototypes from only {N} observations")
        
        # Choose default method based on p
        if method is None:
            method = "kmeans" if self.p == 2 else "IRLS"

        if method not in self.AVAILABLE_METHODS:
            raise ValueError(f"Unknown fitting method '{method}'. Available methods: {self.AVAILABLE_METHODS}")
        
        if method == "kmeans" and self.p != 2:
            raise ValueError(f"kmeans only supports p=2 (Euclidean distance), got p={self.p}")

        if method == "random":
            self._fit_random(X)
        elif method == "IRLS":
            self._fit_IRLS(X, **kwargs)
        elif method == "GD":
            self._fit_GD(X, **kwargs)
        elif method == "PAM":
            self._fit_PAM(X, distX=distX, **kwargs)
        elif method == "kmeans":
            self._fit_kmeans(X, **kwargs)
        
        if self.verbose:
            print(f"Fitting complete using {method} method.")
        return self
            
    def _fit_random(self, X):
        """
        Fit by randomly sampling prototypes from X.
        
        Parameters
        ----------
        X : array, shape (N, d)
            Training data
        """
        N = X.shape[0]
        indices = self._rng.choice(N, size=self.M, replace=False)
        self.W = X[indices].copy()
        self.recaller.update_BMU(X, self.W)
        if self.verbose:
            print(f"Randomly sampled {self.M} prototypes from {N} observations")
    
    def _fit_IRLS(self, X, tol_RFstab=0.01, tol_MQE=1e-4, max_iter=100, init_method="random"):
        """
        Fit using Iteratively Reweighted Least Squares.
        
        Parameters
        ----------
        X : array, shape (N, d)
            Training data
        tol_RFstab : float, default=0.01
            Tolerance for receptive field stability (proportion of changed assignments)
        tol_MQE : float, default=1e-4
            Tolerance for relative change in Mean Quantization Error
        max_iter : int, default=100
            Maximum number of iterations
        init_method : str, default="random"
            Initialization method:
            - "random": sample M observations from X without replacement
            - "PAM": use FasterPAM medoids as starting prototypes
            - "current": use the existing self.W (must be set via set_prototypes
              or a prior fit call)
        """
        if init_method == "random":
            self._fit_random(X)
        elif init_method == "PAM":
            self._fit_PAM(X)
        elif init_method == "current":
            if self.W is None:
                raise RuntimeError(
                    "init_method='current' requires W to be set first via "
                    "set_prototypes() or a prior fit() call."
                )
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")
        
        self.recaller.update_BMU(X, self.W)
        prevBMU = self.recaller.BMU[:, 0].copy()
        prevMQE = np.mean(self.recaller.QE[:, 0])
        
        if self.verbose:
            print(f"Initial MQE: {prevMQE:.6f}")
        
        for iter_num in range(1, max_iter + 1):
            self._update_prototypes_IRLS(X)
            self.recaller.update_BMU(X, self.W)
            
            current_BMU = self.recaller.BMU[:, 0]
            RFstab = np.mean(current_BMU != prevBMU)
            MQE = np.mean(self.recaller.QE[:, 0])
            rel_MQE_change = (MQE - prevMQE) / prevMQE if prevMQE != 0 else 0
            
            if self.verbose:
                print(f"Iter {iter_num:3d}: RFstab = {RFstab:.4f}, MQE = {MQE:.6f}, rel_MQE_change = {rel_MQE_change:.6f}")
            
            prevBMU = current_BMU.copy()
            prevMQE = MQE
            
            if RFstab < tol_RFstab:
                if self.verbose:
                    print(f"Converged after {iter_num} iterations (RFstab < {tol_RFstab})")
                break
            elif abs(rel_MQE_change) < tol_MQE:
                if self.verbose:
                    print(f"Converged after {iter_num} iterations (|rel_MQE_change| < {tol_MQE})")
                break
        else:
            if self.verbose:
                print(f"Reached maximum iterations ({max_iter}) without convergence")

    def _update_prototypes_IRLS(self, X):
        """
        Update prototype vectors using IRLS for minimizing the sum of Lp norms.
        Uses lp_norm_stable for numerical stability at high p values.
        
        Parameters
        ----------
        X : array, shape (N, d)
            Training data matrix
        """
        for j in range(self.W.shape[0]):
            cluster_mask = (self.recaller.BMU[:, 0] == j)
            cluster_points = X[cluster_mask]
            
            if len(cluster_points) == 0:
                continue
            
            # Add tiny perturbation to avoid singularities at exact prototype locations
            W_perturbed = self.W[j] + self._rng.normal(0, 1e-12, self.W[j].shape)
            diff = cluster_points - W_perturbed
            
            distances = lp_norm_stable(diff, self.p)
            # lp_norm_stable returns shape (n_cluster, 1) due to keepdims=True;
            # the comparison and ** below broadcast correctly over that shape.
            distances[distances < 1e-12] = 1e-12

            weights = distances ** (1 - self.p)
            weights_reshaped = weights.reshape(-1, 1)

            numerator = np.sum(cluster_points * weights_reshaped, axis=0)
            denominator = np.sum(weights)
            
            if denominator == 0:
                continue
                
            self.W[j] = numerator / denominator

    def _fit_GD(self, X, tol_RFstab=0.01, tol_MQE=1e-4, max_iter=100, init_method="random"):
        """
        Fit using gradient descent on the Lp centroid objective.
        
        Parameters
        ----------
        X : array, shape (N, d)
            Training data
        tol_RFstab : float, default=0.01
            Tolerance for receptive field stability (proportion of changed assignments)
        tol_MQE : float, default=1e-4
            Tolerance for relative change in Mean Quantization Error
        max_iter : int, default=100
            Maximum number of iterations
        init_method : str, default="random"
            Initialization method:
            - "random": sample M observations from X without replacement
            - "PAM": use FasterPAM medoids as starting prototypes
            - "current": use the existing self.W (must be set via set_prototypes
              or a prior fit call)
        """
        if init_method == "random":
            self._fit_random(X)
        elif init_method == "PAM":
            self._fit_PAM(X)
        elif init_method == "current":
            if self.W is None:
                raise RuntimeError(
                    "init_method='current' requires W to be set first via "
                    "set_prototypes() or a prior fit() call."
                )
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")
        
        self.recaller.update_BMU(X, self.W)
        prevBMU = self.recaller.BMU[:, 0].copy()
        prevMQE = np.mean(self.recaller.QE[:, 0])
        
        if self.verbose:
            print(f"Initial MQE: {prevMQE:.6f}")
        
        for iter_num in range(1, max_iter + 1):
            self._update_prototypes_GD(X)
            self.recaller.update_BMU(X, self.W)
            
            current_BMU = self.recaller.BMU[:, 0]
            RFstab = np.mean(current_BMU != prevBMU)
            MQE = np.mean(self.recaller.QE[:, 0])
            rel_MQE_change = (MQE - prevMQE) / prevMQE if prevMQE != 0 else 0
            
            if self.verbose:
                print(f"Iter {iter_num:3d}: RFstab = {RFstab:.4f}, MQE = {MQE:.6f}, rel_MQE_change = {rel_MQE_change:.6f}")
            
            prevBMU = current_BMU.copy()
            prevMQE = MQE
            
            if RFstab < tol_RFstab:
                if self.verbose:
                    print(f"Converged after {iter_num} iterations (RFstab < {tol_RFstab})")
                break
            elif abs(rel_MQE_change) < tol_MQE:
                if self.verbose:
                    print(f"Converged after {iter_num} iterations (|rel_MQE_change| < {tol_MQE})")
                break
        else:
            if self.verbose:
                print(f"Reached maximum iterations ({max_iter}) without convergence")

    def _update_prototypes_GD(self, X):
        """
        Update prototype vectors using gradient descent for Lp centroids.
        Uses L-BFGS-B with the analytic gradient. For p=2 shortcuts to the mean.
        
        Parameters
        ----------
        X : array, shape (N, d)
            Training data matrix
        """
        for j in range(self.W.shape[0]):
            cluster_mask = (self.recaller.BMU[:, 0] == j)
            cluster_points = X[cluster_mask]
            
            if len(cluster_points) == 0:
                continue
            
            L2_centroid = np.mean(cluster_points, axis=0)

            if self.p == 2:
                self.W[j] = L2_centroid
            else:
                result = minimize(
                    x0=L2_centroid,
                    fun=lp_centroid_objective_p_root,
                    jac=lp_centroid_gradient_p_root,
                    args=(cluster_points, self.p),
                    method='L-BFGS-B'
                )
                self.W[j] = result.x

    def _fit_PAM(self, X, distX=None, **kwargs):
        """
        Fit using PAM (Partitioning Around Medoids) via FasterPAM.
        
        Parameters
        ----------
        X : array, shape (N, d)
            Training data
        distX : array, shape (N, N), optional
            Precomputed distance matrix. If provided, internal distance
            calculation is skipped.
        **kwargs : keyword arguments
            Passed to kmedoids.fasterpam (e.g. max_iter, init, n_cpu).
            random_state is always taken from the fitter and cannot be overridden.
        """
        try:
            from kmedoids import fasterpam
        except ImportError:
            raise ImportError("kmedoids package required for PAM. Install with: pip install kmedoids")
        
        N = X.shape[0]
        
        if distX is None:
            if self.verbose:
                print(f"Computing {N}x{N} Lp distance matrix for PAM...")
            distX = squareform(pdist(X, metric='minkowski', p=self.p))
        
        if self.verbose:
            print(f"Running FasterPAM with {self.M} medoids...")
        
        kwargs.pop('random_state', None)
        result = fasterpam(diss=distX, medoids=self.M, random_state=self.random_state, **kwargs)
        
        self.W = X[result.medoids].copy()
        self.recaller.update_BMU(X, self.W)
        
        if self.verbose:
            print(f"PAM fitting complete. Final MQE: {np.mean(self.recaller.QE[:, 0]):.6f}")

    def _fit_kmeans(self, X, niter=20, nredo=1, init_method="faiss", **kwargs):
        """
        Fit using FAISS k-means (p=2 only).
        
        Parameters
        ----------
        X : array, shape (N, d)
            Training data
        niter : int, default=20
            Number of k-means iterations per run.
        nredo : int, default=1
            Number of independent runs; the best result (lowest objective) is
            kept. Increase to compensate for random initialization, at the cost
            of nredo x the compute time. Ignored when init_method="current".
        init_method : str, default="faiss"
            Initialization method:
            - "faiss": FAISS random initialization (controlled by seed/nredo)
            - "current": use the existing self.W as starting centroids (must be
              set via set_prototypes() or a prior fit() call). nredo is
              overridden to 1 since restarting from a fixed point is not useful.
        **kwargs : keyword arguments
            Additional parameters passed to faiss.Kmeans constructor
            (e.g. verbose, gpu, max_points_per_centroid).
        """
        if init_method not in ("faiss", "current"):
            raise ValueError(f"init_method must be 'faiss' or 'current', got {init_method!r}")
        if init_method == "current" and self.W is None:
            raise RuntimeError(
                "init_method='current' requires W to be set first via "
                "set_prototypes() or a prior fit() call."
            )
        if not _FAISS_AVAILABLE:
            raise ImportError(
                "FAISS is required for k-means fitting but is not installed. "
                "Install it with: pip install faiss-cpu"
            )
        faiss = _faiss_module  # local alias for readability

        d = X.shape[1]
        X_f32 = np.ascontiguousarray(X, dtype=np.float32)
        seed = self.random_state if self.random_state is not None else 1234

        # nredo is meaningless when starting from a fixed W
        effective_nredo = 1 if init_method == "current" else nredo

        kmeans = faiss.Kmeans(
            d, self.M,
            niter=niter,
            nredo=effective_nredo,
            seed=seed,
            **kwargs
        )

        if init_method == "current":
            init_centroids = np.ascontiguousarray(self.W, dtype=np.float32)
            if self.verbose:
                print(f"Running FAISS k-means with {self.M} clusters (niter={niter}, init_method='current')...")
            kmeans.train(X_f32, init_centroids=init_centroids)
        else:
            if self.verbose:
                print(f"Running FAISS k-means with {self.M} clusters (niter={niter}, nredo={effective_nredo})...")
            kmeans.train(X_f32)

        self.W = kmeans.centroids.copy().astype(X.dtype)
        self.recaller.update_BMU(X, self.W)

        if self.verbose:
            print(f"K-means complete. Final MQE: {np.mean(self.recaller.QE[:, 0]):.6f}")

    def set_prototypes(self, W, X):
        """
        Set the prototype matrix W directly, bypassing fitting.

        Useful when prototypes are known in advance (e.g. from a previous run,
        an external algorithm, or domain-specific initialization) and you want
        to use the VQFitter machinery for recall analysis or as a warm start
        for further fitting.

        BMU and QE are computed immediately from X, keeping the object in a
        consistent state — the same guarantee provided by all _fit_* methods.

        Parameters
        ----------
        W : array-like, shape (M, d)
            Prototype matrix. Must have exactly self.M rows. If the data
            dimensionality d is already known (from a prior fit or recall),
            W must match it; otherwise d is inferred from W.
        X : array-like, shape (N, d)
            Data matrix used to compute BMU and QE.

        Returns
        -------
        self : VQFitter
            Returns self for method chaining
        """
        W = np.array(W)
        X = np.array(X)

        if W.shape[0] != self.M:
            raise ValueError(
                f"W has {W.shape[0]} prototypes but this fitter expects M={self.M}"
            )

        # Validate d if already known, otherwise infer from W
        known_d = self.W.shape[1] if self.W is not None else None
        if known_d is not None and W.shape[1] != known_d:
            raise ValueError(
                f"W has {W.shape[1]} features but prior data had d={known_d}"
            )
        if X.shape[1] != W.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features but W has {W.shape[1]} features"
            )

        self.W = W.copy()
        self.recaller.update_BMU(X, self.W)
        if self.verbose:
            print(f"Prototypes set externally. MQE: {np.mean(self.recaller.QE[:, 0]):.6f}")
        return self

    def recall(self, X=None, labels=None):
        """
        Perform recall analysis using the fitted prototypes.

        This is a convenience wrapper around VQRecaller.recall().
        The key difference from calling self.recaller.recall() directly is
        that W is always taken from self.W (the fitted prototypes), so you
        never need to pass it explicitly here.

        If X is provided, performs recall analysis on new (e.g. held-out) data.
        If X is None, finalizes recall analysis on the training data (BMU was
        already updated incrementally during fitting).

        Parameters
        ----------
        X : array-like, shape (N, d), optional
            Data matrix for recall analysis. If None, finalizes training recall.
        labels : array-like, shape (N,), optional
            Observation labels. Any hashable type is accepted (int, str, float,
            etc.). If provided, recall_labels() is called automatically after
            the main recall pipeline to compute WL, WL_Dist, and WL_Purity.

        Returns
        -------
        self : VQFitter
            Returns self for method chaining.

        See Also
        --------
        VQRecaller.recall : The underlying recall method, which also accepts
            explicit W and p arguments for use outside of VQFitter.
        """
        if self.W is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        if X is not None:
            X = np.array(X)
            self.recaller.recall(X, self.W, labels=labels)
        else:
            self.recaller.recall(labels=labels)
        
        return self
    
    def save(self, path):
        """
        Save the VQFitter (and its embedded VQRecaller) to an HDF5 file.

        Parameters
        ----------
        path : str
            Destination file path.  Created or overwritten.

        See Also
        --------
        VQFitter.load   : Reconstruct a VQFitter from a saved file.
        VQRecaller.save : Save a standalone VQRecaller.

        Examples
        --------
        >>> fitter.save("fitter.h5")
        >>> fitter2 = VQFitter.load("fitter.h5")
        """
        from .io import VQFitterIO
        VQFitterIO(self).save(path)

    @classmethod
    def load(cls, path):
        """
        Load a VQFitter (and its embedded VQRecaller) from an HDF5 file
        created by :meth:`save`.

        Parameters
        ----------
        path : str
            Path to the HDF5 file.

        Returns
        -------
        VQFitter

        Examples
        --------
        >>> fitter = VQFitter.load("fitter.h5")
        >>> fitter.W              # prototype matrix
        >>> fitter.recaller.BMU   # recall products, if recall() was called before saving
        """
        from .io import VQFitterIO
        return VQFitterIO.load(path)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self):
        """
        Return a plain dict representation of this VQFitter.

        Only hyperparameters and the prototype matrix W are included.
        The embedded recaller is serialized separately by VQFitterIO.
        The RNG (``_rng``) is not stored; on load it is re-initialized
        from ``random_state``.

        Returns
        -------
        dict with keys: M, p, max_bmu, verbose, random_state, W
        """
        return {
            "M":            self.M,
            "p":            self.p,
            "max_bmu":      self.max_bmu,
            "verbose":      self.verbose,
            "random_state": self.random_state,
            "W":            self.W,
        }

    @classmethod
    def from_dict(cls, d):
        """
        Reconstruct a VQFitter from a dict produced by :meth:`to_dict`.

        Bypasses ``__init__`` so that hyperparameters are restored directly.
        The RNG is re-initialized from ``random_state``.

        Note: ``recaller`` is set to ``None`` by this method.
        :meth:`VQFitterIO.load` attaches the reconstructed VQRecaller after
        calling ``from_dict``.  If you call ``from_dict`` directly, you are
        responsible for setting ``obj.recaller`` before calling any method
        that depends on it.

        Parameters
        ----------
        d : dict
            As returned by :meth:`to_dict`.

        Returns
        -------
        VQFitter
        """
        obj              = cls.__new__(cls)
        obj.M            = int(d["M"])
        obj.p            = float(d["p"])
        obj.max_bmu      = int(d["max_bmu"])
        obj.verbose      = bool(d["verbose"])
        obj.random_state = d.get("random_state")
        obj.W            = d.get("W")
        obj._rng         = np.random.default_rng(obj.random_state)
        obj.recaller     = None   # attached by VQFitterIO.load()
        return obj

    def get_summary(self):
        """
        Get a summary of the fitted model and recall analysis.

        Returns
        -------
        dict
            Summary statistics.  Keys always present:
              - ``"fitted"`` : bool — whether prototypes have been computed
              - ``"M"`` : int — number of prototypes
              - ``"d"`` : int or None — feature dimensionality (None if not yet fitted)
              - ``"p_norm"`` : float — Lp order
              - ``"max_bmu"`` : int — number of BMUs tracked
              - ``"random_state"`` : int or None — seed used

            If recall() has been called, all keys from VQRecaller.get_summary()
            are also included (MQE, RF sizes, connectivity stats, etc.).
        """
        summary = {
            "fitted": self.W is not None,
            "M": self.M,
            "d": self.W.shape[1] if self.W is not None else None,
            "p_norm": self.p,
            "max_bmu": self.max_bmu,
            "random_state": self.random_state
        }
        
        if self.recaller.BMU is not None:
            recall_summary = self.recaller.get_summary()
            summary.update(recall_summary)
        
        return summary
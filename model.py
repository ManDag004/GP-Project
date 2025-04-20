import numpy as np


class OnlineSparseGP:
    """
    Online Sparse Gaussian Process with support for efficient local unlearning.
    """
    def __init__(self, kernel, noise_var=1e-3, max_points=None, jitter=1e-9):
        """
        :param kernel: Kernel function (callable) for covariance (can be SumKernel for combined RBF+Periodic).
        :param noise_var: Observation noise variance.
        :param max_points: Optional maximum number of inducing points to retain.
        :param jitter: Small jitter added for numerical stability.
        """
        self.kernel = kernel
        self.noise_var = noise_var
        self.jitter = jitter
        self.max_points = max_points
        # Inducing point: inputs and outputs
        self.Z = np.empty((0, 0))
        self.Y = np.empty(0)
        # Kernel matrix (without noise) and its inverse (with noise+jitter) for inducing points
        self.K = np.zeros((0, 0))
        self.K_inv = np.zeros((0, 0))
    
    def initialize(self, X_init, Y_init):
        """Initialize the GP with an initial set of inducing points and outputs."""
        X_init = np.atleast_2d(X_init)
        Y_init = np.asarray(Y_init).flatten()
        assert X_init.shape[0] == len(Y_init), "Mismatch in number of points and outputs."
        self.Z = X_init.copy()
        self.Y = Y_init.copy()
        # Compute initial kernel matrix and its inverse
        K = self.kernel(X_init)    # K_mm (without noise)
        Kmm = K + np.eye(len(X_init)) * (self.noise_var + self.jitter)  # add noise variance and jitter
        self.K_inv = np.linalg.inv(Kmm)
        self.K_inv = 0.5 * (self.K_inv + self.K_inv.T)      # enforce symmetry
        self.K = K.copy()
        # If too many points provided and max_points is set, remove oldest extras
        if self.max_points is not None and self.Z.shape[0] > self.max_points:
            excess = self.Z.shape[0] - self.max_points
            for _ in range(excess):
                self._remove_index(0)
    
    def _remove_index(self, idx):
        """Internal helper to remove the inducing point at index idx, updating K_inv via fast downdate (Eq. 3)."""
        M = self.Z.shape[0]
        if M == 0:
            return
        # Remove point from inducing inputs and outputs
        self.Z = np.delete(self.Z, idx, axis=0)
        self.Y = np.delete(self.Y, idx, axis=0)
        # Remove corresponding rows and columns from kernel matrix
        self.K = np.delete(np.delete(self.K, idx, axis=0), idx, axis=1)
        if M == 1:
            # No points left
            self.K_inv = np.zeros((0, 0))
            return
        # Update the inverse kernel matrix using the rank-1 downdate formula (Equation 3)
        inv_old = self.K_inv
        # If removing a non-last index, permute it to the last position for convenience
        if idx != M - 1:
            perm = list(range(M))
            perm[idx], perm[-1] = perm[-1], perm[idx]
            inv_old = inv_old[np.ix_(perm, perm)]
        # Partition the inverse matrix (before removal) as per Equation 3
        inv_11 = inv_old[:M-1, :M-1]
        inv_12 = inv_old[:M-1, M-1].reshape(-1, 1)
        inv_21 = inv_old[M-1, :M-1].reshape(1, -1)
        inv_22 = inv_old[M-1, M-1]  # scalar
        # Compute the new inverse after removal (fast downdate)
        inv_new = inv_11 - (inv_12.dot(inv_21)) / inv_22
        inv_new = 0.5 * (inv_new + inv_new.T)  # enforce symmetry
        # If we permuted, undo permutation on the result
        if idx != M - 1:
            perm_remaining = perm[:-1]
            inv_back = np.argsort(perm_remaining)
            inv_new = inv_new[np.ix_(inv_back, inv_back)]
            # Also re-order self.K to match (though self.K is now smaller)
            self.K = self.K[np.ix_(inv_back, inv_back)]
        self.K_inv = inv_new
    
    def add_point(self, x_new, y_new):
      """Add a new inducing point, update K_inv via Sherman-Morrison,
      and prune the least informative point if over budget."""
      x_new = np.atleast_2d(x_new)
      y_new = float(y_new)

      if self.Z.shape[0] == 0:
          self.Z = x_new.copy()
          self.Y = np.array([y_new])
          k_nn = self.kernel(x_new, x_new)[0, 0]
          self.K = np.array([[k_nn]])
          Kmm = k_nn + self.noise_var + self.jitter
          self.K_inv = np.array([[1.0 / Kmm]])
          return 0

      M = self.Z.shape[0]
      # covariance between new point and existing inducing points
      k_vec = self.kernel(self.Z, x_new).reshape(-1)
      k_nn  = self.kernel(x_new, x_new)[0, 0]

      # append new data
      self.Z = np.vstack([self.Z, x_new])
      self.Y = np.append(self.Y, y_new)

      # extend K
      K_ext = np.zeros((M+1, M+1))
      K_ext[:M, :M] = self.K
      K_ext[:M, M] = k_vec
      K_ext[M, :M] = k_vec
      K_ext[M, M]  = k_nn
      self.K = K_ext

      # Sherman-Morrison rank‑1 update of K_inv
      inv_old = self.K_inv
      a = k_nn + self.noise_var + self.jitter
      v = inv_old.dot(k_vec)
      gamma = a - k_vec.dot(v)
      if gamma <= 1e-12:
          gamma = 1e-12  # enforce positivity :contentReference[oaicite:0]{index=0}

      inv_new = np.zeros((M+1, M+1))
      inv_new[:M, :M] = inv_old + np.outer(v, v) / gamma
      inv_new[:M, M] = -v / gamma
      inv_new[M, :M] = -v.T / gamma
      inv_new[M, M]  = 1.0 / gamma
      # symmetrize to counter round‐off
      self.K_inv = 0.5 * (inv_new + inv_new.T)  # O(m^2)

      # Prune the least informative inducing point
      if self.max_points is not None and self.Z.shape[0] > self.max_points:
          # posterior weights α = K_inv
          alpha = self.K_inv.dot(self.Y)            # O(m^2) compute
          # influence scores e_i = a_i^2 / Q_ii
          diag_Q = np.diag(self.K_inv)
          scores = alpha**2 / diag_Q                # O(m)
          # remove index with smallest score
          idx_remove = int(np.argmin(scores))
          self._remove_index(idx_remove)

      return self.Z.shape[0] - 1

    
    def predict(self, X_test):
        """Predictive mean and variance for given test inputs X_test."""
        X_test = np.atleast_2d(X_test)
        if self.Z.shape[0] == 0:
            # If no inducing points, return prior (mean 0, variance = k(x,x))
            K_tt = self.kernel(X_test)
            mean = np.zeros(X_test.shape[0])
            var = np.diag(K_tt)
            return mean, var
        # Compute cross-covariances
        K_xZ = self.kernel(X_test, self.Z)            # shape (N_test, M)
        # Posterior mean: K_xZ * (K_ZZ + sigma^2 I)^{-1} * Y
        mean = K_xZ.dot(self.K_inv).dot(self.Y)
        # Posterior variance: k(x,x) - K_xZ * (K_ZZ + sigma^2 I)^{-1} * K_Zx
        K_Zx = K_xZ.T                                 # shape (M, N_test)
        cov_term = K_xZ.dot(self.K_inv).dot(K_Zx)     # predictive covariance between test points
        var = np.copy(np.diag(self.kernel(X_test) - cov_term))
        var[var < 0] = 0.0  # numerical safety: eliminate small negative variances
        return mean, var
    
    def unlearn(self, U_points, gamma=0.5):
        """
        Efficient local unlearning: remove inducing points to maximize variance at U_points.
        :param U_points: array-like of shape (P, d) specifying locations of detected changes.
        :param gamma: stopping criterion factor (fraction of initial max gain).
        :return: list of removed inducing point coordinates.
        """
        U_points = np.atleast_2d(U_points)
        removed_points = []
        M = self.Z.shape[0]
        if M == 0:
            return removed_points  # nothing to unlearn
        P = U_points.shape[0]
        # Compute current predictive variance at each u ∈ U with all inducing points present
        K_UZ = self.kernel(U_points, self.Z)                 # shape (P, M)
        K_UU = self.kernel(U_points)                        # shape (P, P) (we will use only its diagonal)
        var_current = np.zeros(P)
        for j in range(P):
            k_uZ = K_UZ[j]                                   # covariance vector between u_j and all inducing points
            var_current[j] = K_UU[j, j] - k_uZ.dot(self.K_inv).dot(k_uZ.T)
            if var_current[j] < 0:
                var_current[j] = 0.0
        # Pre-compute initial variance *gains* for removing each candidate (Equation 2 effect)
        initial_gains = np.full(M, -np.inf)
        for i in range(M):
            # Compute variance at U if point i were removed (using Equation 2)
            inv_old = self.K_inv
            # Compute the inverse downdate for candidate i (without permanently removing it)
            if M > 1:
                if i != M - 1:
                    perm = list(range(M)); perm[i], perm[-1] = perm[-1], perm[i]
                    inv_perm = inv_old[np.ix_(perm, perm)]
                else:
                    inv_perm = inv_old
                inv11 = inv_perm[:M-1, :M-1]
                inv12 = inv_perm[:M-1, M-1].reshape(-1, 1)
                inv21 = inv_perm[M-1, :M-1].reshape(1, -1)
                inv22 = inv_perm[M-1, M-1]
                inv_candidate = inv11 - (inv12.dot(inv21)) / inv22   # (K_inv after removing i)
                inv_candidate = 0.5 * (inv_candidate + inv_candidate.T)
                if i != M - 1:
                    # Unpermute inverse candidate matrix
                    perm_rem = perm[:-1]
                    inv_back = np.argsort(perm_rem)
                    inv_candidate = inv_candidate[np.ix_(inv_back, inv_back)]
            else:
                inv_candidate = np.zeros((0, 0))
            # Compute new variances at U with point i removed
            gain = 0.0
            for j in range(P):
                if M > 1:
                    k_uZ = K_UZ[j]                             # original covariance with all points
                    k_uZ_i_removed = np.delete(k_uZ, i)        # remove the i-th element
                    # Variance with point i removed: k(u,u) - k_{u,Z{i}} * inv_candidate * k_{Z{i},u}
                    v = inv_candidate.dot(k_uZ_i_removed.T)
                    var_no_i = K_UU[j, j] - k_uZ_i_removed.dot(v)
                else:
                    # If only one point in model (i), removing it means prior variance
                    var_no_i = K_UU[j, j]
                if var_no_i < 0: 
                    var_no_i = 0.0
                gain += (var_no_i - var_current[j])            # total increase in variance at all U
            initial_gains[i] = gain
        gD_max = np.max(initial_gains)  # maximum initial gain
        print(f"[unlearn] triggered at {U_points.tolist()} | initial max gain = {gD_max:.4f}")
        # Greedily remove points until gain falls below gamma * initial_gain
        cur_Z = self.Z.copy()
        cur_K_inv = self.K_inv.copy()
        cur_var_U = var_current.copy()
        cur_M = M
        while cur_M > 0:
            # Compute variance gain for each remaining candidate in current model
            gains = np.full(cur_M, -np.inf)
            for i in range(cur_M):
                # Compute effect of removing candidate i (similar to above, using current state)
                if cur_M > 1:
                    inv_old = cur_K_inv
                    if i != cur_M - 1:
                        perm = list(range(cur_M)); perm[i], perm[-1] = perm[-1], perm[i]
                        inv_perm = inv_old[np.ix_(perm, perm)]
                    else:
                        inv_perm = inv_old
                    inv11 = inv_perm[:cur_M-1, :cur_M-1]
                    inv12 = inv_perm[:cur_M-1, cur_M-1].reshape(-1, 1)
                    inv21 = inv_perm[cur_M-1, :cur_M-1].reshape(1, -1)
                    inv22 = inv_perm[cur_M-1, cur_M-1]
                    inv_candidate = inv11 - inv12.dot(inv21) / inv22
                    inv_candidate = 0.5 * (inv_candidate + inv_candidate.T)
                    if i != cur_M - 1:
                        perm_rem = perm[:-1]
                        inv_back = np.argsort(perm_rem)
                        inv_candidate = inv_candidate[np.ix_(inv_back, inv_back)]
                else:
                    inv_candidate = np.zeros((0, 0))
                # Calculate total variance gain at U if removing i
                gain = 0.0
                for j in range(P):
                    if cur_M > 1:
                        k_uZ = self.kernel(U_points[j:j+1], cur_Z).reshape(-1)
                        k_uZ_rem = np.delete(k_uZ, i)
                        v = inv_candidate.dot(k_uZ_rem.T)
                        var_no_i = K_UU[j, j] - k_uZ_rem.dot(v)
                    else:
                        var_no_i = K_UU[j, j]
                    if var_no_i < 0:
                        var_no_i = 0.0
                    gain += (var_no_i - cur_var_U[j])
                gains[i] = gain
            idx_best = int(np.argmax(gains))
            max_gain = gains[idx_best]
            # Check stopping criterion (greedy gain vs. fraction of initial gain)
            if max_gain <= gamma * gD_max or max_gain <= 1e-12:
                print(f"[unlearn] stop: max_gain={max_gain:.4f} ≤ gamma·gD_max")
                break
            # Remove the selected point from the current model state
            print(f"[unlearn] removing Z[{idx_best}] = {cur_Z[idx_best].tolist()} with gain {max_gain:.4f}")
            removed_points.append(cur_Z[idx_best].tolist())
            # Update current inverse and variance for next iteration (apply Equation 3 update)
            if cur_M > 1:
                # Permute selected index to end for removal update
                inv_old = cur_K_inv
                if idx_best != cur_M - 1:
                    perm = list(range(cur_M)); perm[idx_best], perm[-1] = perm[-1], perm[idx_best]
                    inv_old = inv_old[np.ix_(perm, perm)]
                    cur_Z = cur_Z[perm]
                inv11 = inv_old[:cur_M-1, :cur_M-1]
                inv12 = inv_old[:cur_M-1, cur_M-1].reshape(-1, 1)
                inv21 = inv_old[cur_M-1, :cur_M-1].reshape(1, -1)
                inv22 = inv_old[cur_M-1, cur_M-1]
                cur_K_inv = inv11 - inv12.dot(inv21) / inv22
                cur_K_inv = 0.5 * (cur_K_inv + cur_K_inv.T)
                cur_Z = np.delete(cur_Z, cur_M-1, axis=0)
                cur_M -= 1
                # If we had permuted, unpermute remaining points to original order
                if idx_best != cur_M:
                    perm_rem = perm[:-1]
                    inv_back = np.argsort(perm_rem)
                    cur_Z = cur_Z[inv_back]
                    cur_K_inv = cur_K_inv[np.ix_(inv_back, inv_back)]
            else:
                # Removed the last point
                cur_Z = np.empty((0, cur_Z.shape[1])); cur_K_inv = np.zeros((0, 0)); cur_M = 0
            # Update current variance at U after removal of the point
            cur_var_U = np.zeros(P)
            for j in range(P):
                if cur_M > 0:
                    k_uZ = self.kernel(U_points[j:j+1], cur_Z).reshape(-1)
                    cur_var_U[j] = K_UU[j, j] - k_uZ.dot(cur_K_inv).dot(k_uZ.T)
                    if cur_var_U[j] < 0:
                        cur_var_U[j] = 0.0
                else:
                    cur_var_U[j] = K_UU[j, j]
        # Apply the removals to the actual model (remove points from self.Z/self.Y)
        for pt in removed_points:
            # Find matching point in current inducing set and remove it
            if self.Z.shape[0] == 0:
                break
            # Compare coordinates (assuming exact match in float; use allclose for safety)
            for idx, z in enumerate(self.Z):
                if np.allclose(z, pt, atol=1e-8):
                    self._remove_index(idx)
                    break
        print(f"[unlearn] finished; total removed = {removed_points}")
        return removed_points
    
    def update_stream(self, x_new, y_new,
                      k=2.5,         # ≈95 % 2‑sided interval
                      gamma=0.002):    # Ziomek et al. stopping rule
        """
        Complete OSGPU step = drift‑test → (optional) unlearn → online update.
        Returns (drift_flag, standardized_residual).
        """
        # 1) prior prediction
        mean, var = self.predict(x_new)
        sd = float(np.sqrt(var + self.noise_var))  # account for observation noise
        if sd < 1e-12:              # numerical guard
            sd = 1e-6
        resid = float((y_new - mean[0]) / sd)

        # 2) drift test  (Gómez‑Verdejo 2023; Chandola & Vatsavai 2012)
        drift = abs(resid) > k
        if drift:
            # single‑point unlearning as in Ziomek 2024
            self.unlearn(U_points=x_new.reshape(1, -1), gamma=gamma)

        # 3) normal online update (may add or prune an inducing pt)
        self.add_point(x_new, y_new)
        return mean, var, drift, resid
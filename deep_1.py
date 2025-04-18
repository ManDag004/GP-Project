import os
import requests
import zipfile
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.cluster import KMeans
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

class RBFKernel:
    """Radial Basis Function (Gaussian) Kernel."""
    def __init__(self, lengthscale=1.0, variance=1.0):
        self.lengthscale = lengthscale
        self.variance = variance
    def __call__(self, X, Y=None):
        X = np.atleast_2d(X)
        Y = X if Y is None else np.atleast_2d(Y)
        # Squared Euclidean distance between points
        diff = X[:, None, :] - Y[None, :, :]
        sqdist = np.sum(diff**2, axis=-1)
        # RBF covariance
        return self.variance * np.exp(-0.5 * sqdist / (self.lengthscale**2))

class PeriodicKernel:
    """Periodic Kernel."""
    def __init__(self, lengthscale=1.0, variance=1.0, period=1.0):
        self.lengthscale = lengthscale
        self.variance = variance
        self.period = period
    def __call__(self, X, Y=None):
        X = np.atleast_2d(X)
        Y = X if Y is None else np.atleast_2d(Y)
        # Euclidean distance
        diff = X[:, None, :] - Y[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        # Periodic covariance: exp(-2 sin^2(pi * dist / period) / lengthscale^2)
        sine_arg = np.pi * dist / self.period
        sin_term = np.sin(sine_arg)
        return self.variance * np.exp(-2 * (sin_term**2) / (self.lengthscale**2))

class SumKernel:
    """Combined Kernel as sum of multiple kernel components (e.g., RBF + Periodic)."""
    def __init__(self, *kernels):
        self.kernels = kernels
    def __call__(self, X, Y=None):
        # Sum the covariance matrices from each component kernel
        K_total = 0
        for kern in self.kernels:
            K_comp = kern(X, Y)
            K_total = K_comp if isinstance(K_total, int) else K_total + K_comp
        return K_total


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
        # Inducing point inputs and outputs
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
        K = self.kernel(X_init)                             # K_mm (without noise)
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
      """Add a new inducing point, update K_inv via Sherman–Morrison,
      and prune the least informative point if over budget."""
      x_new = np.atleast_2d(x_new)
      y_new = float(y_new)

      # ——— Initialization for the first point ———
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

      # ——— Sherman–Morrison rank‑1 update of K_inv ———
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
      self.K_inv = 0.5 * (inv_new + inv_new.T)  # O(m²) update :contentReference[oaicite:1]{index=1}

      # ——— Prune the least informative inducing point ———
      if self.max_points is not None and self.Z.shape[0] > self.max_points:
          # posterior weights α = K_inv · Y :contentReference[oaicite:2]{index=2}
          alpha = self.K_inv.dot(self.Y)            # O(m²) compute
          # influence scores ε_i = α_i² / Q_ii
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
        # Posterior mean: K_xZ * (K_ZZ + σ^2 I)^{-1} * Y
        mean = K_xZ.dot(self.K_inv).dot(self.Y)
        # Posterior variance: k(x,x) - K_xZ * (K_ZZ + σ^2 I)^{-1} * K_Zx
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
                    # Variance with point i removed: k(u,u) - k_{u,Z\{i\}} * inv_candidate * k_{Z\{i\},u}
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
                break
            # Remove the selected point from the current model state
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
        return removed_points


# -------------------------------------------------------------------
# 2) Data download & processing
# -------------------------------------------------------------------
def download_and_extract(data_dir='data'):
    os.makedirs(data_dir, exist_ok=True)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"
    zip_path = os.path.join(data_dir, "LD2011_2014.txt.zip")
    txt_path = os.path.join(data_dir, "LD2011_2014.txt")
    if not os.path.exists(txt_path):
        print("Downloading dataset…")
        r = requests.get(url)
        with open(zip_path,"wb") as f: f.write(r.content)
        print("Extracting…")
        with zipfile.ZipFile(zip_path,"r") as z: z.extractall(data_dir)
    return txt_path

def load_daily_aggregate(txt_path, data_dir='data', force_reload=False):
    # Define the path where the processed dataframe will be saved
    processed_path = os.path.join(data_dir, "daily_aggregate.pkl")
    
    # Check if the processed file exists and we're not forcing a reload
    if os.path.exists(processed_path) and not force_reload:
        print("Loading pre-processed data...")
        # Load the pre-processed data
        daily = pd.read_pickle(processed_path)
        return daily
    
    # If we need to process the data
    print("Processing raw data...")
    df = pd.read_csv(txt_path, sep=';', index_col=0, parse_dates=True, low_memory=False)
    df = df.replace(',', '.', regex=True).astype(float)
    df['total'] = df.sum(axis=1)
    daily = df[['total']].resample('D').sum()
    
    # Save the processed data
    print("Saving processed data for future use...")
    daily.to_pickle(processed_path)
    
    return daily

def process_electricity_data(txt_path):
    df = pd.read_csv(txt_path, sep=';', index_col=0, parse_dates=True, low_memory=False)
    df = df.replace(',', '.', regex=True).astype(float)
    df['aggregate'] = df.sum(axis=1)
    df = df[['aggregate']]
    
    # Aggregate to daily frequency
    df = df.resample('D').sum()
    return df


# -------------------------------------------------------------------
# 3) Main example: train/test loop + visualizations
# -------------------------------------------------------------------
def main():
    # Download & load
    # txt = download_and_extract()
    daily = load_daily_aggregate('data/electricityloaddiagrams20112014/LD2011_2014.txt')
    daily = daily.loc['2012-01-01':'2014-12-31']
    daily['t_day'] = (daily.index - daily.index[0]).days

    # train/test split
    train = daily.loc[:'2013-12-31']
    test  = daily.loc['2014-01-01':]

    X_train = train['t_day'].values.reshape(-1,1)
    y_train = train['total'].values
    X_test  = test ['t_day'].values.reshape(-1,1)
    y_test  = test ['total'].values

    # normalize targets
    ym, ys = y_train.mean(), y_train.std()
    ytr_n = (y_train - ym)/ys
    yte_n = (y_test  - ym)/ys

    # set up kernels
    # rbf = RBFKernel(lengthscale=4., variance=1.)
    # per1 = PeriodicKernel(period=365., lengthscale=50., variance=0.5)
    # per2 = PeriodicKernel(period=30.5,lengthscale=10., variance=0.5)
    # per3 = PeriodicKernel(period=7.,   lengthscale=5.,  variance=0.5)
    rbf = RBFKernel(lengthscale=4.948961837424606, variance=0.034725889070655436)
    per1 = PeriodicKernel(period=365., lengthscale=72.3039510055688, variance=1.1367173558614303)
    per2 = PeriodicKernel(period=30.5,lengthscale=24.486552583062952, variance=0.03460830317302234)
    per3 = PeriodicKernel(period=7.,   lengthscale=2.9979970724403815,  variance= 0.031133591413764025)
    Ksum = SumKernel(rbf, per1, per2, per3)

    # initialize model
    Z0 = X_train[:20]
    gp  = OnlineSparseGP(Ksum, noise_var=0.002322246118810783, max_points=125)
    gp.initialize(Z0, ytr_n[:20])

    # streaming training (one-step ahead)
    preds_tr, vars_tr, t_tr = [], [], []
    for i,(x,y) in enumerate(zip(X_train, ytr_n)):
        μ, σ2 = gp.predict(x)
        preds_tr.append(μ[0]); vars_tr.append(σ2[0]); t_tr.append(x[0])
        gp.add_point(x, y)
        # if i>=500: break   # limit for plotting

    # streaming test
    preds_te, vars_te, t_te = [], [], []
    for i,(x,y) in enumerate(zip(X_test, yte_n)):
        μ, σ2 = gp.predict(x)
        preds_te.append(μ[0]); vars_te.append(σ2[0]); t_te.append(x[0])
        gp.add_point(x, y)
        # if i>=200: break

    # denormalize
    p_tr = np.array(preds_tr)*ys + ym
    s_tr = np.sqrt(np.array(vars_tr))*ys
    a_tr = y_train[:len(p_tr)]
    p_te = np.array(preds_te)*ys + ym
    s_te = np.sqrt(np.array(vars_te))*ys
    a_te = y_test[:len(p_te)]
    Zx  = gp.Z.flatten()

    # plot training
    plt.figure(figsize=(12,6))
    plt.plot(t_tr, a_tr,    'b',  label="Actual train")
    plt.plot(t_tr, p_tr,   'r--',label="Pred. train")
    plt.fill_between(t_tr, p_tr-1.96*s_tr, p_tr+1.96*s_tr,
                     color='r',alpha=0.2,label="95% CI")
    plt.scatter(Zx, np.interp(Zx, t_tr, p_tr),
                marker='x',s=80,color='k',label="Inducing")
    plt.title("Training: 1‑step ahead"); plt.xlabel("Days"); plt.ylabel("Load")
    plt.legend(); plt.grid(True)

    # plot testing
    plt.figure(figsize=(12,6))
    plt.plot(t_te, a_te,    'b',  label="Actual test")
    plt.plot(t_te, p_te,   'r--',label="Pred. test")
    plt.fill_between(t_te, p_te-1.96*s_te, p_te+1.96*s_te,
                     color='r',alpha=0.2,label="95% CI")
    plt.scatter(Zx, np.interp(Zx, t_te, p_te),
                marker='x',s=80,color='k',label="Inducing")
    plt.title("Testing: 1‑step ahead"); plt.xlabel("Days"); plt.ylabel("Load")
    plt.legend(); plt.grid(True)

    plt.show()


if __name__=="__main__":
    main()

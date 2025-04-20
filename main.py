import os
import requests
import zipfile
import io
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.cluster import KMeans
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.interpolate import make_interp_spline

from kernels import RBFKernel, PeriodicKernel, SumKernel
from model import OnlineSparseGP



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
    processed_path = os.path.join(data_dir, "daily_aggregate.pkl")
    
    if os.path.exists(processed_path) and not force_reload:
        print("Loading pre-processed data")
        daily = pd.read_pickle(processed_path)
        return daily
    
    print("Processing raw data")
    df = pd.read_csv(txt_path, sep=';', index_col=0, parse_dates=True, low_memory=False)
    df = df.replace(',', '.', regex=True).astype(float)
    df['total'] = df.sum(axis=1)
    daily = df[['total']].resample('D').sum()
    
    print("Saving processed data for future use")
    daily.to_pickle(processed_path)
    
    return daily


def main():
    txt = download_and_extract()
    daily = load_daily_aggregate(txt)
    daily = daily.loc['2011-11-01':'2014-12-31']
    daily['t_day'] = (daily.index - daily.index[0]).days

    # variable names should not be called train or be reated to it, need to fix that, but the code will remain the way it is
    train = daily.loc[:]

    X_train = train['t_day'].values.reshape(-1,1)
    y_train = train['total'].values

    ym, ys = y_train.mean(), y_train.std()
    ytr_n = (y_train - ym)/ys

    rbf = RBFKernel(lengthscale=5., variance=0.5)
    per1 = PeriodicKernel(period=365., lengthscale=72, variance=0.3)
    per2 = PeriodicKernel(period=30.5,lengthscale=24, variance=0.3)
    Ksum = SumKernel(rbf, per1, per2)

    t1 = time.time()

    Z0 = X_train[:60]
    gp  = OnlineSparseGP(Ksum, noise_var=0.1, max_points=200)
    gp.initialize(Z0, ytr_n[:60])

    preds_tr, vars_tr, t_tr = [], [], []
    for i,(x,y) in enumerate(zip(X_train, ytr_n)):
        mean, var, drift, E = gp.update_stream(x, y)          # update (may unlearn)
        preds_tr.append(mean[0]); vars_tr.append(var[0]); t_tr.append(x[0])
        if drift:
            print(f"Drift @day {x[0]} |E|={abs(E):.2f}")

    print("Time to run:", time.time() - t1)

    # denormalize
    p_tr = np.array(preds_tr)*ys + ym
    s_tr = np.sqrt(np.array(vars_tr))*ys
    a_tr = y_train[:len(p_tr)]
    Zx  = gp.Z.flatten()

    sorted_tr = np.argsort(t_tr)
    t_tr_sorted = np.array(t_tr)[sorted_tr]
    p_tr_sorted = np.array(p_tr)[sorted_tr]

    t_tr_smooth = np.linspace(t_tr_sorted.min(), t_tr_sorted.max(), 500)
    p_tr_spline = make_interp_spline(t_tr_sorted, p_tr_sorted, k=3)
    p_tr_smooth = p_tr_spline(t_tr_smooth)

    s_tr_sorted = np.array(s_tr)[sorted_tr]
    s_tr_spline = make_interp_spline(t_tr_sorted, s_tr_sorted, k=3)
    s_tr_smooth = s_tr_spline(t_tr_smooth)

    print("MSE:", np.sqrt(np.sum(np.square(y_train - p_tr))))

    plt.figure(figsize=(12,6))
    plt.plot(train['t_day'], y_train, 'o', markersize=2, label="Actual train")
    plt.plot(train['t_day'], p_tr, '-', label="Predicted train")
    plt.fill_between(train['t_day'],
                    p_tr - 1.96*s_tr,
                    p_tr + 1.96*s_tr,
                    alpha=0.2, label="95% CI")
    plt.scatter(Zx, np.interp(Zx, t_tr, p_tr), marker='x', s=80, color='k', label="Inducing")
    plt.title("Training: One-step Ahead")
    plt.xlabel("Days since 2011-11-01"); plt.ylabel("Load")
    plt.legend(); plt.grid(True)

    plt.show()



if __name__=="__main__":
    main()

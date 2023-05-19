import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CLIMATE_DATA_URL = "https://raw.githubusercontent.com/edebrouwer/gru_ode_bayes/master/gru_ode_bayes/datasets/Climate/small_chunked_sporadic.csv"  # noqa
DATA_ROOT = Path(__file__).resolve().parent / "climate"
LOCAL_PATH = DATA_ROOT / "climate-data-preproc.csv"


def download_preprocessed():
    print("Downloading dataset.")
    DATA_ROOT.mkdir(exist_ok=True)
    urllib.request.urlretrieve(CLIMATE_DATA_URL, LOCAL_PATH)


def generate_folds(num_folds=5, seed=432):
    print("Generating folds.")
    # Modified from https://github.com/edebrouwer/gru_ode_bayes/blob/master/data_preproc/Climate/generate_folds.py # noqa
    num_series = pd.read_csv(LOCAL_PATH)["ID"].nunique()
    np.random.seed(seed)

    for fold in range(num_folds):
        train_idx, test_idx = train_test_split(np.arange(num_series), test_size=0.1)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.2)
        fold_dir = DATA_ROOT / f"fold_idx_{fold}/"
        fold_dir.mkdir(exist_ok=True)

        np.save(fold_dir / "train_idx.npy", train_idx)
        np.save(fold_dir / "val_idx.npy", val_idx)
        np.save(fold_dir / "test_idx.npy", test_idx)


if __name__ == "__main__":
    download_preprocessed()
    generate_folds()

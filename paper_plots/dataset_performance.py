import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from data.gia_vslam_data_connector import GiaVSLAMdataConnector
from utils.definitions import get_project_root

root = get_project_root()


def dataset_stats():
    conn = GiaVSLAMdataConnector(floors=[0, 1, 2, 3, 4],
                                 devices=["Galaxy", "LG", "OnePlus", "S20"]
                                 ).load_dataset()

    colors = sns.color_palette('deep', n_colors=5)

    plt.figure(figsize=(6.4, 4.8))
    df = pd.DataFrame({"device": conn.device})
    df = df.groupby(['device']).size()/len(df)
    df.plot(kind='pie', label='', autopct='%1.1f%%', colors=colors)

    plt.figure(figsize=(6.4, 4.8))
    df = pd.DataFrame({"floor": conn.floor})
    df = df.groupby(['floor']).size() / len(df)
    df.plot(kind='pie', label='', autopct='%1.1f%%', colors=colors)
    plt.show()


def plot_cdf(error_vec, name, color, linestyle='solid'):
    n = len(error_vec)
    plt.plot(error_vec, [i / n for i in range(n)], linestyle=linestyle,
             color=color, label=name)


def static_performance(file="metrics.pickle"):

    print(file)
    print("")

    plt.figure()
    colors = sns.color_palette("deep")
    sns.set_theme(style='whitegrid')

    plot_dict = {'model': [], 'floor_ACC': [], 'Mean error': [], 'Median error': []}

    with open(file, 'rb') as f:

        metrics_1 = pickle.load(f)[0]

        for idx, (model, data) in enumerate(metrics_1.items()):
            error_vec = data['error_vec']

            plot_cdf(np.sort(error_vec), name=model, color=colors[idx], linestyle='--')

            plot_dict['model'] += [model]
            plot_dict['floor_ACC'] += [np.round(data['floor_ACC'], decimals=3)]
            plot_dict['Mean error'] += [np.round(data['MSE'], decimals=2)]
            plot_dict['Median error'] += [np.round(data['MSE (median)'], decimals=2)]

        df = pd.DataFrame(data=plot_dict)

        print(df.to_latex(index=False))


if __name__ == '__main__':
    dataset_stats()

    static_performance(file=root + "/exp/gia_vslam/device/random_split/metrics.pickle")

    static_performance(file=root + "/exp/gia_vslam/device/test_on_Galaxy/metrics.pickle")
    static_performance(file=root + "/exp/gia_vslam/device/test_on_LG/metrics.pickle")
    static_performance(file=root + "/exp/gia_vslam/device/test_on_OnePlus/metrics.pickle")
    static_performance(file=root + "/exp/gia_vslam/device/test_on_S20/metrics.pickle")

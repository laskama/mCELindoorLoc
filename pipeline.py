import pickle

import yaml
import argparse
import os

from data.dp_factory import get_data_provider
from utils.folder_setup import setup_directories
from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow as tf

from utils.definitions import get_project_root
from model.model_factory import get_model
from utils.tf_utils import set_seed

disable_eager_execution()
tf.config.experimental.set_visible_devices([], 'GPU')

root = get_project_root()


def run_model(model, fit=True):
    model.dp.split_idx = 0

    if fit:
        model.fit_model()

    metrics, polys, floor_pred = model.evaluate_model(load_weights=True, compute_error_vec=True)

    m_print = {k: v for k, v in metrics.items() if k not in ['error_vec']}

    print(m_print)

    return metrics, polys, floor_pred


def execute_pipeline(model_params, dataset_params, base_dir, seed_val=1):
    set_seed(seed_val, gpu_reproducability=True)

    m_type = model_params['type']

    dp = get_data_provider(dataset_params, m_type)

    model = get_model(model_params, base_dir, dp)

    metrics, polys, floor_pred = run_model(model, fit=not model.pr.get_param('pretrained'))

    return polys, floor_pred, dp, metrics


def execute_pipelines(conf_file):
    with open(conf_file) as conf:
        d = yaml.safe_load(conf)

    seed_vals = d['seed']
    if type(seed_vals) is not list:
        seed_vals = [seed_vals]

    metrics_seed = []
    for s_val in seed_vals:

        base_dir = d['dir']['base'] + str(s_val) + '/'

        setup_directories(base_dir)

        with open(base_dir + os.path.basename(conf_file), 'w') as f:
            yaml.dump(d, f)

        model_polys = []
        floor_preds = []
        metrics = {}

        for model in d['models']:
            polys, floor_pred, dp, m = execute_pipeline(
                model, dataset_params=d['data'], base_dir=base_dir,
                seed_val=s_val)
            model_polys += [polys]
            floor_preds += [floor_pred]

            metrics[model['name']] = m

        for k, m in metrics.items():
            m_print = {k: v for k, v in m.items() if k not in ['error_vec']}
            print("{}: {}".format(k, m_print))

        metrics_seed += [metrics]

    result_file = "metrics.pickle"
    if 'result_file' in d['dir']:
        result_file = d['dir']['result_file']
    with open(d['dir']['base'] + result_file, 'wb') as f:
        pickle.dump(metrics_seed, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Pipeline execution')

    parser.add_argument('-c', help=".yaml config file")

    args = parser.parse_args()

    execute_pipelines(args.c)

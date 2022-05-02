import os
import tensorflow as tf
import numpy as np
import random


def set_seed(seed, gpu_reproducability=False):
    if gpu_reproducability:
        os.environ['PYTHONHASHSEED'] = '0'

        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    else:
        tf.random.set_seed(seed)
        np.random.seed(seed)

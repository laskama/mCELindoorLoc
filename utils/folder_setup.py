import os


def setup_directories(base_dir):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
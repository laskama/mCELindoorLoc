#!/bin/bash
python pipeline.py -c config/gia/random/config_random_split.yml
python pipeline.py -c config/gia/device/config_Galaxy.yml
python pipeline.py -c config/gia/device/config_LG.yml
python pipeline.py -c config/gia/device/config_OnePlus.yml
python pipeline.py -c config/gia/device/config_S20.yml
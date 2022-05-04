from data.gia_vslam_data_connector import GiaVSLAMdataConnector
from data.mcel_data_provider import MCELdataProvider
from data.reg_data_provider import RegDataProvider
from data.tampere_data_connector import TampereDataConnector
from data.uji_data_connector import UJIdataConnector
from data.uts_data_connector import UTSdataConnector


def get_data_provider(dataset_params, m_type):

    d_params = dataset_params['params']
    dataset = dataset_params['dataset']

    # determine preprocessing scaling method
    powed_scaling = False
    std_scaling = False
    if 'scaling' in d_params:
        if d_params['scaling'] == 'standard':
            std_scaling = True
        elif d_params['scaling'] == 'powed':
            powed_scaling = True

    if dataset == 'tampere':
        conn = TampereDataConnector()
    elif dataset == 'uji':
        conn = UJIdataConnector()
    elif dataset == 'uts':
        conn = UTSdataConnector()
    elif dataset == 'gia_vslam':
        conn = GiaVSLAMdataConnector(floors=d_params['floors'],
                                     devices=d_params['devices'],
                                     test_devices=d_params['test_devices'] if 'test_devices' in d_params else None,
                                     test_trajectories=d_params['test_trajectories'] if 'test_trajectories' in d_params else None
                                     ).load_dataset()

    if m_type == 'mCEL':
        dp = MCELdataProvider(dataset_params['params'], dc=conn).load_dataset().generate_split_indices().generate_validation_indices()
        dp = dp.replace_missing_values().standardize_data(scaling_type=d_params['scaling'])
        dp = dp.transform_to_grid_encoding()

        # determine ground truth construction
        class_pad = False
        reg_pad = False
        weighted_grid_labels = False
        if 'class_pad' in d_params and d_params['class_pad']:
            class_pad = True
        if 'reg_pad' in d_params and d_params['reg_pad']:
            reg_pad = True
        if 'weighted_grid_labels' in d_params and d_params['weighted_grid_labels']:
            weighted_grid_labels = True

        dp = dp.compute_multilabel_aug_data(weighted_grid_labels)
        dp = dp.set_labels(class_pad, reg_pad)

    elif m_type == '3D':
        dp = RegDataProvider(dataset_params['params'], dc=conn).load_dataset().generate_split_indices().generate_validation_indices()
        dp = dp.replace_missing_values().standardize_data(scaling_type=d_params['scaling'])
        dp = dp.set_labels(scale_labels=True)

    else:
        dp = None

    return dp

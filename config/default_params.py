DEFAULT_PARAMS = {

    #
    # DLP parameters
    #

    # target vector construction
    'dist_th': 25.0,
    'range_th': 10.0,
    'close_th': 0.5,

    'dlp_reg_lambda': 5.0,

    #
    # mCEL parameters
    #

    # target vector construction
    'grid_size': 10,
    'padding_ratio': 0.3,


    #
    # Model parameters
    #

    # architecture
    'dropout': 0.5,
    'activation': 'relu',

    # fitting parameters
    'lr': 0.001,
    'batch_size': 32,
    'epochs': 50,
    'verbose': 0,
    'val': 0.2,
    'pretrained': True

}

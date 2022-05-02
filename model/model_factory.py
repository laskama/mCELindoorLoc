from model.mcel_model import MCELmodel
from model.model_definition import get_model_from_yaml_definition
from model.three_d_reg_model import ThreeDregModel


def get_model(model_params, base_dir, dp, input_dim):

    output_dim = dp.get_output_dim()

    m = get_model_from_yaml_definition(model_params, input_dim, output_dim, dp=dp)

    m_params = model_params['params'] if 'params' in model_params else {}
    m_type = model_params['type']
    model_name = model_params['name']

    if m_type == 'mCEL':
        model = MCELmodel(m_params, dp, base_dir, model_name=model_name)
    elif m_type == '3D':
        model = ThreeDregModel(m_params, dp, base_dir, model_name=model_name)
    else:
        model = None

    model.setup_model(m)

    return model

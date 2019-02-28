'''
    https://pytorch.org/tutorials/beginner/saving_loading_models.html
'''

import torch
import torch.nn as nn
import os
from collections import OrderedDict
from models.mobilenetv2 import MobileNetv2Model


def load_model(model_name, model_config=[], states_path="", model_path="", input_channels=0, pretrained=False, dropout=0.0, ruido=0.0, growth_rate=0, in_features=0, flat_size=0, out_features=0, out_type='relu', block_type=None, last_pool_size=0, cardinality=32, data_parallel=False):

    if model_path!="" and os.path.exists(model_path):
        return torch.load(model_path)
    elif model_path != "": assert False, "Wrong Model Path!"

    if not os.path.exists(states_path): assert False, "Wrong Models_States Path!"

    if 'MobileNetv2' in model_name:
        my_model = MobileNetv2Model(model_config, input_channels, out_features, flat_size, last_pool_size).cpu()
    else: assert False, "Model '" + str(model_name) + "' not configured!"
    if data_parallel: my_model = torch.nn.DataParallel(my_model, device_ids=range(torch.cuda.device_count()))

    model_state_dict = torch.load(states_path, map_location=lambda storage, location: storage)

    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        if 'module' in k:
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        else:
            my_model.load_state_dict(model_state_dict)
            return my_model

    # load params
    my_model.load_state_dict(new_state_dict)
    return my_model
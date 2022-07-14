##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################

from .optimizers import get_optim_scheduler  # noqa: E401
from .starts import get_machine_info  # noqa: E401
from .starts import (copy_checkpoint, prepare_logger, prepare_seed,
                     save_checkpoint)


def get_procedures(procedure):
    from .basic_main import basic_train, basic_valid
    from .search_main import search_train, search_valid
    from .search_main_v2 import search_train_v2
    from .simple_KD_main import simple_KD_train, simple_KD_valid

    train_funcs = {
        'basic': basic_train,
        'search': search_train,
        'Simple-KD': simple_KD_train,
        'search-v2': search_train_v2
    }
    valid_funcs = {
        'basic': basic_valid,
        'search': search_valid,
        'Simple-KD': simple_KD_valid,
        'search-v2': search_valid
    }

    train_func = train_funcs[procedure]
    valid_func = valid_funcs[procedure]
    return train_func, valid_func

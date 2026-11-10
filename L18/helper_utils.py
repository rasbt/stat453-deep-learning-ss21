import random
import torch
import numpy as np
import os


def set_deterministic():
    # MSCS435 (atwumasi): required by torch.use_deterministic_algorithms(True)
    # on GPU (CuBLAS ops are otherwise nondeterministic); must be set before
    # any CUDA context/op, so it goes first.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    # MSCS435 (atwumasi): torch.set_deterministic was removed from PyTorch;
    # use its replacement instead.
    torch.use_deterministic_algorithms(True)
    
    
def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
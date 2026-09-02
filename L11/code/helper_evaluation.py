# imports from installed libraries
import os
import numpy as np
import random
import torch


def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_deterministic():
    # MSCS435 (atwumasi): required by torch.use_deterministic_algorithms(True)
    # on GPU (CuBLAS ops are otherwise nondeterministic); must be set before
    # any CUDA context/op, so it goes first.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # MSCS435 (atwumasi): dropped the distutils-based torch<1.7 branch —
    # distutils was removed in Python 3.12, and torch<1.7 is no longer relevant.
    torch.use_deterministic_algorithms(True)


def compute_accuracy(model, data_loader, device):

    with torch.no_grad():

        correct_pred, num_examples = 0, 0

        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.float().to(device)

            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)

            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100

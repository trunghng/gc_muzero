from collections import defaultdict
import random
from typing import Dict, Tuple

import numpy as np
import torch


def ftensor(x: np.ndarray, device: str = None) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dictkv_to_dictvk(dict_: Dict) -> Dict:
    new_dict = defaultdict(list)
    for k, v in dict_.items():
        new_dict[v].append(k)
    return new_dict


def to_np(x: torch.Tensor | Tuple[torch.Tensor]) -> np.ndarray | Tuple[np.ndarray]:
    if isinstance(x, tuple):
        return list(map(lambda y: y.cpu().detach().numpy(), x))
    return x.cpu().detach().numpy()


def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def probs_to_logits(probs: np.ndarray) -> np.ndarray:
    tiny = np.finfo(probs.dtype).tiny
    return np.log(np.maximum(probs, tiny))

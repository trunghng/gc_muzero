import numpy as np


def mask_illegal_actions(action_mask: np.ndarray) -> np.ndarray:
    """Returns legal actions only"""
    return np.argwhere(action_mask==1).squeeze(-1)


def mask_illegal_action_logits(
    action_logits: np.ndarray,
    legal_actions: np.ndarray
) -> np.ndarray:
    """Returns logits with zero mass to illegal actions"""
    action_logits = action_logits - np.max(action_logits, keepdims=True)
    min_logit = np.finfo(action_logits.dtype).min
    return np.where(legal_actions, action_logits, min_logit)

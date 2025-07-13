import torch

def safe_delete_tensors(*tensors):
    """Safely deletes tensors and clears memory."""
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensor.detach()
    del tensors
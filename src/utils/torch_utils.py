import os
import torch
import numpy as np
from collections import OrderedDict

from src.utils.distributed_utils import get_current_device, is_distributed, is_main_process


def load_from_checkpoint(net: torch.nn.Module, checkpoint_path: str) -> None:
    '''
    Loads the state dict from stored model in .pt format.
    '''
    assert type(checkpoint_path) == str

    if os.path.isfile(checkpoint_path):
        assert(os.path.splitext(checkpoint_path)[1]) in ['.pt', '.pth']
        
        if is_main_process():
            print(f'Loading model from checkpoint: {checkpoint_path}')

        checkpoint = torch.load(checkpoint_path, map_location=get_current_device())

        state_dict = checkpoint['model']
        new_state_dict = OrderedDict()

        # Creating a new state dict
        if is_distributed():
            # For a parallel environment
            for key, value in state_dict.items():
                if 'module' not in key:
                    new_key = 'module.' + key
                    new_state_dict[new_key] = value
        else:    
            # For a non-parallel environment
            for key, value in state_dict.items():
                new_key = key.replace('module.', '')
                new_state_dict[new_key] = value

        net.load_state_dict(new_state_dict)
    else:
        raise FileNotFoundError(f'Invalid checkpoint path: {checkpoint_path}')


def logits_to_preds(logits: torch.Tensor, is_multiclass: bool=True) -> torch.Tensor:
    '''
    Passes network outputs (logits) through activation functions to compute predictions. 
    '''
    if is_multiclass:
        logits = logits.log_softmax(dim=1).exp()
        _, preds = torch.max(logits, dim=1)
    else:
        logits = torch.nn.functional.logsigmoid(logits).exp()
        preds = torch.where(logits > 0.5, 1., 0.)
    
    return preds


def store_batch_predictions(preds: torch.Tensor, save_path: str, indexes: list[int]) -> None:
    '''
    Stores network predictions as compressed numpy arrays.
    '''
    save_path = os.path.join(save_path, 'predictions')
    os.makedirs(save_path, exist_ok=True)

    preds = preds.detach().cpu().numpy()
    
    for image, idx in zip(preds, indexes):
        image = np.squeeze(image)
        np.savez_compressed(os.path.join(save_path, f'{idx}.npz'), image)
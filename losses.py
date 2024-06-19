from imports import *
from torch.nn.functional import pairwise_distance
import torch
from torch import nn

def nffn_mse_min_dist(y_pred, input, outp_dict, max_y_train_len, device):
    batch_losses = []
    model_output_list = []
    ground_truth_list = []
    y_train_collected = torch.empty((0, len(outp_dict[0][0]))).to(device)
    for idx, (y_pred_single, input_single) in enumerate(zip(y_pred, input)):
        y_train_set = outp_dict[int(input_single[0])]  # Assuming this indexing is valid and returns a tensor

        # If y_train_set is empty, continue to next
        if y_train_set.nelement() == 0:
            continue
        
        # This works by checking across dimension 1 (list_size), if there's any non-zero value
        non_zero_mask = torch.any(y_train_set != 0, dim=1)
        # Filter out zero lists using the mask
        y_train_set = y_train_set[non_zero_mask]
        y_pred_single_expanded = y_pred_single.unsqueeze(0).expand_as(y_train_set)
        distances = torch.norm(y_train_set - y_pred_single_expanded, dim=1)  # Euclidean distance
        
        # Use distances to select the relevant y_train instance
        min_distanCrossEntropyLossce, min_idx = torch.min(distances, dim=0)  # Find minimal distance
        selected_y_train = y_train_set[min_idx]
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_pred_single, selected_y_train.float())
        y_train_collected = torch.cat((y_train_collected, selected_y_train.unsqueeze(0)), dim=0)
        batch_losses.append(loss)

    # Step 2: Aggregate losses from all instances in the batch
    if batch_losses:
        total_loss = torch.stack(batch_losses).mean()  # Stack losses and compute mean
    else:
        total_loss = torch.tensor(0.0, device=y_pred.device)  # No valid losses, return 0
    
    return total_loss, y_train_collected

def reverse(inpt):
    inpt = np.argmax(inpt, axis=1) 
    inpt = [calculi.symb_reverse[num.item()] for num in inpt]
    inpt = ''.join(inpt)
    return inpt

def new_threed_mse_min_dist(y_pred, input, outp_dict, max_y_train_len, device):
    # Assuming each entry in y_pred corresponds to a set in outp_dict directly by index
    # This approach attempts a more vectorized solution, but might need adjustments
    
    # Step 1: Vectorize distance calculation where possible
    # Need careful tensor reshaping to broadcast correctly for batch distance computation
    batch_losses = []
    y_train_collected = torch.empty((0, len(outp_dict[0][0]), 14)).to(device)
    for idx, (y_pred_single, input_single) in enumerate(zip(y_pred, input)):
        if input_single.dim() == 2:
            y_train_set = outp_dict[int(input_single[0][0])]
        else:
            y_train_set = outp_dict[int(input_single[0])]
        if y_train_set.nelement() == 0:
            continue
        diff = y_train_set.size(1) - y_pred_single.size(0)
        non_zero_mask = y_train_set.view(outp_dict.shape[1],-1).any(dim=1)
        y_train_set = y_train_set[non_zero_mask]
        y_pred_single = torch.nn.functional.pad(y_pred_single, (0, 0, 0, diff))
        
        target_tensor = y_pred_single.unsqueeze(0)
        distances = torch.norm(y_train_set - y_pred_single, dim=[1, 2])
        min_index = distances.argmin().item()

        selected_y_train = y_train_set[min_index]
        criterion = nn.CrossEntropyLoss()
        
        loss = criterion(y_pred_single, selected_y_train.float())
        y_train_collected = torch.cat((y_train_collected, selected_y_train.unsqueeze(0)), dim=0)
        batch_losses.append(loss)
    # Step 2: Aggregate losses from all instances in the batch
    if batch_losses:
        total_loss = torch.stack(batch_losses).mean()  # Stack losses and compute mean
    else:
        total_loss = torch.tensor(0.0, device=y_pred.device)  # No valid losses, return 0
    
    return total_loss, y_train_collected


def new_new_threed_mse_min_dist(y_pred, input, outp_dict, max_y_train_len, device):
    # Assuming each entry in y_pred corresponds to a set in outp_dict directly by index
    # This approach attempts a more vectorized solution, but might need adjustments
    
    # Step 1: Vectorize distance calculation where possible
    # Need careful tensor reshaping to broadcast correctly for batch distance computation
    batch_losses = []
    y_train_collected = torch.empty((0, len(outp_dict[0][0]), 14)).to(device)
    for idx, (y_pred_single, input_single) in enumerate(zip(y_pred, input)):
        if input_single.dim() == 2:
            y_train_set = outp_dict[int(input_single[0][0])]
        else:
            y_train_set = outp_dict[int(input_single[0])]
        if y_train_set.nelement() == 0:
            continue
        diff = y_train_set.size(1) - y_pred_single.size(0)
        non_zero_mask = y_train_set.view(outp_dict.shape[1],-1).any(dim=1)
        y_train_set = y_train_set[non_zero_mask]
        y_pred_single = torch.nn.functional.pad(y_pred_single, (0, 0, 0, diff))
        
        target_tensor = y_pred_single.unsqueeze(0)
        distances = torch.norm(y_train_set - y_pred_single, dim=[1, 2])
        min_index = distances.argmin().item()

        selected_y_train = y_train_set[min_index]
        criterion = nn.CrossEntropyLoss()
        
        loss = criterion(y_pred_single, selected_y_train.float())
        y_train_collected = torch.cat((y_train_collected, selected_y_train.unsqueeze(0)), dim=0)
        batch_losses.append(loss)
    
    return batch_losses, y_train_collected
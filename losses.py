from imports import *
from torch.nn.functional import pairwise_distance
import torch

# General custom loss function
def mse_min_dist(y_pred, x_train, outp_dict, max_y_train_len, model):
    counter = 0
    loss_batch_list = []
    while counter < len(y_pred):
        val_i = y_pred[counter]
        x_i = x_train[counter]
        if model == "ffn":
            y_train = outp_dict[int(x_i[0])]
        else:
            y_train = outp_dict[int(x_i[0][0])]
        if len(y_train) > 0:
            if len(y_train) > 1:
                distances = []
                i = 0
                while i < len(y_train):
                    distance = pairwise_distance(y_train[i], val_i, keepdim=True).mean()
                    distances.append(distance)
                    i += 1
                min_dist = max(distances) 
                min = distances.index(min_dist)
                y_train = y_train[min]
            else:
                y_train = y_train[0]
        loss_batch_list.append(torch.mean((val_i - y_train) ** 2))
        counter += 1
    loss = sum(loss_batch_list)/len(loss_batch_list)
    return loss

# The same but with all values loaded to the gpu
def ffn_mse_min_dist(y_pred, x_train, outp_dict, max_y_train_len):
    counter = 0
    loss_batch_list = []
    while counter < len(y_pred):
        val_i = y_pred[counter]
        x_i = x_train[counter]
        y_train = outp_dict[int(x_i[0])]
        if len(y_train) > 0:
            if len(y_train) > 1:
                distances = []
                i = 0
                while i < len(y_train):
                    distance = pairwise_distance(y_train[i], val_i, keepdim=True).mean()
                    distances.append(distance)
                    i += 1
                min_dist = max(distances) 
                min = distances.index(min_dist)
                y_train = y_train[min]
            else:
                y_train = y_train[0]
        loss_batch_list.append(torch.mean((val_i - y_train) ** 2))
        counter += 1
    loss = sum(loss_batch_list)/len(loss_batch_list)
    return loss

def nffn_mse_min_dist(y_pred, x_train, outp_dict, max_y_train_len):
    # Assuming each entry in y_pred corresponds to a set in outp_dict directly by index
    # This approach attempts a more vectorized solution, but might need adjustments
    
    # Step 1: Vectorize distance calculation where possible
    # Need careful tensor reshaping to broadcast correctly for batch distance computation
    batch_losses = []
    model_output_list = []
    ground_truth_list = []
    for idx, (y_pred_single, x_train_single) in enumerate(zip(y_pred, x_train)):
        y_train_set = outp_dict[int(x_train_single[0])]  # Assuming this indexing is valid and returns a tensor

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
        #print(y_pred_single)
        #print(selected_y_train)
        loss = criterion(y_pred_single, selected_y_train.float())
        batch_losses.append(loss)

    # Step 2: Aggregate losses from all instances in the batch
    if batch_losses:
        total_loss = torch.stack(batch_losses).mean()  # Stack losses and compute mean
    else:
        total_loss = torch.tensor(0.0, device=y_pred.device)  # No valid losses, return 0
    
    return total_loss

# Optimization of the above by gpt
def old_nffn_mse_min_dist(y_pred, x_train, outp_dict, max_y_train_len):
    # Assuming each entry in y_pred corresponds to a set in outp_dict directly by index
    # This approach attempts a more vectorized solution, but might need adjustments
    
    # Step 1: Vectorize distance calculation where possible
    # Need careful tensor reshaping to broadcast correctly for batch distance computation
    batch_losses = []
    for idx, (y_pred_single, x_train_single) in enumerate(zip(y_pred, x_train)):
        y_train_set = outp_dict[int(x_train_single[0])]  # Assuming this indexing is valid and returns a tensor

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
        min_distance, min_idx = torch.min(distances, dim=0)  # Find minimal distance
        
        selected_y_train = y_train_set[min_idx]
        
        # MSE Loss for selected instance
        mse_loss = torch.mean((selected_y_train - y_pred_single) ** 2)
        batch_losses.append(mse_loss)
    
    # Step 2: Aggregate losses from all instances in the batch
    if batch_losses:
        total_loss = torch.stack(batch_losses).mean()  # Stack losses and compute mean
    else:
        total_loss = torch.tensor(0.0, device=y_pred.device)  # No valid losses, return 0
    
    return total_loss

def threed_mse_min_dist(y_pred, x_train, outp_dict, max_y_train_len):
    # Assuming each entry in y_pred corresponds to a set in outp_dict directly by index
    # This approach attempts a more vectorized solution, but might need adjustments
    
    # Step 1: Vectorize distance calculation where possible
    # Need careful tensor reshaping to broadcast correctly for batch distance computation
    batch_losses = []
    for idx, (y_pred_single, x_train_single) in enumerate(zip(y_pred, x_train)):
        y_train_set = outp_dict[int(x_train_single[0][0])]  # Assuming this indexing is valid and returns a tensor

        # If y_train_set is empty, continue to next
        if y_train_set.nelement() == 0:
            continue
        
        # This works by checking across dimension 1 (list_size), if there's any non-zero value
        non_zero_mask = torch.any(y_train_set != 0, dim=1)

        # Filter out zero lists using the mask
        #y_train_set = y_train_set.view(1,70,32)
        y_train_set = y_train_set[non_zero_mask]
        y_pred_single_expanded = y_pred_single.unsqueeze(0).expand_as(y_train_set)
        distances = torch.norm(y_train_set - y_pred_single_expanded, dim=1)  # Euclidean distance
        
        # Use distances to select the relevant y_train instance
        min_distance, min_idx = torch.min(distances, dim=0)  # Find minimal distance
        
        selected_y_train = y_train_set[min_idx]
        criterion = nn.CrossEntropyLoss()
        #print(y_pred_single)
        #print(selected_y_train)
        loss = criterion(y_pred_single, selected_y_train.float())
        batch_losses.append(loss)
    # Step 2: Aggregate losses from all instances in the batch
    if batch_losses:
        total_loss = torch.stack(batch_losses).mean()  # Stack losses and compute mean
    else:
        total_loss = torch.tensor(0.0, device=y_pred.device)  # No valid losses, return 0
    
    return total_loss

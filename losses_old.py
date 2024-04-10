from imports import *

def min_dist(y_pred, x_train, y_train, max_y_train_len, model):
    counter = 0
    y_train_collected = []
    while counter < len(y_pred):
        val_i = y_pred[counter]
        x_i = x_train[counter]
        if model == "ffn":
            y_train = y_train[int(x_i[0])]
        else:
            y_train = y_train[int(x_i[0][0])]
            if len(y_train) > 1:
                y = []
                for i in y_train:
                    y_part = util.convert_single(i, max_y_train_len)
                    y.append(y_part)
                i = 0
                distances = []
                while i < len(y):
                    distance = pairwise_distance(y[i], val_i, keepdim=True).mean()
                    distances.append(distancey_train)
                    i += 1
                min_dist = max(distances) 
                min = distances.index(min_dist)
                y_train = y[min]
            else:
                y_train = y_train[0]
                y_train = util.convert_single(y_train, max_y_train_len)
        y_train_collected += y_train
    return y_train_collected

'''
# y_pred for a single input [1, max_length]
# Assume we have an index 'idx' to fetch the correct set of derivations

# Fetch the derivations for the given index; shape = [max_derivations, max_length]
correct_derivations = vals_tensor[idx]

# Calculate pairwise distances or any other suitable metric between y_pred and each correct_derivation
# Implement or choose an appropriate distance metric function that can operate on GPU tensors

# For illustration: using Euclidean distance (simplified and not optimized)
distances = torch.sqrt(((correct_derivations - y_pred) ** 2).sum(dim=1))

# Find the index of the closest derivation
closest_idx = torch.argmin(distances)

# Select the closest derivation for loss calculation
closest_derivation = correct_derivations[closest_idx]
'''
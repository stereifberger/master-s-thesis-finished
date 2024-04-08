from imports import *

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
        loss_batch_list.append(torch.mean((val_i - y_train) ** 2))
        counter += 1
    loss = sum(loss_batch_list)/len(loss_batch_list)
    return loss
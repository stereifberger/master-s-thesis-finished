from imports import *

#https://medium.com/@hugmanskj/hands-on-implementing-a-simple-mathematical-calculator-using-sequence-to-sequence-learning-85b742082c72
def train_model(model, dataloader_train, dataloader_test, optimizer, criterion, epochs, device, max_y_length, y_train):
    CELtrain = []
    CELtest = []
    ACCtrain = []
    ACCtest = []

    for epoch in range(epochs):

        # Train loss
        model.train()
        correct_train = 0
        total_train = 0
        epoch_loss = 0
        for i, x_train in enumerate(dataloader_train):
            x_train = x_train.to(device)
            optimizer.zero_grad()  #  <-- FOR PYTORCH

            y_pred = model(x_train[:,1:], max_y_length)               # Get the model's output for batch
            if y_train.dim() == 3:
                loss, y_train_collected = losses.nffn_mse_min_dist(y_pred, x_train, y_train, max_y_length, device) # Calculate loss
            else:
                loss, y_train_collected  = losses.new_threed_mse_min_dist(y_pred, x_train, y_train, max_y_length, device) # Calculate loss
            loss.backward()  #  <-- FOR PYTORCH
            optimizer.step() #  <-- FOR PYTORCH
            epoch_loss += loss.item()

        pred_labels = torch.argmax(y_pred, dim=-1)
        target_labels = torch.argmax(y_train_collected, dim=-1)
        pred_labels = pred_labels.view(-1)  # shape: [batch size * sequence length]
        target_labels = target_labels.view(-1)
        correct = (pred_labels == target_labels).sum().item()
        total = target_labels.size(0)
        train_accuracy = correct / total
        ACCtrain.append(train_accuracy)

        CELtrain.append(float(epoch_loss / len(dataloader_train)))

        del y_train_collected
        torch.cuda.empty_cache()

        # Test loss
        model.eval()
        correct_test = 0
        total_test = 0
        test_loss = 0
        with torch.no_grad():
            for i, x_test in enumerate(dataloader_test):
                x_test = x_test.to(device)
                y_pred = model(x_test[:, 1:], max_y_length)
                if y_train.dim() == 3:
                    loss, y_test_collected  = losses.nffn_mse_min_dist(y_pred, x_test, y_train, max_y_length, device) # Calculate loss
                else:
                    loss, y_test_collected  = losses.new_threed_mse_min_dist(y_pred, x_test, y_train, max_y_length, device) # Calculate loss
                test_loss += loss.item()

        pred_labels = torch.argmax(y_pred, dim=-1)
        target_labels = torch.argmax(y_test_collected, dim=-1)
        pred_labels = pred_labels.view(-1)  # shape: [batch size * sequence length]
        target_labels = target_labels.view(-1)
        correct = (pred_labels == target_labels).sum().item()
        total = target_labels.size(0)
        test_accuracy = correct / total

        CELtest.append(float(test_loss / len(dataloader_train)))
        ACCtest.append(test_accuracy)
        

        print(f"""Ep. {epoch+1:02}, CEL-Train: {epoch_loss / len(dataloader_train):.4f} | CEL-Test: {test_loss / len(dataloader_test):.4f} | ACC-Train: {train_accuracy:.4f} | ACC-Test:  {test_accuracy:.4f}""")
    return CELtrain, CELtest, ACCtrain, ACCtest 


def sanity(model, dataloader, device, max_y_length):
    with torch.no_grad():
        for i, x_test in enumerate(dataloader):
            x_test = x_test.to(device)
            #x_test = x_test.float()
            y_pred = model(x_test[:,1:], max_y_length)               # Get the model's output for batch
            index = 0
            while index < len(y_pred):
                pred = y_pred[index].reshape(-1, 14)
                test = x_test[index][1:]
                pred = torch.argmax(pred, dim=1) 
                test = [calculi.symb_reverse[num.item()] for num in test]
                test = ''.join(test)
                pred = [calculi.symb_reverse[num.item()] for num in pred]
                pred = ''.join(pred)
                print(f"INPUT: {test}")
                print(f"OUTPUT: {pred}")
                index += 1

def sanity_with_loss(model, dataloader, y_train, device, max_y_length):
    with torch.no_grad():
        for i, x_test in enumerate(dataloader):
            x_test = x_test.to(device)
            #x_test = x_test.float()
            y_pred = model(x_test[:,1:], max_y_length)               # Get the model's output for batch
            if y_train.dim() == 3:
                loss, y_train_collected = losses.nffn_mse_min_dist(y_pred, x_test, y_train, max_y_length, device) # Calculate loss
            else:
                loss, y_train_collected  = losses.new_new_threed_mse_min_dist(y_pred, x_test, y_train, max_y_length, device) # Calculate loss
            index = 0
            while index < len(y_pred):
                pred = y_pred[index].reshape(-1, 14)
                test = x_test[index][1:]
                pred = torch.argmax(pred, dim=1) 
                test = [calculi.symb_reverse[num.item()] for num in test]
                test = ''.join(test)
                pred = [calculi.symb_reverse[num.item()] for num in pred]
                pred = ''.join(pred)
                print(f"INPUT: {test}")
                print(f"OUTPUT: {pred}")
                print(f"OUTPUT: {loss[index]}")
                index += 1
            total_loss = torch.stack(loss).mean()  # Stack losses and compute mean
            print(f"AVERAGE LOSS: {total_loss}")

def sanity_r(model, dataloader, device, max_y_length):
    with torch.no_grad():
        for i, x_test in enumerate(dataloader):
            x_test = x_test.to(device)
            x_test = x_test.float()
            print(x_test.shape)
            y_pred = model(x_test[:,1:], max_y_length)               # Get the model's output for batch
            index = 0
            while index < len(y_pred):
                pred = y_pred[index].reshape(-1, 14)
                print(x_test.shape)
                test = x_test[index][1:].reshape(-1, 14)
                test = torch.argmax(test, dim=1) 
                print(test.shape)
                pred = torch.argmax(pred, dim=1) 
                test = [calculi.symb_reverse[num.item()] for num in test]
                test = ''.join(test)
                pred = [calculi.symb_reverse[num.item()] for num in pred]
                pred = ''.join(pred)
                print(f"INPUT: {test}")
                print(f"OUTPUT: {pred}")
                index += 1
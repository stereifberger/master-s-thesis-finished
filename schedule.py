from imports import *

#https://medium.com/@hugmanskj/hands-on-implementing-a-simple-mathematical-calculator-using-sequence-to-sequence-learning-85b742082c72
def train_model(model, dataloader_train, dataloader_test, optimizer, criterion, epochs, device, max_y_length, y_train):
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

        # Train Accuracy
            _, predicted_train = torch.max(y_pred, 0)
            print(predicted_train.shape)
            print(y_train_collected.shape)
            correct_train += (predicted_train == y_train_collected).sum().item()
            total_train += y_train.size(0)
        train_accuracy = correct_train / total_train
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
                    loss, y_test_collected  = losses.nffn_mse_min_dist(y_pred, x_train, y_train, max_y_length, device) # Calculate loss
                else:
                    loss, y_test_collected  = losses.new_threed_mse_min_dist(y_pred, x_train, y_train, max_y_length, device) # Calculate loss
                test_loss += loss

                _, predicted_test = torch.max(y_pred, 0)
                correct_test += (predicted_test == y_test_collected).sum().item()
                total_test += y_train.size(0)

        test_accuracy = correct_test / total_test

        print(f"""Ep. {epoch+1:02}, CRL-Train: {epoch_loss / len(dataloader_train):.4f} | CRL-Test: {test_loss / len(dataloader_test):.4f} | ACC-Train: {train_accuracy:.4f} | ACC-Tesh:  {test_accuracy:.4f}""")




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
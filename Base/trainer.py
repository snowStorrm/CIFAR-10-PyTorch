import cifar10_nn, torch, torchvision

def iterate(dataloader, model, lossFunc, optimizer):
    #size = len(dataloader.dataset)
    model.train()
    # Loop through dataset
    for batch, (X, Y) in enumerate(dataloader):
        # Load the tensors in batches
        # - Refer to CIFAR10_NN for what these look like
        X, Y = X.to(model.device), Y.to(model.device)
        # Predict where the inputs should go
        prediction = model(X)
        loss = lossFunc(prediction, Y)
        # Backpropagate to adjust the node weights
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # If at end of current batch, print the loss and iteration progress
        if batch % 100 == 0:
            loss = loss.item()
            # print(f"Loss = {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, lossFunc):
    size = len(dataloader.dataset)
    batches = len(dataloader)
    model.eval()
    testLoss, correct = 0, 0
    with torch.no_grad():
        # Loop through dataset
        for X, Y in dataloader:
            X, Y = X.to(model.device), Y.to(model.device)
            # Input an image into the model
            prediction = model(X)
            # Compare what the model classified it as to what it actually is and compute accuracy based on correctness
            testLoss += lossFunc(prediction, Y).item()
            correct += (prediction.argmax(1)==Y).type(torch.float).sum().item()
    testLoss /= batches
    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%, Loss: {testLoss:>0.3f}")
    return correct*100
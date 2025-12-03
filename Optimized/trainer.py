import cifar10_nn, torch, torchvision

def iterate(dataloader, model: cifar10_nn.CIFAR10_NN, lossFunc, optimizer):
    size = len(dataloader.dataset)
    model.train()
    # Loop through dataset
    for batch, (images, categories) in enumerate(dataloader):
        # Load the tensors in batches
        # - Refer to CIFAR10_NN for what these look like
        images, categories = images.to(model.device), categories.to(model.device)
        # Predict where the inputs should go
        prediction = model(images)
        loss = lossFunc(prediction, categories)
        # Backpropagate to adjust the node weights
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Pick the model's answer
        loss += loss.item()
        loss /= size/model.batchSize

        # If at end of current batch, print the loss and iteration progress
        #if batch % 100 == 0:
            #current = (batch)*len(images)
            #print(f"Loss = {loss:>7f} [{current:>5d}/{size:>5d}]")

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
    return correct*100, testLoss
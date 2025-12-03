import cifar10_nn, trainer, torch, analysis

# Use a GPU or other accelerator if available, otherwise use the CPU
computeDevice = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
# Graph x-axis
metric = [float(0)]
# Graph y-axis
accuracy = [float(0)]
# Best iteration tracker
bestAccuracy, bestEpoch = 0, 0
print(f"Using {computeDevice} as compute accelerator...")
# Create an instance of the CIFAR-10 Neural Network model
# Params: batchSize, device, iterations, learningRate
model = cifar10_nn.CIFAR10_NN(16, computeDevice, 100, 0.0075)
model.to(computeDevice)
lossFunc  = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=model.learningRate, weight_decay=1e-6, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=10, min_lr=1e-6)
# Run one training iteration
# - Use the model instance we created alongside the CIFAR-10 dataset
# - Our loss function will be Cross Entropy Loss, which is very useful for categorization neural networks
# - Our optimizer will be Stochastic Gradient Descent with a learning rate of 1%. 
#   - All this does is makes iteration more efficient. Its inner-workings are far too complex to explain in comments.
for j in range(model.iterations):
    print(f"\nEpoch {j+1}: ")
    trainer.iterate(model.trainingDataLoader, model, lossFunc, optimizer)
    metric.append(j)
    # Test accuracy and find average loss
    acc, testLoss = trainer.test(model.testDataLoader, model, torch.nn.CrossEntropyLoss())
    # Save the best model
    if acc > bestAccuracy:
        bestAccuracy = acc
        bestEpoch = j+1
        torch.save(model.state_dict(), f"./models/Best_Model_Bsize16.pt")
    accuracy.append(acc)
    # Adjust learning rate based on loss
    scheduler.step(testLoss)
    print(f"Accuracy: {acc:>0.2f}%, Loss: {testLoss:>0.3f}")
    print(f"Best Accuracy: {bestAccuracy:>0.2f}% in Epoch {bestEpoch}")
# Export the data to an existing sheet
analysis.exportResults(metric, accuracy)

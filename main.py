import cifar10_nn, trainer, torch, analysis

# Use a GPU or other accelerator if available, otherwise use the CPU
computeDevice = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
# Graph x-axis
metric = []
# Graph y-axis
accuracy = []
print(f"Using {computeDevice} as compute accelerator...")
print("Variation: Batch Size")
# Loop through the intended variations and export the data
for i in range(16):
    # Create an instance of the CIFAR-10 Neural Network model
    # Params: batchSize, device, iterations, learningRate
    acc = 0
    varying = (i+1)*16
    metric.append(varying)
    model = cifar10_nn.CIFAR10_NN(varying, computeDevice, 5, 1e-2).to(computeDevice)
    # Run one training iteration
    # - Use the model instance we created alongside the CIFAR-10 dataset
    # - Our loss function will be Cross Entropy Loss, which is very useful for categorization neural networks
    # - Our optimizer will be Stochastic Gradient Descent with a learning rate of 1%. 
    #   - All this does is makes iteration more efficient. Its inner-workings are far too complex to explain in comments.
    print(f"\nVarying parameter value: {varying}")
    for j in range(model.iterations):
        trainer.iterate(model.trainingDataLoader, model, torch.nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr=model.learningRate))
        acc = trainer.test(model.testDataLoader, model, torch.nn.CrossEntropyLoss())
    accuracy.append(acc)
# Export the data to an existing sheet
analysis.exportResults(metric, accuracy, "Batch Sizes")
    

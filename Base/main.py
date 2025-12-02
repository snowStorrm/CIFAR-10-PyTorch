import cifar10_nn, trainer, torch, analysis

# Use a GPU or other accelerator if available, otherwise use the CPU
computeDevice = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
# Graph x-axis
metric = []
# Graph y-axis
accuracy = []
print(f"Using {computeDevice} as compute accelerator...")
print("Variation: NN")
# Loop through the intended variations and export the data
for i in range(1):
    # Create an instance of the CIFAR-10 Neural Network model
    # Params: batc4hSize, device, iterations, learningRate
    acc = 0
    varying = i*1e-3
    metric.append(varying)
    model = cifar10_nn.CIFAR10_NN(16, computeDevice, 10, 0.0075).to(computeDevice)
    # Run one training iteration
    # - Use the model instance we created alongside the CIFAR-10 dataset
    # - Our loss function will be Cross Entropy Loss, which is very useful for categorization neural networks
    # - Our optimizer will be Stochastic Gradient Descent with a learning rate of 1%. 
    #   - All this does is makes iteration more efficient. Its inner-workings are far too complex to explain in comments.
    print(f"\nVarying parameter value: {varying}")
    for j in range(model.iterations):
        trainer.iterate(model.trainingDataLoader, model, torch.nn.CrossEntropyLoss(), torch.optim.Adam(params=model.parameters(), lr=model.learningRate))
        acc = trainer.test(model.testDataLoader, model, torch.nn.CrossEntropyLoss())
    accuracy.append(acc)
# Export the data to an existing sheet
analysis.exportResults(metric, accuracy, "NN Format")
    

import cifar10_nn, trainer, torch, analysis, openpyxl

openpyxl.load_workbook('./out.xlsx')

# Use a GPU or other accelerator if available, otherwise use the CPU
computeDevice = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
metric = []
accuracy = []
print(f"Using {computeDevice} as compute accelerator...")
print("Variation: Batch Size")
# This will loop until the change in accuracy between variations is less than 0.5%, i.e. we've reached the point of diminishing returns
for i in range(1, 50000):
    # Create an instance of the CIFAR-10 Neural Network model
    # Params: batchSize, device, iterations, learningRate
    acc = 0
    batchSize=i
    metric.append(batchSize)
    model = cifar10_nn.CIFAR10_NN(batchSize, computeDevice, 5, 1e-2).to(computeDevice)
    # Run one training iteration
    # - Use the model instance we created alongside the CIFAR-10 dataset
    # - Our loss function will be Cross Entropy Loss, which is very useful for categorization neural networks
    # - Our optimizer will be Stochastic Gradient Descent with a learning rate of 1%. 
    #   - All this does is makes iteration more efficient. Its inner-workings are far too complex to explain in comments.
    print(f"\nVarying parameter value: {i}")
    for j in range(model.iterations):
        trainer.iterate(model.trainingDataLoader, model, torch.nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr=model.learningRate))
        acc = trainer.test(model.testDataLoader, model, torch.nn.CrossEntropyLoss())
    accuracy.append(acc)
    if ((accuracy[0 if i == 1 else i-2] - accuracy[i-1]) <= 0.005) and i > 14: break
# Export the data to an existing sheet
analysis.exportResults(metric, accuracy)
    

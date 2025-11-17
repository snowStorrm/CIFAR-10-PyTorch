import torch
import torchvision

# Load training and test data
# We're using CIFAR-10:
#   - 32x32 full-color images (R, G, and B color channels)
#   - 10 distinct categories:
#       - Airplanes, cars, ships, trucks 
#       - Birds, cats, deer, dogs, frogs, horses
# Training data is what the model will use to train itself
trainingData = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=torchvision.transforms.ToTensor())
# Test data is what we will test the model on once it is trained
testData = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=torchvision.transforms.ToTensor())
# This program is designed to be fully CPU-bound. For GPU acceleration, modify the device parameter
device = "cpu"
# Default batch size
batchSize = 64
# Actually load the data
trainingDataLoader = torch.utils.data.DataLoader(dataset=trainingData, batch_size=batchSize)
testDataLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=batchSize)

print("Loading data...")
for x, y in testDataLoader:
    # Expected outputs:
    # [x,y,z,w] = torch.Size([64, 3, 32, 32])
    #   - x = Batch size
    #   - y = Num. color channels (RGB=3)
    #   - z, w = Image dimensions
    print(f"Shape of Tensor X [x, y, z, w]: {x.shape}")
    # sizeof(int64) = 64 bits
    print(f"Shape of Tensor Y [size, type]: {y.shape}, {y.dtype}")
    break
# Could change device parameter to support CUDA or GPU acceleration
print(f"Using {device} as compute accelerator...")

# Deep learning model object, inherits PyTorch's Neural Network superclass
class CIFAR10_NN (torch.nn.Module):
    def __init__(self):
        super()

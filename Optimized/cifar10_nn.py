import torch, torchvision

# Augmented tensor transform
# - Randomly tweaks color, perspective, and orientation of each image
augmentedTransform = torchvision.transforms.Compose([
    torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor()
])

# Load training and test data
# We're using CIFAR-10:
#   - 32x32 full-color images (R, G, and B color channels)
#   - 10 distinct categories:
#       - Airplanes, cars, ships, trucks 
#       - Birds, cats, deer, dogs, frogs, horses
# Training data is what the model will use to train itself
trainingData = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=augmentedTransform)
# Test data is what we will test the model on once it is trained
testData = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=torchvision.transforms.ToTensor())
# Default batch size, or how many images to process in each training step

# Deep learning model object, inherits PyTorch's Neural Network superclass
class CIFAR10_NN (torch.nn.Module):
    def __init__(self, batchSize: int, device, iterations: int, learningRate: float):
        super().__init__()
        # Initialize the DataLoaders for training
        self.trainingDataLoader = torch.utils.data.DataLoader(dataset=trainingData, batch_size=batchSize, shuffle=True)
        self.testDataLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=batchSize)
        # This program is designed to be fully CPU-bound. For GPU acceleration, modify the device parameter
        self.device = device
        # Number of iterations (epochs) to run the simulation through
        self.iterations = iterations
        # Rate of model learning (higher numbers = faster learning). Keep below 1!
        self.learningRate = learningRate
        # How many images to pass into the model at once
        self.batchSize = batchSize
        self.linear_relu_stack = torch.nn.Sequential(
            # This is the neural network's connections. Takes in every pixel's color and uses deep learning to sort it into 10 categories
            # It's also a modified version of the VGG16 format
            # - After each convolution layer, we zero 10% of the data
            # - Also clamp the outputs between 0-1 as well as make sure they all sum to 100%
            #   - i.e. turn outputs into a percentage confidence that the image is of a category
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(32*2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.1),

            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(64*2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.1),

            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(128*2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.1),

            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(256*2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.1),

            torch.nn.Flatten(),
            torch.nn.Linear(in_features=2048, out_features=2048),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2048, out_features=2048),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(in_features=2048, out_features=10),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
def printDataset(dataset: torch.utils.data.DataLoader):
    print("Printing dataset...")
    for x, y in dataset:
        # Expected outputs:
        # [x,y,z,w] = torch.Size([64, 3, 32, 32])
        #   - x = Batch size (64)
        #   - y = Num. color channels (RGB=3)
        #   - z, w = Image dimensions (32, 32)
        print(f"Shape of Tensor X [x, y, z, w]: {x.shape}")
        # Size = batch size (64)
        # Type = data type (64-bit integer)
        print(f"Shape of Tensor Y [size, type]: {y.shape}, {y.dtype}")
        break
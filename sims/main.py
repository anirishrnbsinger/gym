from diloco_sim import DilocoSimulator, DilocoSimulatorConfig
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Define standard transformation for MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Standard normalization for MNIST
])

# Load MNIST dataset
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Option: Define your own CNN model adjusted for MNIST (1 input channel)
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Output: (16, 14, 14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),  # Output: (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Output: (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # Flatten from (batch, 32, 7, 7) to (batch, 32*7*7)
        output = self.out(x)
        return output

# Create a configuration object with required parameters
config = DilocoSimulatorConfig(
    model_cls=CNN,  # Updated to use the new CNN defined above
    model_kwargs={"num_classes": 10},  # MNIST has 10 classes
    loss_fn=F.cross_entropy,
    train_dataset=train_dataset,
    optimizer_kwargs={"lr": 0.001},
    num_nodes=5,
    #optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.AdamW
    #batch_size: int = 16
    #eval_dataset: Optional[torch.utils.data.Dataset] = None
    #ckpt_interval = 2 #Optional[int] = None  # num of outersteps to save model
    #eval_iters: int = 400
    #save_dir: Optional[str] = None
    #num_epochs: int = 1
    #p_sparta: float = 0.0
    #cosine_anneal: bool = False
    #warmup_steps: int = 0
    #model_path: Optional[str] = None
    #diloco_interval: int = 500
    #outer_optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.SGD
    #outer_optimizer_kwargs: dict = field(default_factory=lambda: {"lr": 0.7, "nesterov": True, "momentum": 0.9})
    #max_local_step: Optional[int] = None
    #wandb_project: Optional[str] = None

    #self.max_local_step = (
    #    self.config.num_epochs * len(self.config.train_dataset) // (self.config.batch_size * self.config.num_nodes)
    #)
    #if self.config.max_local_step:
    #   self.max_local_step = min(self.max_local_step, self.config.max_local_step)


)

if __name__ == "__main__":
    simulator = DilocoSimulator(config)
    simulator.train()


# Example CNN taken from https://medium.com/analytics-vidhya/ml15-56c033cc00e9
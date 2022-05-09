from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms


class MNIST:
    def __init__(self, path="mnist"):
        transform = transforms.Compose([transforms.ToTensor()])
        set = datasets.MNIST(path, train=True, download=True, transform=transform)
        self.loader = DataLoader(set, batch_size=128, shuffle=True)

    def get_loader(self):
        return self.loader

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def load_data(batch_size: int):
    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    return (
        DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
        ),
        DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
        ),
    )

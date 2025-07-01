import torch
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


def train_loop(
    dataloader: DataLoader,
    model: Module,
    loss_fn: Module,
    optimizer: Optimizer,
    device: str = "cpu",
):
    loss_sum = 0
    num_batches = len(dataloader)

    # turn into train mode
    model.train()
    for batch, (x, _) in enumerate(dataloader):
        # copy to gpu
        x = x.to(device)

        # reset gradients
        optimizer.zero_grad()

        # feed forward
        pred, mean, logvar = model(x)
        loss = loss_fn(pred, x, mean, logvar)  # predict input itself

        # backpropagation
        loss.backward()
        optimizer.step()

        # aggregate loss
        loss_sum += loss.item()

        if (batch + 1) % 100 == num_batches % 100:
            print(f"[{batch + 1:>3d}/{num_batches:>3d}] loss: {loss.item():>7f}")

    loss_avg = loss_sum / num_batches
    print(f"Train loss: {loss_avg:>8f}")

    return loss_avg


def test_loop(
    dataloader: DataLoader,
    model: Module,
    loss_fn: Module,
    device: str = "cpu",
):
    loss_sum = 0
    num_batches = len(dataloader)

    # turn into test mode
    model.eval()
    with torch.no_grad():  # temporarily disable auto gradient
        for x, _ in dataloader:
            x = x.to(device)

            # test
            pred, mean, logvar = model(x)
            loss = loss_fn(pred, x, mean, logvar)  # predict input itself

            loss_sum += loss.item()

    loss_avg = loss_sum / num_batches
    print(f"Test loss: {loss_avg:>8f}")

    return loss_avg

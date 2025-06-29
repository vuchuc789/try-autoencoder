import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from app.data import load_data
from app.model import Autoencoder
from app.time import Timer
from app.train import test_loop, train_loop


def main():
    # start a timer
    timer = Timer()

    # hyperparameters
    batch_size = 64
    learning_rate = 1e-3
    epochs = 20
    regularization_rate = 1e-5

    directory = "model"
    filename = "model"
    params_path = f"{directory}/{filename}_params.pth"
    history_path = f"{directory}/{filename}_history.npy"

    # get supported compute device
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using device: {device}\n")

    print("Loading data...")

    # load data to dataloaders (batched)
    train_dataloader, test_dataloader = load_data(batch_size=batch_size)

    # print time to load data
    timer.print()

    # create an autoencoder
    model = Autoencoder().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        # weight_decay=regularization_rate,
    )

    train_loss = np.array([])
    test_loss = np.array([])

    load_model = os.path.exists(params_path) and os.path.exists(history_path)

    if load_model:
        print("Loading model...")
        model.load_state_dict(torch.load(params_path, weights_only=True))
        with open(history_path, "rb") as f:
            train_loss = np.load(f)
            test_loss = np.load(f)

        test_loop(
            dataloader=test_dataloader,
            model=model,
            loss_fn=loss_fn,
            device=device,
        )

        # print time to test model
        timer.print()

    else:
        print("Training model...\n")
        for t in range(1, epochs + 1):
            print(f"Epoch: {t}")

            loss = train_loop(
                dataloader=train_dataloader,
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
            )
            train_loss = np.append(train_loss, loss)

            loss = test_loop(
                dataloader=test_dataloader,
                model=model,
                loss_fn=loss_fn,
                device=device,
            )
            test_loss = np.append(test_loss, loss)

            print()

        # print time to train model
        timer.print()

        print("Saving model...")
        torch.save(model.state_dict(), params_path)
        with open(history_path, "wb") as f:
            np.save(f, train_loss)
            np.save(f, test_loss)

        # print time to save model
        timer.print()

    print("Showing result...")

    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.show()

    imgs = None
    recon_imgs = None

    model.eval()
    with torch.no_grad():
        for x, _ in test_dataloader:
            imgs = np.append(imgs, x.numpy(), axis=0) if imgs is not None else x.numpy()

            x = x.to(device)
            pred = model(x)

            recon_imgs = (
                np.append(recon_imgs, pred.cpu().numpy(), axis=0)
                if recon_imgs is not None
                else pred.cpu().numpy()
            )

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(imgs[i][0])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(recon_imgs[i][0])
        plt.title("reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    timer.print()


if __name__ == "__main__":
    main()

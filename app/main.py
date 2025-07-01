import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch import nn

from app.data import load_data
from app.model import Autoencoder
from app.time import Timer
from app.train import test_loop, train_loop


def main():
    # start a timer
    timer = Timer()

    # hyperparameters
    batch_size = 128
    learning_rate = 1e-3
    epochs = 20
    regularization_rate = 1e-5

    directory = "model"
    filename = "vae_anomaly_detection"
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
    (
        train_dataloader,
        test_dataloader,
        normal_test_dataloader,
        anomalous_test_dataloader,
    ) = load_data(batch_size=batch_size)

    # print time to load data
    timer.print()

    # create an autoencoder
    model = Autoencoder().to(device)
    # loss_fn = nn.MSELoss()

    recon_loss_fn = nn.MSELoss()

    def loss_fn(pred, x, mean, logvar):
        recon_loss = recon_loss_fn(pred, x)
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return recon_loss + kl_div

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

    plt.plot(train_loss, label="Training Loss")
    plt.plot(test_loss, label="Validation Loss")
    plt.legend()
    plt.show()

    all_preds = None  # Initialize to None

    model.eval()
    with torch.no_grad():
        for x, _ in normal_test_dataloader:
            x = x.to(device)

            pred, _, _ = model(x)
            # Aggregate pred tensors into a single NumPy array
            if all_preds is None:
                all_preds = pred.cpu().numpy()
            else:
                all_preds = np.append(all_preds, pred.cpu().numpy(), axis=0)

    plt.plot(normal_test_dataloader.dataset[0][0].numpy(), "b")
    plt.plot(all_preds[0], "r")
    plt.fill_between(
        np.arange(140),
        all_preds[0],
        normal_test_dataloader.dataset[0][0],
        color="lightcoral",
    )
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    plt.show()

    all_preds = None  # Initialize to None

    model.eval()
    with torch.no_grad():
        for x, _ in anomalous_test_dataloader:
            x = x.to(device)

            pred, _, _ = model(x)
            # Aggregate pred tensors into a single NumPy array
            if all_preds is None:
                all_preds = pred.cpu().numpy()
            else:
                all_preds = np.append(all_preds, pred.cpu().numpy(), axis=0)

    plt.plot(anomalous_test_dataloader.dataset[0][0].numpy(), "b")
    plt.plot(all_preds[0], "r")
    plt.fill_between(
        np.arange(140),
        all_preds[0],
        anomalous_test_dataloader.dataset[0][0],
        color="lightcoral",
    )
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    plt.show()

    all_losses = np.array([])

    model.eval()
    with torch.no_grad():
        for x, _ in train_dataloader.dataset:
            x = x.to(device)

            pred, mean, logvar = model(x)
            loss = loss_fn(pred, x, mean, logvar)
            all_losses = np.append(all_losses, loss.cpu().numpy())

    plt.hist(all_losses, bins=50)
    plt.xlabel("Train loss")
    plt.ylabel("No of examples")
    plt.show()

    threshold = np.mean(all_losses) + np.std(all_losses)
    print("Threshold: ", threshold)

    all_losses = np.array([])

    model.eval()
    with torch.no_grad():
        for x, _ in anomalous_test_dataloader.dataset:
            x = x.to(device)

            pred, mean, logvar = model(x)
            loss = loss_fn(pred, x, mean, logvar)
            all_losses = np.append(all_losses, loss.cpu().numpy())

    plt.hist(all_losses, bins=50)
    plt.xlabel("Test loss")
    plt.ylabel("No of examples")
    plt.show()

    all_preds = np.array([])

    model.eval()
    with torch.no_grad():
        for x, _ in test_dataloader.dataset:
            x = x.to(device)

            pred, mean, logvar = model(x)
            loss = loss_fn(pred, x, mean, logvar)
            all_preds = np.append(all_preds, loss.cpu().numpy().item() < threshold)

    all_preds.astype(np.bool)
    all_labels = np.array([label for _, label in test_dataloader.dataset])
    print_stats(all_preds, all_labels)

    timer.print()


def print_stats(predictions, labels):
    print("Accuracy = {}".format(accuracy_score(labels, predictions)))
    print("Precision = {}".format(precision_score(labels, predictions)))
    print("Recall = {}".format(recall_score(labels, predictions)))


if __name__ == "__main__":
    main()

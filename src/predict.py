import torch.nn as nn
import torch
from  model import LSTMAutoEncoder
from config import config
import numpy as np
from trainer import Trainer
from main import get_test_loader, get_train_loader


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(
        "checkpoints/best_model.pt",
        map_location=device,
    )

    model = LSTMAutoEncoder(
        input_dim = config["input_dim"],
        hidden_dim= config["hidden_dim"],
        latent_dim= config["latent_dim"],
        num_layers= config["num_layers"],
        dropout= config["dropout"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    train_dataset, train_loader = get_train_loader()
    test_loaders  = get_test_loader()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loaders=test_loaders,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=100,
        train_dataset=train_dataset,
    )

    scores, labels, _ = trainer.predict(test_loaders["all"])

    scores = np.array(scores)
    labels = np.array(labels)

    


    
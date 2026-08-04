from RoADDataset import Dataset
from dataset_loader import RoadDataset
from torch.utils.data import DataLoader, ConcatDataset
import torch
import torch.nn as nn
from model import LSTMAutoEncoder
from trainer import Trainer
from utility import print_results
from config import config

def get_train_loader():
    road = Dataset(normalize=True)
    train_dataset = RoadDataset(
        road.sets["training"],
        window_size = config["window_size"],
        stride= config["stride"],
        has_labels=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size = config["batch_size"],
        shuffle=True
    )

    return train_dataset, train_loader


def get_test_loader():

    road = Dataset(normalize=True)

    collision_dataset = RoadDataset(
        road.sets["collision"],
        window_size=config["window_size"],
        stride=config["stride"],
        has_labels=True,
    )

    control_dataset = RoadDataset(
        road.sets["control"],
        window_size=config["window_size"],
        stride=config["stride"],
        has_labels=True,
    )

    velocity_dataset = RoadDataset(
        road.sets["velocity"],
        window_size=config["window_size"],
        stride=config["stride"],
        has_labels=True,
    )

    weight_dataset = RoadDataset(
        road.sets["weight"],
        window_size=config["window_size"],
        stride=config["stride"],
        has_labels=True,
    )


    collision_loader = DataLoader(
        collision_dataset,
        batch_size = config["batch_size"],
        shuffle=False,
    )

    control_loader = DataLoader(
        control_dataset,
        batch_size = config["batch_size"],
        shuffle=False,
    )

    velocity_loader = DataLoader(
        velocity_dataset,
        batch_size = config["batch_size"],
        shuffle=False,
    )

    weight_loader = DataLoader(
        weight_dataset,
        batch_size = config["batch_size"],
        shuffle=False,
    )


    all_test_dataset = ConcatDataset([
        collision_dataset,
        control_dataset,
        velocity_dataset,
        weight_dataset,
    ])

    all_test_loader = DataLoader(
        all_test_dataset,
        batch_size = config["batch_size"],
        shuffle=False,
    )

    test_loaders = {
        "collision": collision_loader,
        "control": control_loader,
        "velocity": velocity_loader,
        "weight": weight_loader,
        "all": all_test_loader,
    }

    return test_loaders

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMAutoEncoder(
        input_dim = config["input_dim"],
        hidden_dim= config["hidden_dim"],
        latent_dim= config["latent_dim"],
        num_layers= config["num_layers"],
        dropout= config["dropout"]
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = config["lr"]
    )

    train_dataset, train_loader = get_train_loader()
    test_loaders  = get_test_loader()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loaders=test_loaders,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=config["num_epochs"],
        train_dataset=train_dataset,
    )

    history = trainer.fit()
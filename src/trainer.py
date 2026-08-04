import torch
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
import os

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, average_precision_score

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        test_loaders,
        criterion,
        optimizer,
        device,
        num_epochs,
        train_dataset
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.train_dataset = train_dataset
        self.test_loaders = test_loaders

        self.history = {
            "train_loss": []
        }
        self.threshold = None
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.best_auroc = -float("inf")

    # ------------------------------------------------ #
    def _count_parameters(self):
        return sum(
            p.numel()
            for p in self.model.parameters()
            if p.requires_grad
        )


    def compute_train_scores(self):
        self.model.eval()
        scores = []
        with torch.no_grad():
            for batch in tqdm(
                self.train_loader,
                desc="Computing Training Scores",
                leave=False,
            ):
                batch = batch.to(self.device)
                reconstruction = self.model(batch)
                # error = (
                #     (batch - reconstruction) ** 2
                # ).mean(dim=(1, 2))

                # scores.extend(error.cpu().numpy())
                error = (batch - reconstruction) ** 2
                step_errors = error.mean(dim=2)
                window_scores = step_errors.max(dim=1).values
                scores.extend(window_scores.cpu().numpy())
        return np.array(scores)

    def compute_threshold(
        self,
        method="percentile",
        percentile=99,
        ):
        train_scores = self.compute_train_scores()
        if method == "percentile":
            threshold = np.percentile(
                train_scores,
                percentile
            )
        elif method == "3sigma":
            threshold = (
                train_scores.mean()
                + 3 * train_scores.std()
            )
        else:
            raise ValueError(
                f"Unknown threshold method: {method}"
            )
        self.threshold = threshold

        print("\nThreshold Computation")
        print("-" * 40)
        print(f"Method      : {method}")
        print(f"Threshold   : {threshold:.6f}")
        print("-" * 40)
        return threshold

    def save_checkpoint(
        self,
        epoch,
        train_loss,
        metrics,
    ):
        # Use the "all" loader if available; otherwise average all AUROCs
        if "all" in metrics:
            current_auroc = metrics["all"]["AUROC"]
        else:
            aurocs = []
            for m in metrics.values():

                if not np.isnan(m["AUROC"]):
                    aurocs.append(m["AUROC"])
            current_auroc = np.mean(aurocs)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": train_loss,
            "history": self.history,
            "threshold": self.threshold,
            "metrics": metrics,
            "model_name": self.model.__class__.__name__,
        }

        # Always save last checkpoint
        torch.save(
            checkpoint,
            os.path.join(
                self.checkpoint_dir,
                "last_model.pt"
            )
        )
        # Save best checkpoint
        if current_auroc > self.best_auroc:
            self.best_auroc = current_auroc
            checkpoint["best_auroc"] = current_auroc
            torch.save(
                checkpoint,
                os.path.join(
                    self.checkpoint_dir,
                    "best_model.pt"
                )
            )
            tqdm.write(
                f"✓ Best model saved "
                f"(AUROC = {current_auroc:.4f})"
            )

    def load_checkpoint(
        self,
        path,
    ):
        checkpoint = torch.load(
            path,
            map_location=self.device
        )
        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        self.history = checkpoint["history"]
        self.threshold = checkpoint["threshold"]
        self.best_auroc = checkpoint.get(
            "best_auroc",
            -float("inf")
        )
        start_epoch = checkpoint["epoch"] + 1
        print("=" * 60)
        print("Checkpoint Loaded")
        print("=" * 60)
        print(f"Epoch      : {checkpoint['epoch']}")
        print(f"Threshold  : {self.threshold:.6f}")
        print("=" * 60)
        return start_epoch

    # ------------------------------------------------ #
    def print_experiment_summary(self):

        print("=" * 70)
        print("               LSTM AUTOENCODER TRAINING")
        print("=" * 70)

        print(f"Model               : {self.model.__class__.__name__}")
        print(f"Device              : {self.device}")

        if torch.cuda.is_available():
            print(f"GPU                 : {torch.cuda.get_device_name(0)}")

        print(f"Epochs              : {self.num_epochs}")
        print(f"Batch Size          : {self.train_loader.batch_size}")

        print(
            f"Learning Rate       : "
            f"{self.optimizer.param_groups[0]['lr']}"
        )

        print(
            f"Optimizer           : "
            f"{self.optimizer.__class__.__name__}"
        )

        print(
            f"Loss Function       : "
            f"{self.criterion.__class__.__name__}"
        )

        print(
            f"Training Samples    : "
            f"{len(self.train_dataset):,}"
        )

        print(
            f"Training Batches    : "
            f"{len(self.train_loader):,}"
        )

        sample = self.train_dataset[0]

        print(f"Input Shape         : {tuple(sample.shape)}")
        print(f"Sequence Length     : {sample.shape[0]}")
        print(f"Input Features      : {sample.shape[1]}")

        print(
            f"Trainable Params    : "
            f"{self._count_parameters():,}"
        )

        print("=" * 70)
        print("Starting Training...")
        print("=" * 70)

    # ------------------------------------------------ #
    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        batch_bar = tqdm(
            self.train_loader,
            leave=False,
            desc="Batches"
        )

        for batch in batch_bar:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            reconstruction = self.model(batch)
            loss = self.criterion(
                reconstruction,
                batch
            )
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            batch_bar.set_postfix(
                loss=f"{loss.item():.6f}"
            )
        epoch_loss = running_loss / len(self.train_loader)
        return epoch_loss

    # ------------------------------------------------ #
    def fit(self, resume = False, checkpoint_path = None):
        start_epoch = 0
        if resume:
            start_epoch = self.load_checkpoint(
                checkpoint_path
            )
        epoch_bar = tqdm(
            range(start_epoch, self.num_epochs),
            desc="Training"
        )
        self.print_experiment_summary()
        for epoch in epoch_bar:
            train_loss = self.train_one_epoch()
            self.history["train_loss"].append(train_loss)
            epoch_bar.set_postfix(
                train_loss=f"{train_loss:.6f}"
            )
            should_log = (
                (epoch + 1) % 10 == 0
                or
                (epoch + 1) == self.num_epochs
            )
            if should_log:
                tqdm.write(
                    f"Epoch [{epoch+1:03d}/{self.num_epochs}] "
                    f"Train Loss: {train_loss:.6f}"
                )

            # Evaluate every 10 epochs
            should_evaluate = (
                (epoch + 1) % 10 == 0
                or
                (epoch + 1) == self.num_epochs
            )

            if should_evaluate:
                tqdm.write("\nEvaluation Results")
                tqdm.write("-" * 45)

                self.compute_threshold(
                    method="percentile",
                    percentile=99,
                )

                results = self.evaluate_all()

                for dataset, metrics in results.items():
                    tqdm.write(f"\n{dataset.upper()}")
                    tqdm.write(f"Accuracy      : {metrics['Accuracy']:.4f}")
                    tqdm.write(f"Precision     : {metrics['Precision']:.4f}")
                    tqdm.write(f"Recall        : {metrics['Recall']:.4f}")
                    tqdm.write(f"F1 Score      : {metrics['F1']:.4f}")
                    tqdm.write(f"AUROC         : {metrics['AUROC']:.4f}")
                    tqdm.write(f"AUPRC         : {metrics['AUPRC']:.4f}")

                tqdm.write("-" * 45)

                # Save checkpoint after evaluation
                self.save_checkpoint(
                    epoch=epoch + 1,
                    train_loss=train_loss,
                    metrics=results,
                )

        print("\nTraining Finished!")
        return self.history

    def predict(self, loader):
        self.model.eval()

        scores = []
        labels = []
        actions = []

        with torch.no_grad():
            for x, y, a in loader:
                x = x.to(self.device)
                reconstruction = self.model(x)

                # error = ((x - reconstruction) ** 2).mean(dim=(1,2))
                error = (x - reconstruction) ** 2
                step_errors = error.mean(dim=2)
                window_scores = step_errors.max(dim=1).values
                scores.extend(window_scores.cpu().numpy())

                # scores.extend(error.cpu().numpy())
                labels.extend(y.numpy())
                actions.extend(a.numpy())
        return np.array(scores), np.array(labels), np.array(actions)

    def evaluate_all(self):

        results = {}
        for anomaly_name, loader in self.test_loaders.items():
            results[anomaly_name] = self.evaluate(loader)

        return results


    def evaluate(self, test_loader):
        self.model.eval()

        scores = []
        labels = []

        with torch.no_grad():
            for x, y, _ in test_loader:
                x = x.to(self.device)
                reconstruction = self.model(x)

                # error = ((x - reconstruction) ** 2).mean(dim=(1, 2))
                error = (x - reconstruction) ** 2
                # Average over features
                step_errors = error.mean(dim=2)
                # Maximum timestep error
                window_scores = step_errors.max(dim=1).values

                scores.extend(window_scores.cpu().numpy())
                labels.extend(y.numpy())

        scores = np.array(scores)
        labels = np.array(labels)

        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]

        print(f"Normal Mean   : {normal_scores.mean():.6f}")
        print(f"Anomaly Mean  : {anomaly_scores.mean():.6f}")

        print(f"Normal Std    : {normal_scores.std():.6f}")
        print(f"Anomaly Std   : {anomaly_scores.std():.6f}")

        # Binary predictions
        predictions = (scores > self.threshold).astype(int)

        # print(np.unique(labels, return_counts=True))

        tn, fp, fn, tp = confusion_matrix(
            labels,
            predictions
        ).ravel()

        return {

            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,

            "Accuracy": accuracy_score(
                labels,
                predictions,
            ),

            "Precision": precision_score(
                labels,
                predictions,
                zero_division=0,
            ),

            "Recall": recall_score(
                labels,
                predictions,
                zero_division=0,
            ),

            "F1": f1_score(
                labels,
                predictions,
                zero_division=0,
            ),

            "Cohen Kappa": cohen_kappa_score(
                labels,
                predictions,
            ),

            "AUROC": roc_auc_score(
                labels,
                scores,
            ),

            "AUPRC": average_precision_score(
                labels,
                scores,
            ),

            "Threshold": self.threshold,
            "Samples": len(labels),
        }

    def predict_all(self):
        results = {}
        for name, loader in self.test_loaders.items():
            scores, labels = self.predict(loader)
            results[name] = {
                "scores": scores,
                "labels": labels
            }
        return results
"""
trainer.py

Generic trainer that works with every model.
"""

import copy
import time

import torch
import torch.nn as nn

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
)

from config import (
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    MIN_IMPROVEMENT,
)


class Trainer:

    def __init__(self, model, device):

        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        self.criterion = nn.CrossEntropyLoss()

    # -------------------------------------------------

    def _move_to_device(self, data):
        return data.to(self.device)

    # -------------------------------------------------

    def train(self, data):

        data = self._move_to_device(data)

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        best_state = None
        best_loss = float("inf")
        patience = 0

        start = time.time()

        for epoch in range(EPOCHS):

            # -------------------------
            # Training
            # -------------------------

            self.model.train()

            self.optimizer.zero_grad()

            logits = self.model(
                data.x,
                data.edge_index,
            )

            train_loss = self.criterion(
                logits[data.train_mask],
                data.y[data.train_mask],
            )

            train_loss.backward()

            self.optimizer.step()

            # -------------------------
            # Validation
            # -------------------------

            val_loss, val_acc = self.evaluate(
                data,
                data.val_mask,
            )

            history["train_loss"].append(
                train_loss.item()
            )

            history["val_loss"].append(
                val_loss
            )

            history["val_accuracy"].append(
                val_acc
            )

            print(
                f"Epoch {epoch+1:03d} | "
                f"Train {train_loss:.4f} | "
                f"Val {val_loss:.4f} | "
                f"Val Acc {val_acc:.4f}"
            )

            # -------------------------
            # Early Stopping
            # -------------------------

            if best_loss - val_loss > MIN_IMPROVEMENT:

                best_loss = val_loss
                patience = 0

                best_state = copy.deepcopy(
                    self.model.state_dict()
                )

            else:

                patience += 1

                if patience >= EARLY_STOPPING_PATIENCE:

                    print("\nEarly stopping triggered.")

                    break

        training_time = time.time() - start

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # ------------------------------------
        # Collect everything after training
        # ------------------------------------

        metrics = self.test(data)

        embeddings = self.get_embeddings(data)

        return {

            "history": history,

            "metrics": metrics,

            "embeddings": embeddings,

            "training_time": training_time,

        }

    # -------------------------------------------------

    def evaluate(self, data, mask):

        data = self._move_to_device(data)

        self.model.eval()

        with torch.no_grad():

            logits = self.model(
                data.x,
                data.edge_index,
            )

        loss = self.criterion(
            logits[mask],
            data.y[mask],
        )

        predictions = logits.argmax(dim=1)

        accuracy = MulticlassAccuracy(
            num_classes=logits.shape[1]
        ).to(self.device)

        acc = accuracy(
            predictions[mask],
            data.y[mask],
        )

        return loss.item(), acc.item()

    # -------------------------------------------------

    def test(self, data):

        data = self._move_to_device(data)

        self.model.eval()

        with torch.no_grad():

            logits = self.model(
                data.x,
                data.edge_index,
            )

        predictions = logits.argmax(dim=1)

        accuracy_metric = MulticlassAccuracy(
            num_classes=logits.shape[1]
        ).to(self.device)

        confusion_metric = MulticlassConfusionMatrix(
            num_classes=logits.shape[1]
        ).to(self.device)

        # ------------------------------
        # Test predictions only
        # ------------------------------

        test_predictions = predictions[data.test_mask]

        test_labels = data.y[data.test_mask]

        accuracy = accuracy_metric(
            test_predictions,
            test_labels,
        )

        confusion = confusion_metric(
            test_predictions,
            test_labels,
        )

        return {

            "accuracy": accuracy.item(),

            "confusion_matrix": confusion.cpu(),

            # Entire graph
            "logits": logits.cpu(),

            # Test subset only
            "predictions": test_predictions.cpu(),

            "labels": test_labels.cpu(),

            # Useful later for explainability
            "test_nodes": data.test_mask.nonzero(
                as_tuple=True
            )[0].cpu(),

        }

    # -------------------------------------------------

    def get_embeddings(self, data):

        data = self._move_to_device(data)

        self.model.eval()

        with torch.no_grad():

            embeddings = self.model.get_embeddings(
                data.x,
                data.edge_index,
            )

        return embeddings.cpu()
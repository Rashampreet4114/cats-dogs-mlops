"""Train the baseline CNN on data/processed/{train,val,test} and log the run
to MLflow (params, per-epoch metrics, confusion matrix, loss curve, model)."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from src.data_processing import CLASSES, load_split_dataset
from src.model import SimpleCNN

ROOT = Path(__file__).resolve().parent.parent


def evaluate(model, loader, device, criterion):
    """Returns (avg_loss, accuracy, labels, preds, probs) for one full pass over loader."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += images.size(0)
            all_preds.extend(preds.cpu().numpy().flatten().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
            all_probs.extend(probs.cpu().numpy().flatten().tolist())
    return total_loss / total, correct / total, all_labels, all_preds, all_probs


def train(
    data_dir: str = "data/processed",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 5e-5,
    early_stop_patience: int = 5,
    model_out: str = "models/model.pt",
    mlflow_tracking_uri: str = "file:./mlruns",
    experiment_name: str = "cats-dogs-cnn",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = load_split_dataset(data_dir, "train", augment=True)
    val_ds = load_split_dataset(data_dir, "val", augment=False)
    test_ds = load_split_dataset(data_dir, "test", augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Halve the LR once val_loss stalls for 2 epochs, so descent stays smooth instead
    # of oscillating once the model gets close to a minimum.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    criterion = nn.BCEWithLogitsLoss()

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "early_stop_patience": early_stop_patience,
                "lr_scheduler": "ReduceLROnPlateau(factor=0.5,patience=2)",
                "train_size": len(train_ds),
                "val_size": len(val_ds),
                "test_size": len(test_ds),
                "model": "SimpleCNN",
            }
        )

        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        stopped_epoch = epochs

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        lrs = []
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss, running_correct, n = 0.0, 0, 0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                running_correct += (preds == labels).sum().item()
                n += images.size(0)

            train_loss = running_loss / n
            train_acc = running_correct / n
            val_loss, val_acc, _, _, _ = evaluate(model, val_loader, device, criterion)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            lrs.append(current_lr)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "lr": current_lr,
                },
                step=epoch,
            )
            print(
                f"epoch {epoch}/{epochs} - train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} train_acc={train_acc:.4f} "
                f"val_acc={val_acc:.4f} lr={current_lr:.6f}"
            )

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stop_patience:
                    stopped_epoch = epoch
                    print(
                        f"early stopping at epoch {epoch} "
                        f"(no val_loss improvement for {early_stop_patience} epochs)"
                    )
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        mlflow.log_metric("stopped_epoch", stopped_epoch)

        test_loss, test_acc, test_labels, test_preds, test_probs = evaluate(
            model, test_loader, device, criterion
        )
        test_auc = roc_auc_score(test_labels, test_probs)
        test_avg_precision = average_precision_score(test_labels, test_probs)
        mlflow.log_metrics(
            {
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "test_auc": test_auc,
                "test_avg_precision": test_avg_precision,
            }
        )
        print(
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
            f"test_auc={test_auc:.4f} test_avg_precision={test_avg_precision:.4f}"
        )

        reports_dir = ROOT / "reports"
        completed_epochs = range(1, len(train_losses) + 1)

        # Loss curve
        loss_curve_path = reports_dir / "loss_curve.png"
        plt.figure()
        plt.plot(completed_epochs, train_losses, label="train_loss")
        plt.plot(completed_epochs, val_losses, label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Training / Validation Loss")
        plt.legend()
        plt.savefig(loss_curve_path)
        plt.close()
        mlflow.log_artifact(str(loss_curve_path))

        # Accuracy curve - loss alone can look fine while accuracy tells a different story
        acc_curve_path = reports_dir / "accuracy_curve.png"
        plt.figure()
        plt.plot(completed_epochs, train_accs, label="train_accuracy")
        plt.plot(completed_epochs, val_accs, label="val_accuracy")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title("Training / Validation Accuracy")
        plt.legend()
        plt.savefig(acc_curve_path)
        plt.close()
        mlflow.log_artifact(str(acc_curve_path))

        # Learning rate schedule - shows exactly when ReduceLROnPlateau kicked in
        lr_curve_path = reports_dir / "lr_schedule.png"
        plt.figure()
        plt.plot(completed_epochs, lrs)
        plt.xlabel("epoch")
        plt.ylabel("learning rate")
        plt.title("Learning Rate Schedule")
        plt.yscale("log")
        plt.savefig(lr_curve_path)
        plt.close()
        mlflow.log_artifact(str(lr_curve_path))

        # Confusion matrix - raw counts
        cm = confusion_matrix(test_labels, test_preds)
        cm_path = reports_dir / "confusion_matrix.png"
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.title("Confusion Matrix (test set, counts)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks([0, 1], CLASSES)
        plt.yticks([0, 1], CLASSES)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.colorbar()
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(str(cm_path))

        # Confusion matrix - row-normalized (per-class error rate, easier to compare classes)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm_path = reports_dir / "confusion_matrix_normalized.png"
        plt.figure()
        plt.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        plt.title("Confusion Matrix (test set, row-normalized)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks([0, 1], CLASSES)
        plt.yticks([0, 1], CLASSES)
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                plt.text(j, i, f"{cm_norm[i, j]:.1%}", ha="center", va="center")
        plt.colorbar()
        plt.savefig(cm_norm_path)
        plt.close()
        mlflow.log_artifact(str(cm_norm_path))

        # ROC curve
        fpr, tpr, _ = roc_curve(test_labels, test_probs)
        roc_path = reports_dir / "roc_curve.png"
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {test_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (test set)")
        plt.legend()
        plt.savefig(roc_path)
        plt.close()
        mlflow.log_artifact(str(roc_path))

        # Precision-Recall curve
        precision_vals, recall_vals, _ = precision_recall_curve(test_labels, test_probs)
        pr_path = reports_dir / "precision_recall_curve.png"
        plt.figure()
        plt.plot(recall_vals, precision_vals, label=f"AP = {test_avg_precision:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve (test set, positive class = dog)")
        plt.legend()
        plt.savefig(pr_path)
        plt.close()
        mlflow.log_artifact(str(pr_path))

        # Classification report - per-class precision/recall/F1, both as an artifact
        # and as individual MLflow metrics so runs are comparable at a glance.
        report_dict = classification_report(
            test_labels, test_preds, target_names=CLASSES, output_dict=True, zero_division=0
        )
        report_path = reports_dir / "classification_report.json"
        report_path.write_text(json.dumps(report_dict, indent=2))
        mlflow.log_artifact(str(report_path))
        for cls in CLASSES:
            mlflow.log_metrics(
                {
                    f"{cls}_precision": report_dict[cls]["precision"],
                    f"{cls}_recall": report_dict[cls]["recall"],
                    f"{cls}_f1": report_dict[cls]["f1-score"],
                }
            )

        # Save model + metadata
        torch.save(model.state_dict(), model_out)
        mlflow.log_artifact(model_out)

        metadata = {
            "run_id": run.info.run_id,
            "classes": CLASSES,
            "img_size": 224,
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "test_auc": test_auc,
            "test_avg_precision": test_avg_precision,
            "epochs": epochs,
            "stopped_epoch": stopped_epoch,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
        }
        metadata_path = Path(model_out).parent / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
        mlflow.log_artifact(str(metadata_path))

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--model-out", default="models/model.pt")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_stop_patience=args.early_stop_patience,
        model_out=args.model_out,
    )

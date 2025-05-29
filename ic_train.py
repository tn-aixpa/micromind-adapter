import torch
import torch.nn as nn
from ic_prepare_data import create_loaders, setup_mixup
from timm.loss import (
    BinaryCrossEntropy,
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
)

import micromind as mm
from micromind.networks import PhiNet, XiNet
from micromind.utils import parse_configuration

import sys
import os
from pathlib import Path

import digitalhub as dh

class ImageClassification(mm.MicroMind):
    """Implements an image classification class. Provides support
    for timm augmentation and loss functions."""

    def __init__(self, hparams, *args, **kwargs):
        super().__init__(hparams, *args, **kwargs)

        if hparams.model == "phinet":
            self.modules["classifier"] = PhiNet(
                input_shape=hparams.input_shape,
                alpha=hparams.alpha,
                num_layers=hparams.num_layers,
                beta=hparams.beta,
                t_zero=hparams.t_zero,
                compatibility=False,
                divisor=hparams.divisor,
                downsampling_layers=hparams.downsampling_layers,
                return_layers=hparams.return_layers,
                # classification-specific
                include_top=True,
                num_classes=hparams.num_classes,
            )
        elif hparams.model == "xinet":
            self.modules["classifier"] = XiNet(
                input_shape=hparams.input_shape,
                alpha=hparams.alpha,
                gamma=hparams.gamma,
                num_layers=hparams.num_layers,
                return_layers=hparams.return_layers,
                # classification-specific
                include_top=True,
                num_classes=hparams.num_classes,
            )

        self.mixup_fn, _ = setup_mixup(hparams)

        print("Number of parameters for each module:")
        print(self.compute_params())

        print("Number of MAC for each module:")
        print(self.compute_macs(hparams.input_shape))

    def setup_criterion(self):
        """Setup of the loss function based on augmentation strategy."""
        # setup loss function
        if (
            self.hparams.mixup > 0
            or self.hparams.cutmix > 0.0
            or self.hparams.cutmix_minmax is not None
        ):
            # smoothing is handled with mixup target transform which outputs sparse,
            # soft targets
            if self.hparams.bce_loss:
                train_loss_fn = BinaryCrossEntropy(
                    target_threshold=self.hparams.bce_target_thresh
                )
            else:
                train_loss_fn = SoftTargetCrossEntropy()
        elif self.hparams.smoothing:
            if self.hparams.bce_loss:
                train_loss_fn = BinaryCrossEntropy(
                    smoothing=self.hparams.smoothing,
                    target_threshold=self.hparams.bce_target_thresh,
                )
            else:
                train_loss_fn = LabelSmoothingCrossEntropy(
                    smoothing=self.hparams.smoothing
                )
        else:
            train_loss_fn = nn.CrossEntropyLoss()

        return train_loss_fn

    def forward(self, batch):
        """Computes forward step for image classifier.

        Arguments
        ---------
        batch : List[torch.Tensor, torch.Tensor]
            Batch containing the images and labels.

        Returns
        -------
        Predicted class and augmented class. : Tuple[torch.Tensor, torch.Tensor]
        """
        img, target = batch
        if not self.hparams.prefetcher:
            img, target = img.to(self.device), target.to(self.device)
            if self.mixup_fn is not None:
                img, target = self.mixup_fn(img, target)

        return (self.modules["classifier"](img), target)

    def compute_loss(self, pred, batch):
        """Sets up the loss function and computes the criterion.

        Arguments
        ---------
        pred : Tuple[torch.Tensor, torch.Tensor]
            Predicted class and augmented class.
        batch : List[torch.Tensor, torch.Tensor]
            Same batch as input to the forward step.

        Returns
        -------
        Cost function. : torch.Tensor
        """
        self.criterion = self.setup_criterion()

        # taking it from pred because it might be augmented
        return self.criterion(pred[0], pred[1])

    def configure_optimizers(self):
        """Configures the optimizes and, eventually the learning rate scheduler."""
        opt = torch.optim.Adam(self.modules.parameters(), lr=3e-4, weight_decay=0.0005)
        return opt


def top_k_accuracy(k=1):
    """
    Computes the top-K accuracy.

    Arguments
    ---------
    k : int
       Number of top elements to consider for accuracy.

    Returns
    -------
        accuracy : Callable
            Top-K accuracy.
    """

    def acc(pred, batch):
        if pred[1].ndim == 2:
            target = pred[1].argmax(1)
        else:
            target = pred[1]
        _, indices = torch.topk(pred[0], k, dim=1)
        correct = torch.sum(indices == target.view(-1, 1))
        accuracy = correct.item() / target.size(0)

        return torch.Tensor([accuracy]).to(pred[0].device)

    return acc


def get_latest_checkpoint(exp_folder):
    save_path = exp_folder + "/save"
    p = Path(save_path)
    subdirs = [d for d in p.iterdir() if d.is_dir()]
    if not subdirs:
        return save_path
    latest_subdir = max(subdirs, key=lambda d: d.stat().st_mtime)
    return str(latest_subdir)


def train(context, conf_name: str, dataset: str, data_dir: str, epochs: int):
    project = context.project
    #project = dh.get_or_create_project("micromind-image-classification")

    if(data_dir.endswith("/")):
        data_dir = data_dir.rstrip(data_dir[-1])

    hparams = parse_configuration("/shared/cfg/image_classification/" + conf_name)
    #hparams = parse_configuration("cfg/image_classification/" + conf_name)
    hparams.dataset = dataset 
    hparams.data_dir = data_dir + "/dataset"
    hparams.output_folder = data_dir + "/model"
    hparams.epochs = epochs

    train_loader, val_loader = create_loaders(hparams)

    exp_folder = mm.utils.checkpointer.create_experiment_folder(
        hparams.output_folder, hparams.experiment_name
    )

    checkpointer = mm.utils.checkpointer.Checkpointer(
        exp_folder, hparams=hparams, key="loss"
    )

    mind = ImageClassification(hparams=hparams)

    top1 = mm.Metric("top1_acc", top_k_accuracy(k=1), eval_only=True)
    top5 = mm.Metric("top5_acc", top_k_accuracy(k=5), eval_only=True)

    mind.train(
        epochs=hparams.epochs,
        datasets={"train": train_loader, "val": val_loader},
        metrics=[top5, top1],
        checkpointer=checkpointer,
        debug=hparams.debug,
    )

    mind.test(datasets={"test": val_loader}, metrics=[top1, top5])

    model_path = get_latest_checkpoint(exp_folder) + "/state-dict.pth.tar"

    model = project.log_model(
        name="micromind-model-" + hparams.model,
        kind="model",
        source=model_path,
        algorithm=hparams.model,
        framework="pytorch",
        tags=["image_classification"],
    )

    #TODO log metrics
    #model.log_metric()

if __name__ == "__main__":
    train(None, "phinet.py", "torch/cifar10", "/home/dev/micromind/", 1)

"""TWIST ViT-S/16 on ImageNet-10 (Imagenette). 20 epochs, 1 GPU, no W&B.

Two-view TWIST with the paper's ViT settings for the loss (sharpness weight
0.4, diversity weight 1.0) and optimizer (AdamW, lr 3e-4 per 256 samples,
weight decay 0.06, 10 warmup epochs then cosine). The momentum encoder and
multi-crop of the paper's ViT recipe are not used.

Environment variables: ``MAX_EPOCHS`` (default 20; use 200 for the RESULTS.md
table), ``BASE_LR`` (3e-4) and ``SHARPNESS_WEIGHT`` (0.4).
"""

import os

from two_view import (
    attach_forward_and_optim,
    make_imagenette_data,
    standard_callbacks,
    standard_trainer,
)

import stable_pretraining as spt
from stable_pretraining.methods.twist import TWIST

WARMUP_EPOCHS = 10


def main():
    batch_size = 256
    max_epochs = int(os.environ.get("MAX_EPOCHS", 20))
    base_lr = float(os.environ.get("BASE_LR", 3e-4))
    sharpness_weight = float(os.environ.get("SHARPNESS_WEIGHT", 0.4))

    data = make_imagenette_data(batch_size=batch_size, num_workers=8)
    steps_per_epoch = len(data.train)

    module = TWIST(
        encoder_name="vit_small_patch16_224",
        projector_dims=(4096, 4096, 4096),
        sharpness_weight=sharpness_weight,
        diversity_weight=1.0,
    )
    # ViT-S/16: AdamW as in the paper's ViT recipe (LARS + lr 0.5 is the
    # ResNet50 recipe). Weight decay is held at the paper's starting value.
    attach_forward_and_optim(
        module,
        TWIST,
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": base_lr * batch_size / 256,
                "weight_decay": 0.06,
            },
            "scheduler": {
                "type": "LinearWarmupCosineAnnealing",
                "peak_step": WARMUP_EPOCHS * steps_per_epoch,
                "total_steps": max_epochs * steps_per_epoch,
            },
            "interval": "step",
        },
    )

    callbacks = standard_callbacks(module, embed_dim=module.embed_dim)
    trainer = standard_trainer(
        callbacks, max_epochs=max_epochs, log_name="twist-vits-inet10"
    )
    spt.Manager(trainer=trainer, module=module, data=data)()


if __name__ == "__main__":
    main()

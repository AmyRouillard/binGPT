"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import CfgNode as CN


class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = "auto"
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1  # only applied on matmul weights
        C.grad_norm_clip = 1.0

        C.val_batch_size = C.batch_size
        C.early_stopping_patience = 2

        return C

    def __init__(self, config, model, train_dataset, val_dataset=None):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0
        self.epoch_num = 0

        self.best_metric_val = float("inf")
        self.current_metric_val = float("inf")
        self.patience_counter = 0
        self.stop_training_flag = False
        self.best_model_state = None

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def _run_validation(self, val_loader):
        """Helper function to run validation."""
        self.model.eval()  # Set model to evaluation mode
        total_val_loss = 0.0
        total_val_samples = 0
        # Add other metrics if needed, e.g., correct_predictions = 0

        with torch.no_grad():  # Disable gradient calculations
            for batch in val_loader:
                batch = [t.to(self.device) for t in batch]
                x, y = batch
                logits, loss = self.model(x, y)  # Assuming model returns (logits, loss)
                total_val_loss += loss.item() * x.size(0)  # Weighted by batch size
                total_val_samples += x.size(0)
                # Example for accuracy:
                # _, predicted = torch.max(logits, 1)
                # correct_predictions += (predicted == y).sum().item()

        self.model.train()  # Set model back to training mode

        avg_val_loss = (
            total_val_loss / total_val_samples
            if total_val_samples > 0
            else float("nan")
        )
        print(
            f"Validation Loss: {avg_val_loss:.4f}; total_samples: {total_val_samples}; total_loss: {total_val_loss:.4f}"
        )

        val_metrics = {"val_loss": avg_val_loss}

        return val_metrics

    def _check_early_stopping(self, val_metrics):
        """Helper function to check for early stopping."""
        if self.val_dataset is not None or self.config.early_stopping_patience <= 0:
            return  # No validation or early stopping disabled

        self.current_metric_val = val_metrics.get("val_loss", float("nan"))

        improved = False
        if self.current_metric_val < self.best_metric_val:
            improved = True

        if improved:
            self.best_metric_val = self.current_metric_val
            self.patience_counter = 0

            # self.trigger_callbacks("on_best_model_achieved")
        else:
            self.patience_counter += 1

            if self.patience_counter >= self.config.early_stopping_patience:
                self.stop_training_flag = True

        # self.trigger_callbacks("on_validation_end", val_metrics=val_metrics)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            # sampler=torch.utils.data.RandomSampler(
            #     self.train_dataset, replacement=True, num_samples=int(1e10)
            # ),
            sampler=torch.utils.data.RandomSampler(
                self.train_dataset, replacement=False
            ),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        val_loader = None
        if self.val_dataset is not None:
            val_loader = DataLoader(
                self.val_dataset,
                shuffle=False,  # No need to shuffle validation data
                pin_memory=True,
                batch_size=config.val_batch_size,  # Use validation batch size
                num_workers=config.num_workers,
            )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:

                self.epoch_num += 1

                # --- Validation Check ---
                if val_loader is not None:
                    val_metrics = self._run_validation(val_loader)
                    # --- Early Stopping Check ---
                    self._check_early_stopping(val_metrics)
                    if self.stop_training_flag:
                        break  # Break from the main training loop

                self.trigger_callbacks("on_epoch_end")

                data_iter = iter(train_loader)
                batch = next(data_iter)

            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss = model(x, y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            self.trigger_callbacks("on_batch_end")
            self.iter_num += 1

            if (
                self.stop_training_flag
            ):  # Check after each iteration as well, e.g. if max_iters hit
                break

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break

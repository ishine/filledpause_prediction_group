import numpy as np
import torch
from torch import nn, optim
import pytorch_lightning as pl

# My library
from util.eval_util import calc_score_all, calc_score_each_filler

class MyLightningModel(pl.LightningModule):

    def __init__(
        self,
        model,
        fillers,
        train_filler_rate_dict=None,
        dev_filler_rate_dict=None,
        loss_weights=None,
        optimizer_name="Adam",
        optimizer_params=None,
        lr_scheduler_name="StepLR",
        lr_scheduler_params=None,
    ):

        super().__init__()

        self.model = model

        self.fillers = fillers
        self.train_filler_rate_dict = train_filler_rate_dict
        self.dev_filler_rate_dict = dev_filler_rate_dict

        if loss_weights:
            self.criterion = nn.CrossEntropyLoss(torch.Tensor(loss_weights))
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer_name=optimizer_name
        self.optimizer_params=optimizer_params
        self.lr_scheduler_name=lr_scheduler_name
        self.lr_scheduler_params=lr_scheduler_params

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_index):
        x, target = batch
        output = self.model(x)

        # Loss
        loss = self.criterion(output.transpose(1, -1), target.to(torch.long))

        # Logging
        train_logger = self.logger[0].experiment
        train_logger.add_scalar("Loss", loss, global_step=self.global_step)

        return {
            "loss": loss,
            "output": output.detach(),
            "target": target.detach(),
        }

    # def training_epoch_end(self, training_step_outputs):
    #     train_logger = self.logger[0].experiment

    #     # Scores
    #     epoch_outputs = []
    #     epoch_targets = []
    #     for output in training_step_outputs:
    #         for o in output["output"]:
    #             epoch_outputs.append(o)
    #         for o in output["target"]:
    #             epoch_targets.append(o)

    #     precision, recall, f_score, specificity = calc_score_all(epoch_outputs, epoch_targets)
    #     precision = np.nan if precision is None else precision
    #     recall = np.nan if recall is None else recall
    #     f_score = np.nan if f_score is None else f_score
    #     specificity = np.nan if specificity is None else specificity

    #     train_logger.add_scalar("Filler_position/precision", precision, global_step=self.global_step)
    #     train_logger.add_scalar("Filler_position/recall", recall, global_step=self.global_step)
    #     train_logger.add_scalar("Filler_position/f_score", f_score, global_step=self.global_step)
    #     train_logger.add_scalar("Filler_position/specificity", specificity, global_step=self.global_step)

    #     # Score and logging in each filler
    #     precision_word = 0
    #     recall_word = 0
    #     f_score_word = 0
    #     specificity_word = 0
    #     rate_sum = 0
    #     for i, filler in enumerate(self.fillers):
    #         filler_rate = self.train_filler_rate_dict[filler]
    #         rate_sum += filler_rate

    #         precision, recall, f_score, specificity = \
    #             calc_score_each_filler(epoch_outputs, epoch_targets, i+1)
    #         if precision is None:
    #             precision = np.nan
    #         elif torch.isnan(precision).sum() == 0:
    #             precision_word += precision * filler_rate
    #         if recall is None:
    #             recall = np.nan
    #         elif torch.isnan(recall).sum() == 0:
    #             recall_word += recall * filler_rate
    #         if f_score is None:
    #             f_score = np.nan
    #         elif torch.isnan(f_score).sum() == 0:
    #             f_score_word += f_score * filler_rate
    #         if specificity is None:
    #             specificity = np.nan
    #         elif torch.isnan(specificity).sum() == 0:
    #             specificity_word += specificity * filler_rate

    #         train_logger.add_scalar(f"{filler}/precision", precision, global_step=self.global_step)
    #         train_logger.add_scalar(f"{filler}/recall", recall, global_step=self.global_step)
    #         train_logger.add_scalar(f"{filler}/f_score", f_score, global_step=self.global_step)
    #         train_logger.add_scalar(f"{filler}/specificity", specificity, global_step=self.global_step)

    #     train_logger.add_scalar(
    #         "Filler_word/precision", precision_word / rate_sum, global_step=self.global_step)
    #     train_logger.add_scalar(
    #         "Filler_word/recall", recall_word / rate_sum, global_step=self.global_step)
    #     train_logger.add_scalar(
    #         "Filler_word/f_score", f_score_word / rate_sum, global_step=self.global_step)
    #     train_logger.add_scalar(
    #         "Filler_word/specificity", specificity_word / rate_sum, global_step=self.global_step)

    def validation_step(self, batch, batch_index):
        x, target = batch
        output = self.model(x)

        # Loss
        loss = self.criterion(output.transpose(1, -1), target.to(torch.long))
        self.log("val_loss", loss)

        return {
            "loss": loss,
            "output": output.detach(),
            "target": target.detach(),
        }

    # def validation_epoch_end(self, validation_step_outputs):
    #     val_logger = self.logger[1].experiment

    #     # Loss
    #     epoch_losses = []
    #     for output in validation_step_outputs:
    #         epoch_losses.append(output["loss"])
    #     epoch_loss = torch.mean(torch.Tensor(epoch_losses)) if len(epoch_losses) != 0 else np.nan
    #     val_logger.add_scalar("Loss", epoch_loss, global_step=self.global_step)

    #     # Scores
    #     epoch_outputs = []
    #     epoch_targets = []
    #     for output in validation_step_outputs:
    #         for o in output["output"]:
    #             epoch_outputs.append(o)
    #         for o in output["target"]:
    #             epoch_targets.append(o)

    #     precision, recall, f_score, specificity = calc_score_all(epoch_outputs, epoch_targets)
    #     precision = np.nan if precision is None else precision
    #     recall = np.nan if recall is None else recall
    #     f_score = np.nan if f_score is None else f_score
    #     specificity = np.nan if specificity is None else specificity

    #     val_logger.add_scalar("Filler_position/precision", precision, global_step=self.global_step)
    #     val_logger.add_scalar("Filler_position/recall", recall, global_step=self.global_step)
    #     val_logger.add_scalar("Filler_position/f_score", f_score, global_step=self.global_step)
    #     val_logger.add_scalar("Filler_position/specificity", specificity, global_step=self.global_step)

    #     # Score and logging in each filler
    #     precision_word = 0
    #     recall_word = 0
    #     f_score_word = 0
    #     specificity_word = 0
    #     rate_sum = 0
    #     for i, filler in enumerate(self.fillers):
    #         filler_rate = self.dev_filler_rate_dict[filler]
    #         rate_sum += filler_rate

    #         precision, recall, f_score, specificity = \
    #             calc_score_each_filler(epoch_outputs, epoch_targets, i+1)
    #         if precision is None:
    #             precision = np.nan
    #         elif torch.isnan(precision).sum() == 0:
    #             precision_word += precision * filler_rate
    #         if recall is None:
    #             recall = np.nan
    #         elif torch.isnan(recall).sum() == 0:
    #             recall_word += recall * filler_rate
    #         if f_score is None:
    #             f_score = np.nan
    #         elif torch.isnan(f_score).sum() == 0:
    #             f_score_word += f_score * filler_rate
    #         if specificity is None:
    #             specificity = np.nan
    #         elif torch.isnan(specificity).sum() == 0:
    #             specificity_word += specificity * filler_rate

    #         val_logger.add_scalar(f"{filler}/precision", precision, global_step=self.global_step)
    #         val_logger.add_scalar(f"{filler}/recall", recall, global_step=self.global_step)
    #         val_logger.add_scalar(f"{filler}/f_score", f_score, global_step=self.global_step)
    #         val_logger.add_scalar(f"{filler}/specificity", specificity, global_step=self.global_step)

    #     val_logger.add_scalar(
    #         "Filler_word/precision", precision_word / rate_sum, global_step=self.global_step)
    #     val_logger.add_scalar(
    #         "Filler_word/recall", recall_word / rate_sum, global_step=self.global_step)
    #     val_logger.add_scalar(
    #         "Filler_word/f_score", f_score_word / rate_sum, global_step=self.global_step)
    #     val_logger.add_scalar(
    #         "Filler_word/specificity", specificity_word / rate_sum, global_step=self.global_step)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        # this calls forward
        if len(batch) == 2:
            x, y = batch
            return {
                "predictions": self(x),
                "targets": y,
                "batch_idx": batch_idx,
            }
        elif len(batch) == 3:
            x, y, t = batch
            return {
                "predictions": self(x),
                "targets": y,
                "texts": t,
                "batch_idx": batch_idx,
            }

    def configure_optimizers(self):
        # Optimizer
        optimizer_class = getattr(optim, self.optimizer_name)
        optimizer = optimizer_class(
            self.parameters(), **self.optimizer_params
        )
        # lr scheduler
        lr_scheduler_class = getattr(optim.lr_scheduler, self.lr_scheduler_name)
        lr_scheduler = lr_scheduler_class(
            optimizer, **self.lr_scheduler_params
        )

        # return optimizer
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

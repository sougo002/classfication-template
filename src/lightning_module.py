from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torchmetrics

from factory import get_model, get_loss, get_lr_scheduler, get_optimizer

from utils.custom_logging import CustomLogger

logger = CustomLogger(__name__).get_logger()


class CustomLightningModule(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super(CustomLightningModule, self).__init__()
        self.cfg = cfg
        # TODO: model nameによって取得するモデルを変更する
        self.net = get_model(cfg.Model)
        self.cfg_optimizer = cfg.Model.optimizer
        self.cfg_lr_scheduler = cfg.Model.lr_scheduler
        self.loss = get_loss(cfg.Model.loss)
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        logits = self.net(x).squeeze(1)
        return logits

    def get_net(self):
        return self.net

    def training_step(self, batch, batch_index):
        img, targets = batch
        logits = self.forward(img)
        loss = self.loss(logits, targets)
        acc = self.accuracy(logits, targets)
        self.log('train loss', loss)
        self.log('train acc', acc)
        return loss

    def validation_step(self, batch, batch_index):
        img, targets = batch
        logits = self.forward(img)
        loss = self.loss(logits, targets)
        preds = logits.sigmoid()
        output = OrderedDict({"targets": targets.detach(), "preds": preds.detach(), "loss": loss.detach()})
        return output

    def validation_epoch_end(self, outputs):
        log_dict = dict()
        log_dict['epoch'] = int(self.current_epoch)
        log_dict['val_loss'] = torch.stack([output["loss"] for output in outputs]).mean().item()

        targets = torch.cat([output["targets"].view(-1) for output in outputs])
        preds = []
        for score in [output["preds"].view(-1, len(self.cfg.Model.classes)) for output in outputs]:
            preds.append(score)
        preds = torch.cat(preds)
        val_acc = self.accuracy(preds, targets)
        logger.info(f'epoch : {self.current_epoch},val acc : {val_acc}\n')
        log_dict['val_acc'] = val_acc

        targets = targets.cpu().numpy()
        preds = preds.cpu().numpy()

        self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self):
        cfg_optimizer = self.cfg_optimizer
        cfg_lr_scheduler = self.cfg_lr_scheduler
        optimizer = get_optimizer(cfg_optimizer)(self.parameters(), **cfg_optimizer.params)
        scheduler = get_lr_scheduler(cfg_lr_scheduler)(optimizer, **cfg_lr_scheduler.params)
        return [optimizer], [scheduler]

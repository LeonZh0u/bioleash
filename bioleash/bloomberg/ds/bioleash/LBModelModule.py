from bloomberg.ds.bioleash.chemberta import LMModel
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
import torch
from torchmetrics import AveragePrecision

class LBModelModule(L.LightningModule):
    def __init__(self, model_name):
        super().__init__()
        self.model = LMModel(model_name)
        self.map = AveragePrecision(task="binary")

    def forward(self, batch):
        return self.model(batch)

    def calculate_loss(self, batch, batch_idx):
        return self.model.calculate_loss(batch)

    def training_step(self, batch, batch_idx):
        ret = self.calculate_loss(batch, batch_idx)
        self.log("train_loss", ret["loss"], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return ret["loss"]

    def validation_step(self, batch, batch_idx):
        ret = self.calculate_loss(batch, batch_idx)
        self.log("val_loss", ret["loss"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.map.update(F.sigmoid(ret["logits"]), batch["labels"].long())

    def on_validation_epoch_end(self):
        val_map = self.map.compute()
        self.log("val_map", val_map, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.map.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self.forward(batch)["logits"]
        probs = F.sigmoid(logits)
        return probs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return {
            "optimizer": optimizer,
        }

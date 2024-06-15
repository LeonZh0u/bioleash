from bloomberg.ds.bioleash.LMDataset import LMDataset
import lightning as L
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader

class LMDataModule(L.LightningDataModule):
    def __init__(self, train_df, val_df, test_df, tokenizer, config):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.config = config

    def _generate_dataset(self, stage):
        if stage == "train":
            df = self.train_df
        elif stage == "val":
            df = self.val_df
        elif stage == "test":
            df = self.test_df
        else:
            raise NotImplementedError
        dataset = LMDataset(df, self.tokenizer, stage=stage)
        return dataset

    def _generate_dataloader(self, stage):
        dataset = self._generate_dataset(stage)
        shuffle=False
        drop_last=False
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=True,
            collate_fn=DataCollatorWithPadding(self.tokenizer),
            num_workers=self.config.num_workers,
        )

    def train_dataloader(self):
        return self._generate_dataloader("train")

    def val_dataloader(self):
        return self._generate_dataloader("val")

    def test_dataloader(self):
        return self._generate_dataloader("test")

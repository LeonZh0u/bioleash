from torch.utils.data import IterableDataset
import numpy as np
from bloomberg.ds.bioleash.config import *

def tokenize(batch, tokenizer):
    output = tokenizer(batch["molecule_smiles"], truncation=True)
    return output


class LMDataset(IterableDataset):
    def __init__(self, df, tokenizer, stage="train"):
        super().__init__()
        assert stage in ["train", "val", "test"]
        print(stage)
        self.tokenizer = tokenizer
        self.stage = stage
        self.df_iter = (
            df
            .map(tokenize, batched=True, fn_kwargs={"tokenizer": self.tokenizer})
            .map(lambda item: {"label": 
                               np.array([0, 0, 0]) if stage=="test" else np.array([item["binds_BRD4"], item["binds_HSA"], item["binds_sEH"]])
                              }).remove_columns(["molecule_smiles"] if stage=="test" else ["binds_BRD4", "binds_HSA", "binds_sEH","molecule_smiles", "binds"])
        )

    def __len__(self):
        return 98_415_610

    def __iter__(self):
        # data = self._generate_data(index)
        # data["label"] = self._generate_label(index)
        return iter(self.df_iter)        

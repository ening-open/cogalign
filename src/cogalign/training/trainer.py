import torch.distributed as dist
from typing import Optional

import torch
from transformers.trainer_utils import has_length
from trl import SFTTrainer

class SeqSFTTrainer(SFTTrainer):
    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return self._get_dataloader(
            dataset=self.train_dataset,
            description="Training",
            batch_size=self._train_batch_size,
            sampler_fn=self._get_sequence_sampler,
            is_training=True,
        )

    def _get_sequence_sampler(self, train_dataset: Optional[torch.utils.data.Dataset] = None) -> Optional[torch.utils.data.Sampler]:
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None or not has_length(train_dataset):
            return None

        print("*"*16)
        print("Training with Sequential Sampler!!!")
        print("*"*16)

        return torch.utils.data.SequentialSampler(train_dataset)

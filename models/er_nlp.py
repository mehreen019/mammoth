"""
Experience Replay for NLP tasks (specifically for Hindi->Bangla NER)
Adapted from the standard ER model to handle BERT tokenized inputs.

Example usage:
    python main.py --model er_nlp --dataset seq-hindi-bangla-ner --buffer_size 200 --batch_size 16 --n_epochs 3
"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


class ErNlp(ContinualModel):
    """
    Experience Replay adapted for NLP tasks.
    Handles BERT-style tokenized inputs with attention masks.
    """
    NAME = 'er_nlp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        """
        Returns an ArgumentParser object with predefined arguments for the ErNlp model.
        """
        add_rehearsal_args(parser)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """
        NLP version of Experience Replay.
        Buffer stores tokenized sequences with attention masks.
        """
        super(ErNlp, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)

    def forward(self, x):
        """
        Forward pass through the BERT backbone.
        x can be: (input_ids, attention_mask) tuple or just input_ids tensor
        """
        if isinstance(x, tuple):
            input_ids, attention_mask = x
            return self.net((input_ids, attention_mask))
        else:
            return self.net(x)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        ER trains on the current task data, augmented with buffer samples.

        Args:
            inputs: Tokenized input_ids (batch_size, seq_len)
            labels: Sentence-level labels (batch_size,)
            not_aug_inputs: Attention masks (batch_size, seq_len)
        """
        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()

        # Combine current batch with buffer samples
        if not self.buffer.is_empty():
            # Get buffer samples
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size,
                transform=self.transform,
                device=self.device
            )

            # Concatenate current and buffer data
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

            # Handle attention masks (stored in not_aug_inputs during training)
            if not_aug_inputs is not None and isinstance(not_aug_inputs, torch.Tensor):
                # For buffer, create attention masks from input_ids
                buf_attention_mask = (buf_inputs != 0).long().to(self.device)
                attention_mask = torch.cat((not_aug_inputs, buf_attention_mask))
            else:
                attention_mask = (inputs != 0).long().to(self.device)
        else:
            # First task - no buffer samples
            if not_aug_inputs is not None and isinstance(not_aug_inputs, torch.Tensor):
                attention_mask = not_aug_inputs
            else:
                attention_mask = (inputs != 0).long().to(self.device)

        # Forward pass with BERT
        # Pass as tuple: (input_ids, attention_mask)
        outputs = self.net((inputs, attention_mask))

        # Compute loss
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        # Add current batch to buffer (only the original samples, not buffer samples)
        # Store input_ids only (attention masks will be reconstructed)
        self.buffer.add_data(
            examples=inputs[:real_batch_size],  # Original input_ids
            labels=labels[:real_batch_size]
        )

        return loss.item()

# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
BERT Multilingual Backbone for NLP tasks in Mammoth
Uses bert-base-multilingual-cased from HuggingFace
"""

import torch
import torch.nn as nn

from backbone import MammothBackbone, register_backbone

try:
    from transformers import AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@register_backbone('bert-multilingual')
class BERTMultilingualBackbone(MammothBackbone):
    """
    BERT Multilingual backbone for cross-lingual NER tasks.
    Uses bert-base-multilingual-cased model.
    """

    def __init__(self, n_classes: int):
        """
        Args:
            n_classes: Number of output classes
        """
        super(BERTMultilingualBackbone, self).__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required for BERT backbone. "
                "Install with: pip install transformers"
            )

        self.n_classes = n_classes

        # Load pretrained BERT model
        self.bert = AutoModel.from_pretrained('bert-base-multilingual-cased')

        # Classification head
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

        # Initialize classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, x, returnt='out'):
        """
        Forward pass through BERT and classification head.

        Args:
            x: Input can be:
                - dict with 'input_ids' and 'attention_mask' keys
                - tuple of (input_ids, attention_mask)
                - tensor of input_ids (attention_mask will be created)
            returnt: Return type ('out', 'features', 'both', 'all')

        Returns:
            Logits of shape (batch_size, n_classes)
        """
        # Handle different input formats
        if isinstance(x, dict):
            input_ids = x['input_ids']
            attention_mask = x.get('attention_mask', None)
        elif isinstance(x, tuple):
            input_ids = x[0]
            attention_mask = x[1] if len(x) > 1 else None
        else:
            input_ids = x
            attention_mask = None

        # Ensure tensors are on the same device as model
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        else:
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = (input_ids != 0).long()

        # Get BERT embeddings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Use [CLS] token representation (first token)
        features = outputs.last_hidden_state[:, 0, :]

        # Apply classification head
        logits = self.classifier(features)

        # Return based on returnt parameter
        if returnt == 'out':
            return logits
        elif returnt == 'features':
            return features
        elif returnt == 'both':
            return logits, features
        elif returnt == 'all':
            return logits, features
        else:
            raise ValueError(f"Invalid returnt value: {returnt}")

    def get_params(self):
        """Returns all trainable parameters"""
        return torch.nn.utils.parameters_to_vector(self.parameters())

    def set_params(self, new_params):
        """Sets all parameters from a vector"""
        torch.nn.utils.vector_to_parameters(new_params, self.parameters())

    def get_grads(self):
        """Returns all gradients concatenated"""
        grads = []
        for param in self.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
            else:
                grads.append(torch.zeros_like(param).view(-1))
        return torch.cat(grads)

    def freeze_bert(self):
        """Freeze BERT parameters (only train classifier)"""
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert(self):
        """Unfreeze BERT parameters"""
        for param in self.bert.parameters():
            param.requires_grad = True

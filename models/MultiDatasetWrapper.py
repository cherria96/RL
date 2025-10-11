import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiDatasetWrapper(nn.Module):
    def __init__(self, model_core: nn.Module, feature_dims: dict, shared_input_dim: int):
        """
        model_core: your existing Model(configs)
        feature_dims: dict mapping dataset_id to its input feature dim
        shared_input_dim: common input dim for the model (i.e., d_model)
        """
        super().__init__()
        self.model_core = model_core
        self.adapters = nn.ModuleDict({
            ds_id: nn.Linear(input_dim, shared_input_dim)
            for ds_id, input_dim in feature_dims.items()
        })

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, dataset_id,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # [B, L, input_dim] → [B, L, shared_input_dim]
        x_enc = self.adapters[dataset_id](x_enc)
        x_dec = self.adapters[dataset_id](x_dec)
        return self.model_core(
            x_enc, x_mark_enc, x_dec, x_mark_dec,
            enc_self_mask, dec_self_mask, dec_enc_mask
        )

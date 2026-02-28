import logging
from typing import Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig
from omegaconf.errors import MissingMandatoryValue

from src.data import SyntheticDatasetCollection
from src.models.time_varying_model import TimeVaryingCausalModel
from src.models.utils import OutcomeHead
from src.models.utils_pfn import (
    DataEmbedding_FeaturePatching,
    Encoder,
    EncoderLayer,
    AttentionLayer,
    FullAttention,
)

logger = logging.getLogger(__name__)


class DynamicCausalPFN(TimeVaryingCausalModel):
    """
    Baselines-repo compatible DynamicCausalPFN:
    - GT-style single model (model_type is a single key)
    - GT-style G-computation training logic (pseudo outcomes) when projection_horizon > 0
    - Standard masked MSE loss (like other baselines)

    IMPORTANT:
      For cancer_sim, processed sequences have length (dataset.max_seq_length - 1) due to offset=1 in process_data().
      We therefore infer seq_len directly from the processed dataset tensors to avoid time-dimension mismatches.
    """

    model_type = "dynamic_causal_pfn"
    possible_model_types = {"dynamic_causal_pfn"}
    tuning_criterion = "rmse"

    def __init__(
        self,
        args: DictConfig,
        dataset_collection: Union[SyntheticDatasetCollection] = None,
        autoregressive: bool = None,
        has_vitals: bool = None,
        projection_horizon: int = None,
        bce_weights: np.array = None,
        **kwargs,
    ):
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)

        sub_args = args.model[self.model_type]

        # --- projection horizon logic (same as GT) ---
        if projection_horizon is not None:
            self.projection_horizon = projection_horizon
        elif sub_args.projection_horizon is not None:
            self.projection_horizon = sub_args.projection_horizon
        elif self.dataset_collection is not None:
            self.projection_horizon = args.dataset.projection_horizon
        else:
            raise MissingMandatoryValue()

        # Used by GT g-computation under fixed treatment sequence intervention
        self.treatment_sequence = torch.tensor(args.dataset.treatment_sequence)[: self.projection_horizon + 1, :]
        self.max_projection = (
            args.dataset.projection_horizon if self.dataset_collection is not None else self.projection_horizon
        )
        assert self.projection_horizon <= self.max_projection

        # --- infer TRUE processed sequence length ---
        # train script typically calls dataset_collection.process_data_multi() BEFORE instantiating the model
        if self.dataset_collection is not None and hasattr(self.dataset_collection, "train_f") and self.dataset_collection.train_f is not None:
            # current_treatments time dim is the canonical time length in this repo
            self.seq_len = int(self.dataset_collection.train_f.data["current_treatments"].shape[1])
        else:
            # fallback (cancer_sim offset=1)
            self.seq_len = int(args.dataset.max_seq_length) - 1

        # --- repo tuning convention (same as GT) ---
        self.input_size = max(self.dim_treatments, self.dim_static_features, self.dim_vitals, self.dim_outcome)
        logger.info(f"Max input size of {self.model_type}: {self.input_size}")
        logger.info(f"[{self.model_type}] Using seq_len={self.seq_len} (processed time length).")

        self._init_specific(args)
        self.save_hyperparameters(args)

    def prepare_data(self) -> None:
        # Match GT: multi-input (prev_treatments, prev_outputs, vitals, static)
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_multi:
            self.dataset_collection.process_data_multi()

    def _init_specific(self, args: DictConfig) -> None:
        """
        PFN backbone + g-computation heads init.
        Must output per-time hidden representation hr: [B,T,hr_size] with T matching batch time dim.
        """
        try:
            sub_args = args.model[self.model_type]

            # Keep for logging/config, but PFN itself uses self.seq_len
            self.max_seq_length = sub_args.max_seq_length

            self.hr_size = sub_args.hr_size
            self.seq_hidden_units = sub_args.seq_hidden_units  # embed_dim
            self.fc_hidden_units = sub_args.fc_hidden_units
            self.dropout_rate = sub_args.dropout_rate

            self.patch_size = sub_args.patch_size
            self.n_heads = sub_args.n_heads
            self.d_ff = sub_args.d_ff
            self.e_layers = sub_args.e_layers
            self.activation = sub_args.activation

            if (
                self.hr_size is None
                or self.seq_hidden_units is None
                or self.fc_hidden_units is None
                or self.dropout_rate is None
                or self.patch_size is None
                or self.n_heads is None
                or self.d_ff is None
                or self.e_layers is None
            ):
                raise MissingMandatoryValue()

            # Input channels to PFN embedding: prev_treatments + prev_outputs + vitals(optional) + static(broadcast)
            # NOTE: For cancer_sim multi-input, prev_outputs has dim_outcome and static_features has dim_static_features.
            self.in_channels = self.dim_treatments + self.dim_outcome + self.dim_static_features
            if self.has_vitals:
                self.in_channels += self.dim_vitals

            L = self.seq_len  # <-- critical

            # 1) PFN embedding (expects x: [B,L,C])
            self.enc_embedding = DataEmbedding_FeaturePatching(
                seq_len=L,
                patch_size=self.patch_size,
                embed_dim=self.seq_hidden_units,
                dropout=self.dropout_rate,
            )

            # 2) Encoder-only transformer
            self.transformer_encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(False, attention_dropout=self.dropout_rate, output_attention=False),
                            self.seq_hidden_units,
                            self.n_heads,
                        ),
                        self.seq_hidden_units,
                        self.d_ff,
                        dropout=self.dropout_rate,
                        activation=self.activation,
                    )
                    for _ in range(self.e_layers)
                ],
                norm_layer=nn.LayerNorm(self.seq_hidden_units),
            )

            # 3) Patch tokens -> per-time representation
            n_patches = (L - self.patch_size) // (self.patch_size // 2) + 1
            self._n_patches = n_patches

            # After embedding we reshape to [B, in_channels, n_patches*D] then project to time length L
            self.proj_tokens_to_time = nn.Linear(n_patches * self.seq_hidden_units, L)

            # Map per-time channel features -> hr_size
            self.hr_output_transformation = nn.Linear(self.in_channels, self.hr_size)
            self.hr_dropout = nn.Dropout(self.dropout_rate)

            # 4) G-computation heads (exactly like GT)
            self.G_comp_heads = nn.ModuleList(
                [
                    OutcomeHead(
                        self.seq_hidden_units,  # kept consistent with other models
                        self.hr_size,
                        self.fc_hidden_units,
                        self.dim_treatments,
                        self.dim_outcome,
                    )
                    for _ in range(self.projection_horizon + 1)
                ]
            )

        except MissingMandatoryValue:
            logger.warning(
                f"{self.model_type} not fully initialised - some mandatory args are missing "
                f"(ok if doing hyperparameter search)."
            )

    def _build_series_input(self, prev_treatments, vitals, prev_outputs, static_features) -> torch.Tensor:
        """
        Build x: [B,T,C] on a discrete grid (same spirit as GT; no torchcde coefficients).
        """
        parts = [prev_treatments, prev_outputs]
        if self.has_vitals:
            parts.append(vitals)
        s = static_features.unsqueeze(1).expand(-1, prev_treatments.size(1), -1)  # [B,T,S]
        parts.append(s)
        x = torch.cat(parts, dim=-1)  # [B,T,in_channels]
        return x

    def build_hr(self, prev_treatments, vitals, prev_outputs, static_features, active_entries) -> torch.Tensor:
        """
        Produce per-time hidden representation hr: [B,T,hr_size].
        Ensures time dimension matches the batch time dimension for compatibility with OutcomeHead.
        """
        x = self._build_series_input(prev_treatments, vitals, prev_outputs, static_features)  # [B,T,C]
        B, T_batch, C = x.shape

        # PFN modules were built for self.seq_len. Pad/trim x to that length.
        L = self.seq_len
        if T_batch < L:
            pad = torch.zeros(B, L - T_batch, C, device=x.device, dtype=x.dtype)
            x_in = torch.cat([x, pad], dim=1)
        elif T_batch > L:
            x_in = x[:, :L, :]
        else:
            x_in = x

        # embedding + transformer
        enc = self.enc_embedding(x_in)               # [B, in_channels*n_patches, D]
        enc, _ = self.transformer_encoder(enc)       # same

        # reshape: [B, in_channels, n_patches*D]
        # IMPORTANT: use self.in_channels, not C (they should match, but this is safer)
        enc = enc.reshape(B, self.in_channels, self._n_patches * self.seq_hidden_units)

        # [B, in_channels, L] -> [B, L, in_channels]
        time_feats = self.proj_tokens_to_time(enc).transpose(1, 2)

        # [B, L, hr_size]
        hr = F.elu(self.hr_output_transformation(time_feats))
        hr = self.hr_dropout(hr)

        # Final alignment: return hr with EXACT batch time length T_batch
        if hr.size(1) > T_batch:
            hr = hr[:, :T_batch, :]
        elif hr.size(1) < T_batch:
            pad = torch.zeros(B, T_batch - hr.size(1), hr.size(2), device=hr.device, dtype=hr.dtype)
            hr = torch.cat([hr, pad], dim=1)

        return hr

    def forward(self, batch):
        """
        GT-compatible forward:
        - training:
            - if projection_horizon==0: return (pred_factuals, None, None, active_entries)
            - else: return (None, pred_pseudos_all_steps, pseudo_outcomes_all_steps, active_entries_all_steps)
        - eval:
            return (pred_outcomes, hr)
        """
        prev_treatments = batch["prev_treatments"]
        vitals = batch["vitals"] if self.has_vitals else None
        prev_outputs = batch["prev_outputs"]
        static_features = batch["static_features"]
        curr_treatments = batch["current_treatments"]
        active_entries = batch["active_entries"].clone()

        batch_size = prev_treatments.size(0)
        time_dim = prev_treatments.size(1)

        if self.training:
            # ---- factual one-step training ----
            if self.projection_horizon == 0:
                hr = self.build_hr(prev_treatments, vitals, prev_outputs, static_features, active_entries)
                # Guard: ensure time dims match
                if hr.size(1) != curr_treatments.size(1):
                    T = curr_treatments.size(1)
                    hr = hr[:, :T, :]
                pred_factuals = self.G_comp_heads[0].build_outcome(hr, curr_treatments)  # [B,T,Y]
                return pred_factuals, None, None, active_entries

            # ---- GT-style g-computation training ----
            pseudo_outcomes_all_steps = torch.zeros(
                (batch_size, time_dim - self.projection_horizon - 1, self.projection_horizon + 1, self.dim_outcome),
                device=self.device,
            )
            pred_pseudos_all_steps = torch.zeros_like(pseudo_outcomes_all_steps)
            active_entries_all_steps = torch.zeros(
                (batch_size, time_dim - self.projection_horizon - 1, 1), device=self.device
            )

            for t in range(1, time_dim - self.projection_horizon):
                current_active_entries = batch["active_entries"].clone()
                current_active_entries[:, int(t + self.projection_horizon) :] = 0.0
                active_entries_all_steps[:, t - 1, :] = current_active_entries[:, t + self.projection_horizon - 1, :]

                # 1) pseudo outcomes under counterfactual treatment sequence (no grad)
                with torch.no_grad():
                    indexes_cf = (torch.arange(0, time_dim, device=self.device) >= (t - 1)) & (
                        torch.arange(0, time_dim, device=self.device) < (t + self.projection_horizon)
                    )

                    curr_treatments_cf = curr_treatments.clone()
                    curr_treatments_cf[:, indexes_cf, :] = self.treatment_sequence.to(self.device)

                    prev_treatments_cf = torch.cat((prev_treatments[:, :1, :], curr_treatments_cf[:, :-1, :]), dim=1)

                    hr_cf = self.build_hr(
                        prev_treatments_cf, vitals, prev_outputs, static_features, current_active_entries
                    )

                    pseudo_outcomes = torch.zeros(
                        (batch_size, self.projection_horizon + 1, self.dim_outcome), device=self.device
                    )

                    for i in range(self.projection_horizon, 0, -1):
                        pseudo_outcome = self.G_comp_heads[i].build_outcome(hr_cf, curr_treatments_cf)[:, t + i - 1, :]
                        pseudo_outcomes[:, i - 1, :] = pseudo_outcome

                    pseudo_outcomes[:, -1, :] = batch["outputs"][:, t + self.projection_horizon - 1, :]
                    pseudo_outcomes_all_steps[:, t - 1, :, :] = pseudo_outcomes

                # 2) predict pseudo outcomes from factual hr (grad)
                hr = self.build_hr(prev_treatments, vitals, prev_outputs, static_features, current_active_entries)

                pred_pseudos = torch.zeros(
                    (batch_size, self.projection_horizon + 1, self.dim_outcome), device=self.device
                )
                for i in range(self.projection_horizon, -1, -1):
                    pred_pseudo = self.G_comp_heads[i].build_outcome(hr, curr_treatments)[:, t + i - 1, :]
                    pred_pseudos[:, i, :] = pred_pseudo

                pred_pseudos_all_steps[:, t - 1, :, :] = pred_pseudos

            return None, pred_pseudos_all_steps, pseudo_outcomes_all_steps, active_entries_all_steps

        # ---- evaluation / prediction ----
        fixed_split = batch["sequence_lengths"] - self.max_projection if self.projection_horizon > 0 else batch["sequence_lengths"]
        for i in range(len(active_entries)):
            active_entries[i, int(fixed_split[i] + self.projection_horizon) :] = 0.0

        hr = self.build_hr(prev_treatments, vitals, prev_outputs, static_features, active_entries)

        if self.projection_horizon > 0:
            pred_outcomes = self.G_comp_heads[0].build_outcome(hr, curr_treatments)
            index_pred = (torch.arange(0, time_dim, device=self.device) == fixed_split[..., None] - 1)
            pred_outcomes = pred_outcomes[index_pred]  # [B,Y] (GT-style)
        else:
            pred_outcomes = self.G_comp_heads[0].build_outcome(hr, curr_treatments)  # [B,T,Y]

        return pred_outcomes, hr

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        pred_factuals, pred_pseudos, pseudo_outcomes, active_entries_all_steps = self(batch)

        if self.projection_horizon > 0:
            active_entries_all_steps = active_entries_all_steps.unsqueeze(-2)  # [B,steps,1,1]
            mse = F.mse_loss(pred_pseudos, pseudo_outcomes, reduction="none")  # [B,steps,H+1,Y]
            mse = (mse * active_entries_all_steps).sum(dim=(0, 1)) / (
                active_entries_all_steps.sum(dim=(0, 1)).clamp_min(1.0) * self.dim_outcome
            )

            for i in range(mse.shape[0]):
                self.log(
                    f"{self.model_type}_mse_{i}",
                    mse[i].mean(),
                    on_epoch=True,
                    on_step=False,
                    sync_dist=True,
                    prog_bar=True,
                )

            loss = mse.mean()
        else:
            mse = F.mse_loss(pred_factuals, batch["outputs"], reduction="none")  # [B,T,Y]
            loss = (mse * batch["active_entries"]).sum() / (batch["active_entries"].sum().clamp_min(1.0) * self.dim_outcome)

        self.log(f"{self.model_type}_train_loss", loss, on_epoch=True, on_step=False, sync_dist=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataset_idx=None):
        pred_outcomes, hr = self(batch)
        return pred_outcomes.cpu(), hr.cpu()

    def get_predictions(self, dataset: Dataset) -> np.array:
        logger.info(f"Predictions for {dataset.subset_name}.")
        data_loader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)
        outcome_pred, _ = [torch.cat(arrs) for arrs in zip(*self.trainer.predict(self, data_loader))]
        return outcome_pred.numpy()
    

    def get_normalised_n_step_rmses(self, dataset: Dataset):
        logger.info(f'RMSE calculation for {dataset.subset_name}.')

        outputs_scaled = self.get_predictions(dataset)
        unscale = self.hparams.exp.unscale_rmse
        percentage = self.hparams.exp.percentage_rmse

        # Only evaluate RMSE on final outcome (same as GT)
        if unscale:
            output_stds, output_means = dataset.scaling_params['output_stds'], dataset.scaling_params['output_means']
            outputs_unscaled = outputs_scaled * output_stds + output_means
            mse = ((outputs_unscaled - dataset.data_processed_seq['unscaled_outputs'][:, (self.projection_horizon - 1)]) ** 2)
        else:
            mse = ((outputs_scaled - dataset.data_processed_seq['outputs'][:, (self.projection_horizon - 1)]) ** 2)

        nan_idx = np.unique(np.where(np.isnan(dataset.data_processed_seq['outputs']))[0])
        not_nan = np.array([i for i in range(outputs_scaled.shape[0]) if i not in nan_idx])
        mse = mse[not_nan]

        mse = mse.mean()  # mean across batch
        rmse_normalised = np.sqrt(mse) / dataset.norm_const

        if percentage:
            rmse_normalised *= 100.0

        return rmse_normalised



    def configure_optimizers(self):
        optimizer = self._get_optimizer(list(self.named_parameters()))
        if self.hparams.model[self.model_type]["optimizer"]["lr_scheduler"]:
            return self._get_lr_schedulers(optimizer)
        return optimizer

    @staticmethod
    def set_hparams(model_args: DictConfig, new_args: dict, input_size: int, model_type: str):
        sub_args = model_args[model_type]
        sub_args.optimizer.learning_rate = new_args["learning_rate"]
        sub_args.batch_size = new_args["batch_size"]

        if "seq_hidden_units" in new_args:
            sub_args.seq_hidden_units = int(input_size * new_args["seq_hidden_units"])
        if "hr_size" in new_args:
            sub_args.hr_size = int(input_size * new_args["hr_size"])

        sub_args.fc_hidden_units = int(sub_args.hr_size * new_args["fc_hidden_units"])
        sub_args.dropout_rate = new_args["dropout_rate"]
        sub_args.num_layer = new_args["num_layer"] if "num_layer" in new_args else sub_args.num_layer

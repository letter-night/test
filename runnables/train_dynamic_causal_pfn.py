import logging
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor

from src.models.utils import FilteringMlFlowLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)


@hydra.main(config_name="config.yaml", config_path="../config/")
def main(args: DictConfig):
    """
    Training / evaluation script for DynamicCausalPFN (GT-style multi-horizon loop)

    Returns:
        dict with results (one-step and n-step-ahead RMSEs)
    """
    results = {}

    # Non-strict access to fields
    OmegaConf.set_struct(args, False)
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)
    logger.info("\n" + OmegaConf.to_yaml(args, resolve=True))

    # ============================== Data init ===========================================
    seed_everything(args.exp.seed)
    dataset_collection = instantiate(args.dataset, _recursive_=True)

    # PFN uses multi-input preprocessed dataset (same as GT)
    dataset_collection.process_data_multi()

    # Populate model dims from dataset
    args.model.dim_outcomes = dataset_collection.train_f.data["outputs"].shape[-1]
    args.model.dim_treatments = dataset_collection.train_f.data["current_treatments"].shape[-1]
    args.model.dim_vitals = dataset_collection.train_f.data["vitals"].shape[-1] if dataset_collection.has_vitals else 0
    args.model.dim_static_features = dataset_collection.train_f.data["static_features"].shape[-1]

    # ============================== Callbacks & Logger =================================
    pfn_callbacks = []

    if args.exp.logging:
        experiment_name = f'{args.model.name}/{args.dataset.name}'
        mlf_logger = FilteringMlFlowLogger(
            filter_submodels=[],
            experiment_name=experiment_name,
            tracking_uri=args.exp.mlflow_uri,
            run_name="0",
        )
        pfn_callbacks += [LearningRateMonitor(logging_interval="epoch")]
        artifacts_path = hydra.utils.to_absolute_path(
            mlf_logger.experiment.get_run(mlf_logger.run_id).info.artifact_uri
        )
    else:
        mlf_logger = None
        artifacts_path = None

    # ============================== 1-step ahead (projection_horizon=0) =================
    args.model.dynamic_causal_pfn.projection_horizon = 0
    pfn_model = instantiate(args.model.dynamic_causal_pfn, args, dataset_collection, _recursive_=False)

    if getattr(args.model.dynamic_causal_pfn, "tune_hparams", False):
        pfn_model.finetune(resources_per_trial=args.model.dynamic_causal_pfn.resources_per_trial)

    pfn_trainer = Trainer(
        gpus=eval(str(args.exp.gpus)),
        logger=mlf_logger,
        max_epochs=args.exp.max_epochs,
        callbacks=pfn_callbacks,
        terminate_on_nan=True,
        gradient_clip_val=args.model.dynamic_causal_pfn.max_grad_norm,
    )
    pfn_trainer.fit(pfn_model)

    # Validation factual RMSE
    val_dataloader = DataLoader(dataset_collection.val_f, batch_size=args.dataset.val_batch_size, shuffle=False)
    pfn_trainer.test(pfn_model, test_dataloaders=val_dataloader)

    val_rmse_orig, val_rmse_all = pfn_model.get_normalised_masked_rmse(dataset_collection.val_f)
    logger.info(f"Val normalised RMSE (all): {val_rmse_all}; Val normalised RMSE (orig): {val_rmse_orig}")

    encoder_results = {}
    if hasattr(dataset_collection, "test_cf_one_step"):
        test_rmse_orig, test_rmse_all, test_rmse_last = pfn_model.get_normalised_masked_rmse(
            dataset_collection.test_cf_one_step, one_step_counterfactual=True
        )
        logger.info(
            f"Test normalised RMSE (all): {test_rmse_all}; "
            f"Test normalised RMSE (orig): {test_rmse_orig}; "
            f"Test normalised RMSE (only counterfactual): {test_rmse_last}"
        )
        encoder_results = {
            "encoder_val_rmse_all": val_rmse_all,
            "encoder_val_rmse_orig": val_rmse_orig,
            "encoder_test_rmse_all": test_rmse_all,
            "encoder_test_rmse_orig": test_rmse_orig,
            "encoder_test_rmse_last": test_rmse_last,
        }
    elif hasattr(dataset_collection, "test_f"):
        test_rmse_orig, test_rmse_all = pfn_model.get_normalised_masked_rmse(dataset_collection.test_f)
        logger.info(
            f"Test normalised RMSE (all): {test_rmse_all}; "
            f"Test normalised RMSE (orig): {test_rmse_orig}."
        )
        encoder_results = {
            "encoder_val_rmse_all": val_rmse_all,
            "encoder_val_rmse_orig": val_rmse_orig,
            "encoder_test_rmse_all": test_rmse_all,
            "encoder_test_rmse_orig": test_rmse_orig,
        }

    if args.exp.logging:
        mlf_logger.log_metrics(encoder_results)
    results.update(encoder_results)

    # ============================== multi-step ahead loop ===============================
    for t in range(1, args.dataset.projection_horizon + 1):
        seed_everything(args.exp.seed)

        decoder_results = {
            "decoder_val_rmse_all": val_rmse_all,
            "decoder_val_rmse_orig": val_rmse_orig,
        }

        if args.exp.logging:
            mlf_logger = FilteringMlFlowLogger(
                filter_submodels=[],
                experiment_name=experiment_name,
                tracking_uri=args.exp.mlflow_uri,
                run_name=str(t),
            )

        # Train for horizon t
        args.model.dynamic_causal_pfn.projection_horizon = t
        pfn_model = instantiate(args.model.dynamic_causal_pfn, args, dataset_collection, _recursive_=False)

        pfn_trainer = Trainer(
            gpus=eval(str(args.exp.gpus)),
            logger=mlf_logger,
            max_epochs=args.exp.max_epochs,
            callbacks=pfn_callbacks,
            terminate_on_nan=True,
            gradient_clip_val=args.model.dynamic_causal_pfn.max_grad_norm,
        )
        pfn_trainer.fit(pfn_model)

        # Test n-step RMSE (same branching as GT)
        if hasattr(dataset_collection, "test_cf_treatment_seq"):
            test_rmse = pfn_model.get_normalised_n_step_rmses(dataset_collection.test_cf_treatment_seq)
        elif hasattr(dataset_collection, "test_f_multi"):
            test_rmse = pfn_model.get_normalised_n_step_rmses(dataset_collection.test_f_multi)
        else:
            # Fall back: if no processed seq datasets exist, just evaluate on test_f (if present)
            if hasattr(dataset_collection, "test_f"):
                test_rmse = pfn_model.get_normalised_n_step_rmses(dataset_collection.test_f)
            else:
                raise AttributeError(
                    "No suitable test dataset found for multi-step evaluation "
                    "(expected test_cf_treatment_seq or test_f_multi)."
                )

        test_rmses = {f"{t+1}-step": test_rmse}
        logger.info(f"Test normalised RMSE (n-step prediction): {test_rmses}")

        decoder_results.update({("decoder_test_rmse_" + k): v for (k, v) in test_rmses.items()})

        if args.exp.logging:
            mlf_logger.log_metrics(decoder_results)
        results.update(decoder_results)

    if args.exp.logging:
        mlf_logger.experiment.set_terminated(mlf_logger.run_id)

    return results


if __name__ == "__main__":
    main()
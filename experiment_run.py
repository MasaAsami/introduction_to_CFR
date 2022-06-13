import hydra
from omegaconf import DictConfig
from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG
from datetime import datetime
import numpy as np

import mlflow
from mlflow import log_metric, log_param, log_artifacts

from src.utils import torch_fix_seed, fetch_sample_data, ipm_scores
from src.cfr import CFR


@hydra.main(config_path="configs", config_name="experiments.yaml")
def run_experiment(cfg: DictConfig):

    # start new run
    mlflow.set_tracking_uri("file://" + hydra.utils.get_original_cwd() + "/mlruns")
    mlflow.set_experiment("CFR experiments")

    logger = getLogger("run DFR")
    logger.setLevel(DEBUG)
    handler_format = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(handler_format)
    file_handler = FileHandler(
        hydra.utils.get_original_cwd()
        + "/logs/"
        + "cfr"
        + "-"
        + "{:%Y-%m-%d-%H:%M:%S}.log".format(datetime.now()),
        "a",
    )
    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.debug(f"Start process...")

    with mlflow.start_run():

        torch_fix_seed()

        # パラメータの保存
        log_param("alpha", cfg["alpha"])
        log_param("split_outnet", cfg["split_outnet"])
        log_param("train_test_random", cfg["random_state"])

        (
            dataloader,
            X_train,
            y_train,
            t_train,
            X_test,
            y_test,
            t_test,
        ) = fetch_sample_data(
            random_state=cfg["random_state"], test_size=0.25, StandardScaler=cfg["StandardScaler"], data_path=hydra.utils.get_original_cwd() + "/data/sample_data.csv"
        )
        model = CFR(in_dim=8, out_dim=1, cfg=cfg)
        within_result, outof_result, train_mse, ipm_result = model.fit(
            dataloader, X_train, y_train, t_train, X_test, y_test, t_test, logger
        )

        within_ipm = ipm_scores(model, X_train, t_train, sig=0.1)
        outof_ipm = ipm_scores(model, X_test, t_test, sig=0.1)
        log_param("within_IPM_pre", within_ipm["ipm_lin_before"])
        log_param("outof_IPM_pre", outof_ipm["ipm_lin_before"])

        # metricの保存
        log_metric("within_ATT", within_result["ATT"])
        log_metric("within_ATTerror", np.abs(within_result["ATT"] - 1676.3426))
        log_metric("within_RMSE", within_result["RMSE"])
        log_metric("within_IPM", within_ipm["ipm_lin"])

        log_metric("outof_ATT", outof_result["ATT"])
        log_metric("outof_ATTerror", np.abs(outof_result["ATT"] - 1676.3426))
        log_metric("outof_RMSE", outof_result["RMSE"])
        log_metric("outof_IPM", outof_ipm["ipm_lin"])


if __name__ == "__main__":
    run_experiment()

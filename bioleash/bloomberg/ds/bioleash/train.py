import torch, argparse
import os
from loguru import logger
from pathlib import Path
import polars as pl
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TQDMProgressBar,
)
from transformers import AutoTokenizer
import time
from argparse import ArgumentParser
import boto3
from boto3.s3.transfer import TransferConfig
from bloomberg.ds.bioleash.config import *
from bloomberg.ds.bioleash.LMDataModule import LMDataModule
from bloomberg.ds.bioleash.LBModelModule import LBModelModule
from datasets import load_dataset

bcos_client = boto3.client(
    "s3",
    endpoint_url="https://s3.dev.bcs.bloomberg.com",
    aws_access_key_id="0KWJUSV760E0T97W8UIU",
    aws_secret_access_key="h6NxaQnaBl4Heql4w0BRQgpFkcz61ZgxhDuMcfWT",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=config_defaults.model_name)
    parser.add_argument("--debug", type=bool, default=config_defaults.debug)
    parser.add_argument("--batch_size", type=int, default=config_defaults.batch_size)
    parser.add_argument("--max_epochs", type=int, default=config_defaults.max_epochs)
    parser.add_argument("--num_workers", type=int, default=config_defaults.num_workers)
    parser.add_argument(
        "--enable_progress_bar", type=bool, default=config_defaults.enable_progress_bar
    )
    parser.add_argument("--accelerator", type=str, default=config_defaults.accelerator)
    parser.add_argument("--precision", type=str, default=config_defaults.precision)
    parser.add_argument(
        "--gradient_clip_val", type=int, default=config_defaults.gradient_clip_val
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=config_defaults.accumulate_grad_batches,
    )
    parser.add_argument(
        "--devices", type=list, nargs="+", default=config_defaults.devices
    )
    parser.add_argument(
        "--limit_train_batches", type=str, default=config_defaults.limit_train_batches
    )
    parser.add_argument("--data_dir", type=str, default=config_defaults.data_dir)
    parser.add_argument("--model_dir", type=str, default=config_defaults.model_dir)
    parser.add_argument("--train_file", type=str, default=config_defaults.train_file)
    parser.add_argument("--validation_file", type=str, default=config_defaults.validation_file)
    parser.add_argument("--test_file", type=str, default=config_defaults.test_file)

    return parser.parse_args()


def getDataModule(config, bucket_name="ckpt-test"):
    response = bcos_client.list_objects_v2(Bucket=bucket_name)

    logger.info("-------------------Start Loading Dataset-------------------")
    for content in response.get("Contents", []):
        logger.info(content["Key"], content["Size"])
    for file_path in [config.train_file, config.validation_file, config.test_file]:
        bcos_client.download_file(
            Bucket=bucket_name,
            Key=file_path,
            Filename=os.path.join(config.data_dir, file_path),
            Config=TransferConfig(
                multipart_chunksize=MULTIPART_CHUNKSIZE,
                max_concurrency=MAX_CONCURRENCY,
                use_threads=USE_THREADS,
            ),
        )

    N_ROWS = 180_000_000
    if config.debug:
        N_SAMPLES = 10_000
    else:
        N_SAMPLES = 98_415_610
    train_df = load_dataset(
        "parquet",
        data_files=os.path.join(config.data_dir, config.train_file),
        split="train",
        streaming=True,
    )
    # train_df = pl.read_parquet(
    #     "/notebooks/train_df.parquet",
    #     n_rows=100 if config.debug else None,
    # )
    val_df = load_dataset(
        "parquet",
        data_files=os.path.join(config.data_dir, config.validation_file),
        split="train",
        streaming=True,
    )
    # val_df = pl.read_parquet(
    #     "/notebooks/val_df.parquet",
    #     n_rows=100 if config.debug else None,
    # )
    # test_df = load_dataset(
    #     "parquet",
    #     data_files=os.path.join(config.data_dir, config.test_file),
    #     split="train",
    #     streaming=True,
    # )
    test_df = pl.read_parquet(
        os.path.join(config.data_dir, config.test_file),
        columns=["molecule_smiles"],
    ).unique(maintain_order=True)
    logger.info(next(iter(train_df)))

    # get tokenizer
    def get_tokenizer_from_hf_hub(hf_model_name):
        os.environ["http_proxy"] = "http://devproxy.bloomberg.com:82"
        os.environ["https_proxy"] = "http://devproxy.bloomberg.com:82"
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        # unset proxies
        _ = os.environ.pop("http_proxy", None)
        _ = os.environ.pop("https_proxy", None)
        return tokenizer

    tokenizer = get_tokenizer_from_hf_hub(config.model_name)

    # create data module
    datamodule = LMDataModule(train_df, val_df, test_df, tokenizer, config)
    return datamodule


def getModelModule(model_name):
    return LBModelModule(model_name)


def check_cuda(config):
    if torch.cuda.is_available():
        config.precision = "16-mixed"
        config.pt_version = torch.__version__
        config.cuda_version = torch.version.cuda
    return config


def train(dataModule, modelModule, config):

    checkpoint_callback = ModelCheckpoint(
        filename=f"model-{{val_map:.4f}}",
        save_weights_only=True,
        dirpath=os.path.join(config.model_dir, "checkpoints"),
        verbose=1,
    )
    early_stop_callback = EarlyStopping(monitor="val_map", mode="max", patience=3)
    progress_bar_callback = TQDMProgressBar(refresh_rate=1)
    callbacks = [
        checkpoint_callback,
        progress_bar_callback,
    ]

    logger.info("-------------------Start Training-------------------")
    logger.info(torch.backends.cuda.flash_sdp_enabled())
    # True
    logger.info(torch.backends.cuda.mem_efficient_sdp_enabled())
    # True
    logger.info(torch.backends.cuda.math_sdp_enabled())
    # True
    trainer_params = {
        "max_epochs": config.max_epochs,
        "enable_progress_bar": config.enable_progress_bar,
        "accelerator": config.accelerator,
        "precision": config.precision,
        "gradient_clip_val": config.gradient_clip_val,
        "accumulate_grad_batches": config.accumulate_grad_batches,
        "devices": config.devices,
    }
    os.mkdir(os.path.join(config.data_dir, "lightning_logs"))
    os.mkdir(os.path.join(config.model_dir, "checkpoints"))
    trainer = L.Trainer(
        callbacks=callbacks, **trainer_params, default_root_dir=config.data_dir
    )
    trainer.fit(modelModule, dataModule)
    return trainer
    # if config.inference:
    #     trainer.kserve_deploy(train_loader, config)
    #     trainer.inference(test_loader)


def get_gpu_name():
    return torch.cuda.get_device_name()


def get_capability_score():
    return torch.cuda.get_device_capability()


if __name__ == "__main__":
    logger.info(
        f"pt{torch.__version__} cuda{torch.version.cuda} gpu name{get_gpu_name()} compatibility score{get_capability_score()}"
    )
    args = parse_args()
    args = check_cuda(args)
    dataModule = getDataModule(config=args)
    modelModule = getModelModule(args.model_name)

    trainer = train(dataModule, modelModule, args)
    test_df = pl.read_parquet(
        Path(args.data_dir, args.test_file),
        columns=["molecule_smiles"],
    ).unique(maintain_order=True)
    logger.info(f"test file length {len(test_df)}")
    working_dir = Path(os.path.join(args.model_dir, "checkpoints"))
    model_paths = working_dir.glob("*.ckpt")
    test_dataloader = dataModule.test_dataloader()
    for model_path in model_paths:
        print(model_path)
        modelmodule = LBModelModule.load_from_checkpoint(
            checkpoint_path=model_path,
            model_name=args.model_name,
        )
        predictions = trainer.predict(modelmodule, test_dataloader)
        predictions = torch.cat(predictions).numpy()
        logger.info(f"prediction length {len(test_df)}")
        pred_dfs = []
        for i, protein_name in enumerate(PROTEIN_NAMES):
            pred_dfs.append(
                test_df.with_columns(
                    pl.lit(protein_name).alias("protein_name"),
                    pl.lit(predictions[:, i]).alias("binds"),
                )
            )
        pred_df = pl.concat(pred_dfs)
        submit_df = (
            pl.read_parquet(
                Path(args.data_dir, args.test_file),
                columns=["id", "molecule_smiles", "protein_name"],
            )
            .join(pred_df, on=["molecule_smiles", "protein_name"], how="left")
            .select(["id", "binds"])
            .sort("id")
        )
        logger.info(submit_df.head())

        submission_file = f"submission_{model_path.stem}.csv"
        submit_df.write_csv(Path(working_dir, submission_file))
        bcos_client.upload_file(
            Bucket="ckpt-test",
            Key=submission_file,
            Filename=Path(working_dir, submission_file),
            Config=TransferConfig(
                multipart_chunksize=MULTIPART_CHUNKSIZE,
                max_concurrency=MAX_CONCURRENCY,
                use_threads=USE_THREADS,
            ),
        )

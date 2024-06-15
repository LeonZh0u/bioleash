from types import SimpleNamespace
import os

AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# boto3 optimizations https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3.html#using-the-transfer-manager
MULTIPART_CHUNKSIZE = 16777216
MAX_CONCURRENCY = 20
USE_THREADS = True


PROTEIN_NAMES = ["BRD4", "HSA", "sEH"]
config_defaults = SimpleNamespace(
    model_name="DeepChem/ChemBERTa-77M-MTR",
    debug=False,
    normalize=True,
    batch_size=3000,
    max_epochs=5,
    enable_progress_bar=True,
    accelerator="auto",
    precision="16-mixed",
    gradient_clip_val=None,
    accumulate_grad_batches=1,
    devices=[0],
    limit_train_batches=1000_000_000,
    data_dir="/notebooks/dataset/",
    model_dir ="/notebooks/",
    num_workers = 8,
    test_file="test.parquet",
    train_file="train_df.parquet",
    validation_file="val_df.parquet"
)

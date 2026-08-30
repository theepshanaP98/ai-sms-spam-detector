# Dataset directory

The training script downloads the **UCI SMS Spam Collection** automatically and stores the extracted file here as `SMSSpamCollection`.

Dataset source: UCI Machine Learning Repository, dataset ID 228.

The raw downloaded dataset file is intentionally not required to be committed because `scripts/train_model.py` can obtain it during local or cloud build. If automatic download is unavailable, download the dataset manually from UCI and place the file named `SMSSpamCollection` in this directory.

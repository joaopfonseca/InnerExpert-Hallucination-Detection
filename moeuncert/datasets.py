import pandas as pd
import requests


def fetch_truthfulqa():
    """
    Fetch the TruthfulQA dataset from Hugging Face.

    Returns:
        pd.DataFrame: The dataset as a pandas DataFrame.
    """
    # Load the dataset from Hugging Face
    return pd.read_csv("hf://datasets/domenicrosati/TruthfulQA/train.csv")


def fetch_nq_open(split="validation"):
    """
    Fetch the Natural Questions Open dataset from Hugging Face.

    Returns:
        pd.DataFrame: The dataset as a pandas DataFrame.
    """
    # Load the dataset from Hugging Face
    splits = {
        "train": "nq_open/train-00000-of-00001.parquet",
        "validation": "nq_open/validation-00000-of-00001.parquet",
    }
    df = pd.read_parquet(
        "hf://datasets/google-research-datasets/nq_open/" + splits[split]
    )
    return df


def fetch_squad(split="validation"):
    """
    Fetch the SQuAD dataset from Hugging Face.

    Returns:
        pd.DataFrame: The dataset as a pandas DataFrame.
    """
    # Login using e.g. `huggingface-cli login` to access this dataset
    splits = {
        "train": "plain_text/train-00000-of-00001.parquet",
        "validation": "plain_text/validation-00000-of-00001.parquet",
    }
    df = pd.read_parquet("hf://datasets/rajpurkar/squad/" + splits[split])
    return df


def fetch_freshqa():
    """
    Fetch the FreshQA dataset from Google Sheets (Nov. 11, 2025 update).

    Returns:
        pd.DataFrame: The dataset as a pandas DataFrame.
    """
    SPREADSHEET_ID = "1X6oTXzU1L9PWc2uim1eVzdX8V7y7_4crWfdJhVV08L4"
    url = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/export?format=csv"
    return pd.read_csv(url, skiprows=2)


def fetch_realtimeqa(split="latest", month=None):
    """
    Fetch the RealTimeQA dataset.

    Args:
        split (str, int): The dataset split to fetch. Can be "latest" or a specific year.
        month (int, optional): If split is not "latest", specify the month (1-12) to fetch.

    Returns:
        pd.DataFrame: The dataset as a pandas DataFrame.
    """

    url = f"https://api.github.com/repos/realtimeqa/realtimeqa_public/contents/"

    data_dir = "latest" if split == "latest" else f"past/{split}"
    response = requests.get(url + data_dir)
    response.raise_for_status()

    files = pd.DataFrame(response.json())[["name", "url"]]
    # "qa_gen" -> Questions + evidence
    #
    # "qa_nota" -> Same questions with 4 options (with one being "none of the
    #              above") + evidence
    # "qa" -> Same questions with options (without "none of the above") +
    #         evidence
    target_files = files["name"].apply(
        lambda x: x.replace("_public", "").endswith("_qa.jsonl")
    )
    if split != "latest" and month is not None:
        target_files &= files["name"].str.startswith(f"{split}{month:02d}")

    file_urls = (
        files[target_files]["url"]
        .apply(
            lambda x: (
                x.split("?")[0]
                .replace("api.github.com/repos", "raw.githubusercontent.com")
                .replace("/contents/", "/refs/heads/main/")
            )
        )
        .tolist()
    )
    data = [pd.read_json(file_url, lines=True) for file_url in file_urls]
    df = pd.concat(data, ignore_index=True)
    if "answer" in df.columns:
        df["answer_idx"] = df["answer"].apply(lambda x: x[0]).astype(int)
        df["answer_str"] = df.apply(
            lambda row: row["choices"][row["answer_idx"]], axis=1
        )
    else:
        df["answer_idx"] = df.apply(
            lambda row: [
                (choice.lower() in row["evidence"].lower()) for choice in row["choices"]
            ],
            axis=1,
        )
        df["answer_idx"] = df["answer_idx"].apply(
            lambda x: x.index(True) if True in x else -1
        )
        df["answer_str"] = df.apply(
            lambda row: (
                row["choices"][row["answer_idx"]]
                if row["answer_idx"] != -1
                else row["evidence"]
            ),
            axis=1,
        )

    return df


def fetch_xsum(split="test"):
    """
    Fetch the XSum dataset from Hugging Face.

    Returns:
        pd.DataFrame: The dataset as a pandas DataFrame.
    """
    splits = {
        "train": "data/train-00000-of-00001.parquet",
        "validation": "data/validation-00000-of-00001.parquet",
        "test": "data/test-00000-of-00001.parquet",
    }
    df = pd.read_parquet("hf://datasets/EdinburghNLP/xsum/" + splits[split])
    return df

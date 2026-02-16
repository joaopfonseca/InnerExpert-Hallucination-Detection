import pandas as pd

def fetch_truthfulqa():
    """
    Fetch the TruthfulQA dataset from Hugging Face.
    
    Returns:
        pd.DataFrame: The dataset as a pandas DataFrame.
    """
    # Load the dataset from Hugging Face
    return pd.read_csv("hf://datasets/domenicrosati/TruthfulQA/train.csv")

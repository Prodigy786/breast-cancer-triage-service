import os
import argparse
import pandas as pd
from sklearn.datasets import load_breast_cancer


def prepare_dataset(output_path: str = "data/raw/breast_cancer.csv") -> None:
    data = load_breast_cancer()
    feature_names = [f.replace(" ", "_") for f in data.feature_names]
    
    df = pd.DataFrame(data.data, columns=feature_names)
    df["target"] = data.target

    # Validation checks
    assert len(feature_names) == 30, f"Expected 30 features, got {len(feature_names)}"
    assert df.shape[1] == 31, f"Expected 31 columns (30 features + target), got {df.shape[1]}"
    assert df.isnull().sum().sum() == 0, "Dataset contains null values"
    assert len(df) == 569, f"Expected 569 samples, got {len(df)}"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully prepared and saved to {output_path}")
    print(f"Shape: {df.shape}, Features: 30, No null values.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export breast cancer dataset to CSV")
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/raw/breast_cancer.csv",
        help="Path where breast cancer CSV should be exported",
    )
    args = parser.parse_args()
    prepare_dataset(args.output_path)

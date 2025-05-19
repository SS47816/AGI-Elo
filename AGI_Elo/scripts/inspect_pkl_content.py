import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Inspect .pickle files')
    parser.add_argument('--dataset_name', default="ImageNet", type=str, help='"ImageNet", "COCO", "MMLU", "LiveCodeBench", "Waymo", "NAVSIM" ')
    parser.add_argument('--dataset_split', default="val", type=str, help='"train", "val", "test", "mini" ')
    parser.add_argument('--result_folder', default="predictions", type=str, help='"predictions", "matches", "ratings" ')
    args = parser.parse_args()

    # Set the folder containing the .pkl files
    if args.dataset_name == "ImageNet":
        pkl_folder = Path(f"./data/classification")
    elif args.dataset_name == "COCO":
        pkl_folder = Path(f"./data/detection")
    elif args.dataset_name == "MMLU":
        pkl_folder = Path(f"./data/question_answering")
    elif args.dataset_name == "LiveCodeBench":
        pkl_folder = Path(f"./data/coding")
    elif args.dataset_name == "Waymo":
        pkl_folder = Path(f"./data/motion_prediction")
    elif args.dataset_name == "NAVSIM":
        pkl_folder = Path(f"./data/motion_planning")

    # Find all .pkl files in the folder
    pkl_folder = Path.joinpath(pkl_folder, args.dataset_name, args.dataset_split, args.result_folder)
    pkl_files = sorted(pkl_folder.glob("*.pkl"))

    # Loop through each .pkl file and display its first few rows
    for pkl_file in pkl_files:
        print(f"Inspecting: {pkl_file.name}")
        df = pd.read_pickle(pkl_file)

        if isinstance(df, pd.DataFrame):
            print(df.head(20))
            # print(df.tail(20))
        elif isinstance(df, dict):
            # print(df.keys())
            print(list(df.items())[0])
            print(list(df.items())[-1])
        else:
            print(type(df))

        print("=" * 80)  # Separator

if __name__ == "__main__":
    main()

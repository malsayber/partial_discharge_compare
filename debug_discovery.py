import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import config
from preprocess import discovery


def debug_discovery():
    print(f"RAW_DIR: {config.RAW_DIR}")
    dataset_name = "iris"
    sessions = discovery.discover_sessions(dataset_name)
    print(f"Discovered sessions for {dataset_name}: {sessions}")
    if not sessions:
        print(f"No sessions found for {dataset_name}. Checking directory contents...")
        dataset_path = config.RAW_DIR / dataset_name
        if dataset_path.exists():
            print(f"Contents of {dataset_path}:")
            for item in dataset_path.iterdir():
                print(f"  - {item.name} (Is Dir: {item.is_dir()})")
                if item.is_dir():
                    print(f"    Contents of {item.name}:")
                    for sub_item in item.iterdir():
                        print(f"      - {sub_item.name}")
        else:
            print(f"Dataset path {dataset_path} does not exist.")

if __name__ == "__main__":
    debug_discovery()

import os
from pathlib import Path

def split_celeba():
    # 1. Resolve Paths relative to the project root
    # This looks for 'data' in the parent directory of this script (src/..)
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data"
    
    external_dir = data_dir / "external"
    raw_images_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    partition_file = external_dir / "list_eval_partition.txt"

    # 2. Validation Checks
    if not partition_file.exists():
        print(f"Error: Could not find partition file at {partition_file}")
        return

    if not raw_images_dir.exists():
        print(f"Error: Could not find raw images at {raw_images_dir}")
        return

    # Mapping: 0=train, 1=val, 2=test
    id_to_dir = {0: "train", 1: "val", 2: "test"}

    # 3. Create subdirectories
    for folder in id_to_dir.values():
        (processed_dir / folder).mkdir(parents=True, exist_ok=True)

    print("Starting split process...")
    count = 0

    # 4. Process the manifest
    with open(partition_file, "r") as f:
        for line in f:
            filename, part_id = line.strip().split()
            part_id = int(part_id)
            
            target_folder = id_to_dir[part_id]
            source_path = raw_images_dir / filename
            link_path = processed_dir / target_folder / filename

            # Only create link if it doesn't already exist
            if not link_path.exists():
                # We use absolute paths for the source to ensure symlinks work
                os.symlink(source_path.resolve(), link_path)
                count += 1
    
    print(f"Operation complete.")

if __name__ == "__main__":
    split_celeba()
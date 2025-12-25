import pytest
import os
from pathlib import Path

@pytest.fixture
def data_paths():
    """Provides consistent path resolution for tests."""
    root = Path(__file__).resolve().parent.parent
    return {
        "manifest": root / "data" / "external" / "list_eval_partition.txt",
        "processed": root / "data" / "processed"
    }

@pytest.fixture
def manifest_data(data_paths):
    """Parses the manifest into a dictionary."""
    with open(data_paths["manifest"], "r") as f:
        return {line.split()[0]: int(line.split()[1]) for line in f}

@pytest.fixture
def processed_files(data_paths):
    """Gathers actual filenames present in each split folder."""
    base = data_paths["processed"]
    return {
        "train": set(os.listdir(base / "train")),
        "val": set(os.listdir(base / "val")),
        "test": set(os.listdir(base / "test")),
    }

def test_exclusivity(processed_files):
    """Verify sets are disjoint: S_train ∩ S_val ∩ S_test = ∅"""
    s = processed_files
    assert s["train"].isdisjoint(s["val"]), "Overlap between Train and Val"
    assert s["train"].isdisjoint(s["test"]), "Overlap between Train and Test"
    assert s["val"].isdisjoint(s["test"]), "Overlap between Val and Test"

def test_completeness(processed_files, manifest_data):
    """Verify total count matches the manifest."""
    total_found = sum(len(files) for files in processed_files.values())
    assert total_found == len(manifest_data), f"Expected {len(manifest_data)} files, found {total_found}"

def test_mapping_integrity(processed_files, manifest_data):
    """Verify every file is in its mathematically correct folder."""
    id_map = {0: "train", 1: "val", 2: "test"}
    for split_id, folder_name in id_map.items():
        for filename in processed_files[folder_name]:
            assert manifest_data[filename] == split_id
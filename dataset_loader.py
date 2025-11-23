"""
01_dataset_loader.py
Load multiple cancer detection datasets from Hugging Face
"""

import os
from datasets import load_dataset, DatasetDict

def load_all_datasets():
    """Load all available cancer detection datasets from Hugging Face"""
    
    datasets_loaded = []
    
    print("🔍 Loading datasets from Hugging Face...")
    print("=" * 60)
    
    # Dataset 1: Skin Cancer
    try:
        print("\n📦 Loading Skin Cancer dataset...")
        skin_cancer = load_dataset("marmal88/skin_cancer")
        datasets_loaded.append(('skin_cancer', skin_cancer))
        print("✅ Skin Cancer dataset loaded successfully")
    except Exception as e:
        print(f"⚠️ Skin Cancer dataset failed: {e}")
    
    # Dataset 2: Breast Cancer Histopathology
    try:
        print("\n📦 Loading Breast Cancer Histopathology dataset...")
        breast_cancer = load_dataset("1aurent/BreakHis")
        datasets_loaded.append(('breast_cancer', breast_cancer))
        print("✅ Breast Cancer dataset loaded successfully")
    except Exception as e:
        print(f"⚠️ Breast Cancer dataset failed: {e}")
    
    # Dataset 3: Chest X-ray
    try:
        print("\n📦 Loading Chest X-ray dataset...")
        chest_xray = load_dataset("keremberke/chest-xray-classification", name="full")
        datasets_loaded.append(('chest_xray', chest_xray))
        print("✅ Chest X-ray dataset loaded successfully")
    except Exception as e:
        print(f"⚠️ Chest X-ray dataset failed: {e}")
    
    # Dataset 4: Colorectal Cancer
    try:
        print("\n📦 Loading Colorectal Cancer dataset...")
        colorectal = load_dataset("zjysteven/Kather_texture_2016_image_tiles_5000")
        datasets_loaded.append(('colorectal', colorectal))
        print("✅ Colorectal Cancer dataset loaded successfully")
    except Exception as e:
        print(f"⚠️ Colorectal Cancer dataset failed: {e}")
    
    print("\n" + "=" * 60)
    print(f"✅ Successfully loaded {len(datasets_loaded)} datasets")
    
    return datasets_loaded

def explore_dataset(dataset, name):
    """Explore dataset structure"""
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")
    
    for split in dataset.keys():
        print(f"\n{split} split:")
        print(f"  Samples: {len(dataset[split])}")
        print(f"  Features: {dataset[split].features}")
        
        if len(dataset[split]) > 0:
            sample = dataset[split][0]
            print(f"  Sample keys: {sample.keys()}")

if __name__ == "__main__":
    datasets = load_all_datasets()
    
    # Explore each dataset
    for name, dataset in datasets:
        explore_dataset(dataset, name)
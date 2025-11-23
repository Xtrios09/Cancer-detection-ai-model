"""
02_data_processor.py
Process and combine multiple datasets into unified format
"""

import os
from PIL import Image
from datasets import load_dataset

def process_and_save_dataset(dataset, dataset_name, output_dir='combined_data', target_size=(224, 224)):
    """
    Process dataset and save to disk in organized format
    Binary classification: benign vs malignant
    """
    
    print(f"\n🔄 Processing {dataset_name}...")
    
    # Create directories
    for split in ['train', 'validation', 'test']:
        os.makedirs(f'{output_dir}/{split}/benign', exist_ok=True)
        os.makedirs(f'{output_dir}/{split}/malignant', exist_ok=True)
    
    # Mapping for different cancer types to binary
    cancer_mappings = {
        'skin_cancer': {
            'benign': ['nv', 'bkl', 'df'],
            'malignant': ['mel', 'bcc', 'akiec', 'vasc']
        },
        'breast_cancer': {
            'benign': [0, 'benign', 'normal'],
            'malignant': [1, 'malignant', 'cancer']
        },
        'chest_xray': {
            'benign': ['normal', 'NORMAL'],
            'malignant': ['cancer', 'CANCER', 'tumor', 'TUMOR']
        },
        'colorectal': {
            'benign': ['normal', 'NORM'],
            'malignant': ['tumor', 'TUM', 'cancer']
        }
    }
    
    stats = {'benign': 0, 'malignant': 0, 'skipped': 0}
    
    for split in dataset.keys():
        print(f"  Processing {split} split...")
        
        for idx, sample in enumerate(dataset[split]):
            try:
                # Get image
                img = sample.get('image', sample.get('img', None))
                if img is None:
                    stats['skipped'] += 1
                    continue
                
                # Convert to PIL Image
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                
                img = img.convert('RGB')
                img = img.resize(target_size, Image.LANCZOS)
                
                # Get label
                label = sample.get('label', sample.get('labels', None))
                if label is None:
                    stats['skipped'] += 1
                    continue
                
                # Map to binary
                binary_label = None
                mapping = cancer_mappings.get(dataset_name, {})
                
                if label in mapping.get('benign', []):
                    binary_label = 'benign'
                elif label in mapping.get('malignant', []):
                    binary_label = 'malignant'
                elif isinstance(label, (int, float)):
                    binary_label = 'benign' if label == 0 else 'malignant'
                else:
                    stats['skipped'] += 1
                    continue
                
                # Save image
                save_dir = f'{output_dir}/{split}/{binary_label}'
                filename = f"{dataset_name}_{split}_{idx}.jpg"
                img.save(f'{save_dir}/{filename}', 'JPEG', quality=95)
                
                stats[binary_label] += 1
                
                if (idx + 1) % 100 == 0:
                    print(f"    Processed {idx + 1} samples...")
                    
            except Exception as e:
                stats['skipped'] += 1
                continue
    
    print(f"\n  📊 {dataset_name} Statistics:")
    print(f"    Benign: {stats['benign']}")
    print(f"    Malignant: {stats['malignant']}")
    print(f"    Skipped: {stats['skipped']}")
    print(f"    Total: {stats['benign'] + stats['malignant']}")
    
    return stats

if __name__ == "__main__":
    from dataset_loader import load_all_datasets
    
    # Load datasets
    datasets = load_all_datasets()
    
    # Process each dataset
    all_stats = {}
    for name, dataset in datasets:
        stats = process_and_save_dataset(dataset, name)
        all_stats[name] = stats
    
    # Print overall statistics
    print(f"\n{'='*60}")
    print(f"OVERALL STATISTICS")
    print(f"{'='*60}")
    total_benign = sum(s['benign'] for s in all_stats.values())
    total_malignant = sum(s['malignant'] for s in all_stats.values())
    
    print(f"Total Benign: {total_benign}")
    print(f"Total Malignant: {total_malignant}")
    print(f"Total Samples: {total_benign + total_malignant}")
    print(f"Balance: {total_benign/(total_benign+total_malignant)*100:.1f}% benign")
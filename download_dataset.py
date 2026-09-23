import os
import random
from datasets import load_dataset
from PIL import Image

def main():
    print("Loading dataset from HuggingFace with streaming...")
    # Load dataset
    dataset = load_dataset("jonathan-roberts1/Ships-In-Satellite-Imagery", split="train", streaming=True)
    
    # Total is 4000. Let's reserve 800 for blind test (20%), 3200 for training/cv.
    # We will randomly assign them.
    random.seed(42)
    
    os.makedirs('dataset/train_val/ship', exist_ok=True)
    os.makedirs('dataset/train_val/no_ship', exist_ok=True)
    os.makedirs('dataset/blind_test/ship', exist_ok=True)
    os.makedirs('dataset/blind_test/no_ship', exist_ok=True)
    
    print("Saving images...")
    train_count = 0
    test_count = 0
    
    for i, item in enumerate(dataset):
        img = item['image']
        label = item['labels'] # 1 for ship, 0 for no_ship
        
        # Determine folder based on label
        label_str = 'ship' if label == 1 else 'no_ship'
        
        # Determine split (20% chance for test)
        if random.random() < 0.2:
            folder = f'dataset/blind_test/{label_str}'
            test_count += 1
        else:
            folder = f'dataset/train_val/{label_str}'
            train_count += 1
            
        filename = f"{folder}/img_{i}.png"
        img.save(filename)
        
    print("Dataset saved successfully!")
    print(f"Train/Val: {train_count} images")
    print(f"Blind Test: {test_count} images")

if __name__ == "__main__":
    main()

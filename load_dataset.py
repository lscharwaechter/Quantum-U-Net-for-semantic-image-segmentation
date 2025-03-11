# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 20:16:31 2025

@author: Leon Scharwächter
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import os


# A class to load AI4Mars Dataset with image and label folders
class AI4MarsDataset(Dataset):
    def __init__(self, image_dir, label_dir, test_version=None, transform=None, subset_size=None):
        self.image_dir = image_dir

        # If using test dataset, append the selected test version subfolder
        if test_version:
            self.label_dir = os.path.join(label_dir, test_version)
            self.is_test = True # Flag to check if it's test data
        else:
            self.label_dir = label_dir
            self.is_test = False

        self.transform = transform
        
        # Define label mapping (RGB to class index)
        self.label_map = {
            (0, 0, 0): 0,         # Soil
            (1, 1, 1): 1,         # Bedrock
            (2, 2, 2): 2,         # Sand
            (3, 3, 3): 3,         # Big Rock
            (255, 255, 255): 255  # NULL (ignored during training)
        }

        # List available label filenames
        label_filenames = set(os.listdir(self.label_dir))

        # Filter image filenames to include only those with matching labels
        self.image_filenames = []
        unmatched_files = [] # Track missing labels

        for img_name in sorted(os.listdir(self.image_dir)):
            # Convert image name to corresponding label filename
            if self.is_test:
                label_name = img_name.replace(".JPG", "_merged.png").replace(".jpg", "_merged.png")
            else:
                label_name = img_name.replace(".JPG", ".png").replace(".jpg", ".png")

            if label_name in label_filenames:
                self.image_filenames.append(img_name)
            else:
                unmatched_files.append(img_name)

        # Debugging Info
        print(f"Found {len(self.image_filenames)} images with labels.")
        print(f"Skipping {len(unmatched_files)} images due to missing labels.")
        if unmatched_files:
            print("Example missing labels:", unmatched_files[:10]) # Show first 10 missing files

        # Select a subset of data if requested
        if subset_size and subset_size < len(self.image_filenames):
            self.image_filenames = random.sample(self.image_filenames, subset_size)

        # Select a subset of data if requested
        if subset_size and subset_size < len(self.image_filenames):
            self.image_filenames = random.sample(self.image_filenames, subset_size)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Handle test dataset label naming
        if "test" in self.label_dir:
            label_name = img_name.replace(".JPG", "_merged.png").replace(".jpg", "_merged.png")
        else:
            label_name = img_name.replace(".JPG", ".png").replace(".jpg", ".png")
    
        label_path = os.path.join(self.label_dir, label_name)
    
        # Load image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image file could not be loaded: {img_path}")
        image = cv2.resize(image, (128, 128))
    
        # Load label
        label = cv2.imread(label_path)
        if label is None:
            raise FileNotFoundError(f"Label file could not be loaded: {label_path}")
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        label = cv2.resize(label, (128, 128), interpolation=cv2.INTER_NEAREST)
    
        # Convert label to class indices
        label_mask = np.full((128, 128), 255, dtype=np.uint8) # Default to 255 (ignore)
        for rgb, class_idx in self.label_map.items():
            if class_idx != 255: # Ignore NULL pixels
                mask = (label[:, :, 0] == rgb[0]) & (label[:, :, 1] == rgb[1]) & (label[:, :, 2] == rgb[2])
                label_mask[mask] = class_idx
    
        if self.transform:
            image = self.transform(image)
            label_mask = torch.tensor(label_mask, dtype=torch.long)
    
        return image, label_mask
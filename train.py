# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 20:20:40 2025

@author: Leon Scharwächter
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

import pennylane as qml

from tqdm import tqdm
import matplotlib.pyplot as plt

from hybrid_quantum_model import QuantumUNet
from load_dataset import AI4MarsDataset


# Transformations (Normalize and Convert to Tensor)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1]
])

# Create dataloader
# Set subset size for smaller experiments
subset_size_train = 1000
subset_size_test = 300
train_dataset = AI4MarsDataset(
    image_dir="AI4Mars/images/edr", 
    label_dir="AI4Mars/labels/train", 
    transform=transform, 
    subset_size=subset_size_train
)
test_dataset = AI4MarsDataset(
    image_dir="AI4Mars/images/edr", 
    label_dir="AI4Mars/labels/test", 
    test_version="masked-gold-min3-100agree", 
    transform=transform, 
    subset_size=subset_size_test
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

#%%

# Training the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QuantumUNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=255)

# Store training loss values
train_losses = []

# Training function
def train(model, train_loader, epochs=20, save_path="quantum_unet.pt"):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for images, masks in tqdm(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader) # Compute average loss
        train_losses.append(avg_loss)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Save trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved as {save_path}")

# Train the Quantum U-Net
train(model, train_loader, epochs=20)

# Plot training loss curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', color='b')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.show()

#%%

# Calculate Intersection over Union (IoU)
def compute_iou(preds, labels, num_classes=4, ignore_index=255):
    iou_per_class = []
    preds = torch.argmax(preds, dim=1) # Get the class index with highest probability

    for cls in range(num_classes): # Loop over classes (0,1,2,3)
        pred_class = preds == cls
        label_class = labels == cls

        intersection = (pred_class & label_class).sum().float()
        union = (pred_class | label_class).sum().float()

        if union == 0: # Avoid division by zero
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append(intersection / union)

    return torch.tensor(iou_per_class).nanmean() # Average over all classes

# Calculate Dice Score (F1 Score for Segmentation)
def compute_dice(preds, labels, num_classes=4, ignore_index=255):
    dice_per_class = []
    preds = torch.argmax(preds, dim=1)

    for cls in range(num_classes):
        pred_class = preds == cls
        label_class = labels == cls

        intersection = (pred_class & label_class).sum().float()
        dice_score = (2.0 * intersection) / (pred_class.sum().float() + label_class.sum().float() + 1e-6)

        dice_per_class.append(dice_score)

    return torch.tensor(dice_per_class).mean()

# Evaluate the model
def test(model, test_loader):
    model.eval()
    iou_per_class = {i: [] for i in range(4)} # IoU storage per class
    dice_per_class = {i: [] for i in range(4)} # Dice score storage per class

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)
            output = model(images)

            # Convert model outputs to class predictions
            predictions = torch.argmax(output, dim=1)

            # Compute IoU and Dice for each class
            for class_idx in range(4):
                pred_mask = (predictions == class_idx)
                true_mask = (masks == class_idx)

                intersection = torch.logical_and(pred_mask, true_mask).sum().item()
                union = torch.logical_or(pred_mask, true_mask).sum().item()
                dice_denominator = pred_mask.sum().item() + true_mask.sum().item()

                if union > 0:
                    iou = intersection / union
                    iou_per_class[class_idx].append(iou)

                if dice_denominator > 0:
                    dice = (2 * intersection) / dice_denominator
                    dice_per_class[class_idx].append(dice)

    # Compute mean IoU & Dice per class
    mean_iou_per_class = {
        cls: np.mean(iou_per_class[cls]) if iou_per_class[cls] else 0
        for cls in iou_per_class
    }
    mean_dice_per_class = {
        cls: np.mean(dice_per_class[cls]) if dice_per_class[cls] else 0
        for cls in dice_per_class
    }

    mean_iou = np.mean(list(mean_iou_per_class.values()))
    mean_dice = np.mean(list(mean_dice_per_class.values()))

    print("Mean IoU per class:", mean_iou_per_class)
    print("Overall Mean IoU:", mean_iou)
    print("Mean Dice per class:", mean_dice_per_class)
    print("Overall Mean Dice Score:", mean_dice)

    return mean_iou, mean_dice

test_iou, test_dice = test(model, test_loader)
print(f"IoU Score: {test_iou:.4f}, Dice Score: {test_dice:.4f}")


#%%

# Extract the trained quantum parameters
quantum_layer = model.quantum_conv # Access the QuantumConvLayer
trained_q_params = quantum_layer.q_params.detach().cpu().numpy() # Convert to numpy for analysis

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(trained_q_params, bins=20, alpha=1, edgecolor='black', linewidth=2)
ax.set_xlabel("Quantum Weights (Angles)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Trained Quantum Weights (N=16)")
ax.grid(axis='y')
for spine in ax.spines.values():
    spine.set_linewidth(2)
plt.show()


#%%

# Select a few samples from the dataset
num_samples = 5
sample_images, _ = next(iter(train_loader)) # Get a batch of images
sample_images = sample_images[:num_samples].to(device) # Select subset

# Pass images through the trained quantum layer
quantum_layer = model.quantum_conv
quantum_outputs = []
with torch.no_grad():
    for img in sample_images:
        img_flat = img.view(-1).unsqueeze(0) # Flatten image & add batch dim
        q_output = quantum_layer(img_flat) # Pass through quantum layer
        quantum_outputs.append(q_output.cpu().numpy())

quantum_outputs = np.array(quantum_outputs).squeeze() # Shape: (num_samples, n_qubits)

# Plot images and quantum outputs side-by-side
fig, axes = plt.subplots(num_samples, 2, figsize=(12, 10))

for i in range(num_samples):
    # Plot the image (convert tensor to numpy)
    axes[i, 0].imshow(sample_images[i].cpu().squeeze(), cmap="gray")
    axes[i, 0].set_title(f"Example No. {i+1}")
    axes[i, 0].axis("off")

    # Plot the quantum output (bar chart)
    axes[i, 1].bar(range(quantum_outputs.shape[1]), quantum_outputs[i], alpha=1, linewidth=2)
    if i == 0:
        axes[i, 1].set_title("Quantum Output")
    if i == num_samples-1:
        axes[i, 1].set_xlabel("Qubit Index")
    axes[i, 1].set_ylabel("Output")
    axes[i, 1].set_ylim([-1, 1]) # Since outputs are Pauli-Z expectation values
    axes[i, 1].grid()

for row in axes:
    for ax in row:
        for spine in ax.spines.values():
            spine.set_linewidth(2)

plt.tight_layout()
plt.show()


#%%
n_qubits = 16

# Define PennyLane device
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, params):
    # Encode classical inputs using RY rotations
    for i in range(n_qubits):
        qml.RY(inputs[i] * np.pi, wires=i)

    # Apply quantum transformations (entanglement in 4x4 grid structure)
    for i in range(0, n_qubits - 1, 2):
        qml.RX(params[i], wires=i) # Trainable RX gate
        qml.RY(params[i+1], wires=i+1) # Trainable RY gate
        qml.CNOT(wires=[i, i+1]) # Entanglement

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Generate sample inputs and random quantum parameters
sample_inputs = np.random.rand(n_qubits)
sample_params = np.random.randn(n_qubits)

# Visualize the quantum circuit
qml.draw_mpl(quantum_circuit)(sample_inputs, sample_params)
plt.show()


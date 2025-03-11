# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 20:19:03 2025

@author: Leon Scharwächter
"""

import torch
import torch.nn as nn
import numpy as np
import pennylane as qml

# Class for the bottleneck Quantum Convolution Layer
class QuantumConvLayer(nn.Module):
    def __init__(self, n_qubits=16):
        super().__init__()
        self.n_qubits = n_qubits

        # PennyLane quantum device
        self.device = qml.device("default.qubit", wires=n_qubits)

        # Trainable quantum parameters
        self.q_params = nn.Parameter(torch.randn(n_qubits, dtype=torch.float32))

        # Define PennyLane QNode
        self.qnode = qml.QNode(self.pennylane_circuit, self.device, interface="torch")

    def pennylane_circuit(self, inputs):
        """Quantum circuit in PennyLane"""
        for i in range(self.n_qubits):
            qml.RY(inputs[i] * np.pi, wires=i) # Angle Encoding

        for i in range(0, self.n_qubits - 1, 2):
            qml.RX(self.q_params[i], wires=i) # Trainable RX
            qml.RY(self.q_params[i+1], wires=i+1) # Trainable RY
            qml.CNOT(wires=[i, i+1]) # Entanglement

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x):
        batch_size = x.shape[0]
        q_outputs = []

        for i in range(batch_size):
            q_out = self.qnode(x[i].view(-1)) # Apply quantum circuit
            q_outputs.append(torch.tensor(q_out, dtype=torch.float32))

        return torch.stack(q_outputs)

class QuantumUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder (Downsampling)
        self.enc1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)     # 128x128 -> 128x128
        self.pool1 = nn.MaxPool2d(2, 2)                                      # 128x128 -> 64x64

        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)   # 64x64 -> 64x64
        self.pool2 = nn.MaxPool2d(2, 2)                                      # 64x64 -> 32x32

        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # 32x32 -> 32x32
        self.pool3 = nn.MaxPool2d(2, 2)                                      # 32x32 -> 16x16

        self.enc4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)  # 16x16 -> 16x16
        self.pool4 = nn.MaxPool2d(2, 2)                                      # 16x16 -> 8x8

        self.enc5 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1) # 8x8 -> 8x8
        self.pool5 = nn.MaxPool2d(2, 2)                                      # 8x8 -> 4x4

        # Quantum Processing
        self.reduce_channels = nn.Conv2d(1024, 1, kernel_size=1) # Reduce channels before quantum
        self.quantum_conv = QuantumConvLayer(n_qubits=16)
        self.expand_channels = nn.Conv2d(1, 1024, kernel_size=1) # Expand quantum output back
        
        # Decoder (Upsampling)
        self.up1 = nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=2, padding=1, output_padding=1) # 4x4 -> 8x8
        self.dec1 = nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1)

        self.up2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)  # 8x8 -> 16x16
        self.dec2 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)   # 16x16 -> 32x32
        self.dec3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)

        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)   # 32x32 -> 64x64
        self.dec4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)

        self.up5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)    # 64x64 -> 128x128
        self.final_conv = nn.Conv2d(128, 4, kernel_size=1)  # Final segmentation (4 classes)

    def forward(self, x):
        ################################
        ##          Encoder           ##
        ################################
        x1 = torch.relu(self.enc1(x))
        x1p = self.pool1(x1)

        x2 = torch.relu(self.enc2(x1p))
        x2p = self.pool2(x2)

        x3 = torch.relu(self.enc3(x2p))
        x3p = self.pool3(x3)

        x4 = torch.relu(self.enc4(x3p))
        x4p = self.pool4(x4)

        x5 = torch.relu(self.enc5(x4p))
        x5p = self.pool5(x5) # 8x8 -> 4x4 (Quantum Input)

        ################################
        ##     Quantum Processing     ##
        ################################
        
        # Collapse channels before quantum processing
        x5_reduced = self.reduce_channels(x5p) # (batch, 1024, 4, 4) -> (batch, 1, 4, 4)
        
        # Flatten for quantum input
        q_input = x5_reduced.view(x5_reduced.shape[0], -1) # (batch, 16)
        
        # Quantum feature extraction
        q_output = self.quantum_conv(q_input) # (batch, 16)
    
        # Reshape and expand Back    
        q_output = q_output.view(q_output.shape[0], 1, 4, 4) # Reshape to (batch, 16, 4, 4)
        q_output = self.expand_channels(q_output) # Expand back to (batch, 1024, 4, 4)
        
        ################################
        ##          Decoder           ##
        ################################
        d1 = torch.cat([self.up1(q_output), x5], dim=1)
        d1 = torch.relu(self.dec1(d1)) # (batch, 1024, 8, 8)

        d2 = torch.cat([self.up2(d1), x4], dim=1)
        d2 = torch.relu(self.dec2(d2)) # (batch, 512, 16, 16)

        d3 = torch.cat([self.up3(d2), x3], dim=1)
        d3 = torch.relu(self.dec3(d3)) # (batch, 256, 32, 32)

        d4 = torch.cat([self.up4(d3), x2], dim=1) 
        d4 = torch.relu(self.dec4(d4)) # (batch, 128, 64, 64)
        
        d5 = torch.cat([self.up5(d4), x1], dim=1)
        output = self.final_conv(d5) # (batch, 4, 128, 128): 4 terrain class possibilities
        
        return output

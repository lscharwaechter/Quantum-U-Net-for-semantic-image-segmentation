# Quantum U-Net for semantic image segmentation
In this project, quantum processing is integrated into a convolutional U-Net architecture for semantic segmentation of Mars terrain. Instead of a purely classical feature extractor, quantum computing is leveraged at the bottleneck layer of the U-Net encoder-decoder scheme.

A series of classical convolutional and pooling layers serve as an encoder and extract spatial features from the input image. The encoded feature maps are then flattened and fed into a quantum circuit with 16 qubits. The library [PennyLane](https://pennylane.ai/) is used to define the quantum grid and to implement the quantum computations within PyTorch's computational graph, which ensures that the gradients flow properly through the hybrid quantum-classical model. The inputs for the quantum computation are converted into qubit states via RY rotations. Trainable RX, RY, and CNOT gates are applied to process the quantum states. The circuit afterwards outputs expectation values of Pauli-Z measurements, forming new feature representations. These quantum-enhanced features are finally upsampled via transposed convolutions in the decoder and skip connections (from the encoder layers to the decoder layers), reconstructing a pixel-wise classification terrain segmentation map. The optimization process compares the predicted segmentation map with labeled image masks that contain the true label per pixel.

In this experiment, the [AI4Mars dataset](https://data.nasa.gov/Space-Science/AI4MARS-A-Dataset-for-Terrain-Aware-Autonomous-Dri/cykx-2qix/about_data) is utilized. This dataset consists of images captured by the Navigation Cameras and Panoramic Cameras aboard Mars rovers, including Curiosity, Perseverance, Spirit, and Opportunity. The images represent the Martian terrain and are annotated with pixel-wise labels for terrain classification. Each pixel of an image is labeled as one of the following terrain types:

- Soil (0,0,0): Loose Martian regolith, often appearing as smooth or granular areas.
- Bedrock (1,1,1): Exposed solid rock formations, important for navigation and scientific analysis.
- Sand (2,2,2): Soft, fine-grained surfaces that can pose mobility risks to rovers.
- Big Rock (3,3,3): Large, scattered boulders that obstruct movement.
- NULL (255,255,255): Unlabeled regions, including distant terrain or rover components.

The quantum circuit in the U-Net bottleneck uses a specific alternating RX and RY gate scheme and CNOT gates for entanglement:
<p align="center">
<img src="https://github.com/user-attachments/assets/3f1e00e1-c5d2-457e-8e91-e3db59b614cd" width="250"/>
</p>

<p align="center">
<img src="https://github.com/user-attachments/assets/e7432284-8296-4da8-9d63-1f5f176c9af3" width="600"/>
</p>

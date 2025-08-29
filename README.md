RISC-V Edge AI with VSDSquadron PRO: Simulated MNIST Digit Recognition
Overview
This repository documents my learning journey through the RISC-V Edge AI with VSDSquadron PRO course, offered by VLSI System Design (VSD). The course focuses on implementing machine learning (ML) and artificial intelligence (AI) models on resource-constrained embedded systems, targeting the VSDSquadron PRO board powered by the SiFive FE310-G002 RISC-V SoC. Due to the absence of physical hardware, this project was developed and simulated using Visual Studio Code (VSCode) for both the camera-based image capture and the training of a quantized neural network for MNIST handwritten digit classification.
The project uses a training.py script to train and quantize an 8-bit neural network model for MNIST digit recognition, with inference simulated in a bare-metal environment compatible with the SiFive FE310-G002. This README outlines the course content, project setup, key learnings, and instructions for running the simulated application.
Course Content
The course comprises 27 modules, progressing from foundational ML concepts to advanced neural network deployment on RISC-V microcontrollers:

AI On A Microchip: Introduction to edge computing with the VSDSquadron PRO.
Understanding Your RISC-V Board: Prerequisites for AI on a 16KB RAM microcontroller.
Best-Fitting Lines 101: Basics of machine learning with linear regression.
Gradient Descent Unlocked: Building a simple AI model from scratch.
Visualizing Gradient Descent: Understanding optimization in action.
Predicting Startup Profits: Applying AI for business decisions.
Degree Up: Fitting complex patterns for edge AI.
From Python to Silicon: Simulating model deployment on RISC-V.
From Regression to Classification: Building a binary AI classifier.
Implementing KNN Classifier: Creating smarter decision boundaries in Python.
From KNN to SVM: Exploring support vector machines for embedded systems.
Deploying SVM Models: Simulating SVM models in C for VSDSquadron PRO.
Handwritten Digit Recognition with SVM: Applying SVM to the MNIST dataset.
Running MNIST on VSDSquadron PRO: Simulating MNIST classification.
Beating RAM Limits: Introduction to model quantization.
Quantization Demystified: Techniques to fit AI models on tiny devices.
Post-Training Quantization: Reducing model size to MCU-ready.
Fitting AI into 16KB RAM: Optimizing for resource-constrained environments.
Regression to Real-Time Recognition: Recap of the embedded ML pipeline.
From Brain to Code: Exploring the biological inspiration behind neural networks.
From SVM to Neural Networks: Adding hidden layers for improved accuracy.
Neural Networks in Action: Building a neural network with 98% accuracy.
Can We Fit a Neural Network on VSDSQ PRO: Memory optimization strategies.
From VSDSQ Mini to VSDSQ Pro: Simulating real-time AI digit recognition.
Neural Network Implementation Repository: Setting up a neural network project.
Training Bit-Quantized Neural Network: Quantization-aware training techniques.
Exporting Bit-Quantized Neural Network to RISC-V: Simulating deployment on RISC-V.

Project Objective
The objective of this project is to simulate the deployment of a quantized neural network for MNIST handwritten digit classification, optimized for the SiFive FE310-G002 RISC-V microcontroller. The model was trained and quantized using a training.py script in VSCode, with image preprocessing and inference simulated to mimic the behavior of the VSDSquadron PRO board in a bare-metal environment.
Hardware and Software Requirements
Simulated Hardware

Target: SiFive FE310-G002 RISC-V SoC (32-bit RV32IMAC, 320 MHz, 16KB L1 Instruction Cache, 16KB SRAM, 128Mbit QSPI Flash).
Note: No physical VSDSquadron PRO board was used; all code was simulated in VSCode.

Software

Visual Studio Code (VSCode): IDE for running Python scripts and simulating embedded C code.
Python 3.x: With the following libraries:
tensorflow (version 2.15.0)
numpy
opencv-python
pyserial (for simulated UART communication)


Freedom Studio 3.1.1 (optional): For simulating RISC-V bare-metal code, if applicable.
RISC-V GNU Toolchain (optional): For compiling C code for RV32IMAC, if used in simulation.

Project Structure
.
├── src/
│   ├── training.py                     # Script for training and quantizing the MNIST model
│   ├── app_inference.h                 # Core C inference functions (processfclayer, ReLUNorm)
│   ├── main.c                          # Main application logic for simulated inference
│   ├── mnist_model_data.h              # Raw TFLite model data (for reference)
│   ├── mnist_model_params.c            # Generated C file with quantized weights and biases
│   ├── mnist_model_params.h            # Header with model parameter declarations
│   ├── mnist_quantized_model.tflite    # Quantized TensorFlow Lite model
│   ├── cam_capture_image.py            # Python script for simulated webcam image capture
│   ├── send_image_uart.py              # Script for simulated UART image transmission
│   ├── Image_Processing.py             # Script for image preprocessing
│   └── Makefile                        # Build configuration for simulation (if applicable)
└── README.md                           # This file

Setup Instructions
1. Set Up VSCode Environment

Install VSCode and the Python extension.
Install required Python libraries:pip install tensorflow==2.15.0 numpy opencv-python pyserial



2. Train and Quantize the Model

Open VSCode and navigate to the project directory.
Run the training.py script to train and quantize the MNIST model:python src/training.py

This generates mnist_quantized_model.tflite and C-compatible arrays (mnist_model_params.c and mnist_model_params.h).

3. Simulate Image Capture and Preprocessing

Run cam_capture_image.py in VSCode to simulate capturing a handwritten digit image (e.g., using a sample image or webcam input):python src/cam_capture_image.py


Preprocess the image using Image_Processing.py to crop and resize it to 12x12:python src/Image_Processing.py



4. Simulate Inference

Run send_image_uart.py to simulate sending the preprocessed image over UART:python src/send_image_uart.py


Simulate the inference process using the generated C code in main.c, either within VSCode (if configured for C simulation) or by observing the output of the Python scripts.

5. Optional: Simulate in Freedom Studio
If you used Freedom Studio for RISC-V simulation:

Open Freedom Studio and import the project.
Clean and build the project:
Go to Project -> Clean... and Project -> Build Project.


Configure the debug launch for simulation, ensuring the executable points to main.elf.
Run the simulation and observe output in the Freedom Studio console.

Model Details

Architecture:
Input: 12x12 grayscale image (144 features, quantized to 8-bit integers).
Hidden Layers: Two dense layers with 64 neurons each, using LeakyReLU activation.
Output Layer: Dense layer with 10 neurons (for digits 0-9).


Quantization: 8-bit integer format, reducing model size to ~17.35 KB.
Inference Engine: Simulated C implementation with processfclayer and ReLUNorm for efficient integer-only computations.

Key Learnings

Edge AI Simulation: Learned to simulate ML model deployment for resource-constrained RISC-V microcontrollers using VSCode.
Model Training and Quantization: Mastered training and quantizing neural networks using training.py in VSCode.
Image Preprocessing: Implemented cropping and resizing to reduce input size from 28x28 to 12x12, optimizing for memory and speed.
Simulated UART Communication: Developed a simulated serial protocol for image transmission and inference results.
Bare-Metal Concepts: Understood bare-metal programming principles for RISC-V, simulated through C code.
Resource Optimization: Learned to fit a neural network into 16KB RAM using quantization and preprocessing techniques.

Sample Output
8-bit Quantized TFLite MNIST Simulation
By [Your Name]
Starting MNIST inference...
Processing input for sample 1
Starting first layer...
Processing layer: in=144, out=64
Applying ReLU and Requantizing first layer...
Layer1 ReLU range: -96 to 127
Starting second layer...
Processing layer: in=64, out=64
Applying ReLU and Requantizing second layer...
Layer2 ReLU range: -103 to 127
Starting final layer...
Processing layer: in=64, out=10
Predicted digit: 8, True Label: 8, Status: PASS

Acknowledgments

VLSI System Design (VSD): For providing the course resources and project guidance.
SiFive: For the FE310-G002 SoC specifications.
TensorFlow Lite Team: For quantization and inference tools.

License
This project is licensed under the Apache 2.0 License. See the LICENSE file for details.

# AI_VS_REAL-IMAGE-DETECTION

The rapid advancement of generative AI models such as Stable Diffusion and Midjourney has made it increasingly difficult to distinguish real photographs from synthetic images. This raises concerns in digital misinformation, media authenticity, and forensic verification.
This project presents an end-to-end deep learning system designed to classify images as Real or AI-Generated. The solution compares a baseline CNN trained from scratch with a pretrained ResNet-50 using transfer learning and fine-tuning. The final optimized model achieved approximately 98% accuracy on the test set.
The system is deployed via Streamlit for real-time inference.

 # Problem Statement

With diffusion-based image generation models becoming highly realistic, traditional visual inspection is no longer sufficient to verify image authenticity.
The problem addressed in this project is:
Can we design a robust deep learning system capable of detecting synthetic AI-generated images by learning subtle statistical differences between real camera images and diffusion-generated content?

Key challenges include:
Subtle texture inconsistencies
High-frequency distribution shifts
Dataset bias
Overfitting to generator-specific artifacts
Generalization stability

# Dataset Description

Dataset Used: CIFAKE (Kaggle)
Dataset Characteristics:
Binary classification:
Class 0: Real Images (CIFAR-10)
Class 1: AI-Generated Images (Stable Diffusion v1.4)
Total samples: ~120,000 images
Balanced dataset
Original resolution: 32×32

# Preprocessing:

Images resized to 224×224 (to match ResNet input requirements)
Normalization using ImageNet mean & standard deviation
Data augmentation applied during training:
Random horizontal flip
Random rotation

# Methodology

The project was structured into three major experimental phases:
Baseline CNN (trained from scratch)
Transfer Learning (Feature Extraction)
Fine-Tuning of pretrained model
Performance was evaluated using:
Accuracy
Validation loss
Confusion matrix
Training convergence behavior

The project was structured into three experimental phases:

# Baseline CNN (From Scratch)

Convolution + BatchNorm + MaxPool
Global Average Pooling
Dense layer + Dropout
Trained end-to-end
Test Accuracy: 96%

# Transfer Learning (Feature Extraction)

Pretrained ResNet-50 (ImageNet)
Backbone frozen
Replaced final classifier
Trained classifier head only
Test Accuracy: 93%
Observation:
Faster convergence
Stable validation behavior
Required domain adaptation

# Fine-Tuning (Final Model)

Unfroze final residual block (layer4)
Lower learning rate
Early stopping & checkpointing
Test Accuracy: ~98%
Key Insight:
Transfer learning alone is not enough — controlled fine-tuning significantly improves performance and robustness.

# Comparative Results
Model	Test Accuracy	Notes
Baseline CNN	96%	Learned dataset-specific features
ResNet-50 (Frozen)	93%	Faster convergence, needed adaptation
ResNet-50 (Fine-Tuned)	~98%	Best performance & stability

# Training Strategy

Optimizer: Adam
Loss: CrossEntropyLoss
Early stopping implemented
Best model checkpoint saved
Validation monitoring for overfitting control

# Why ResNet-50?

ResNet-50 was selected because:
Residual connections improve gradient flow
Strong representational capacity
Stable fine-tuning behavior
Industry-standard backbone for transfer learning
The goal was to study robustness and representation strength rather than lightweight deployment efficiency.

# Deployment

The final fine-tuned model was deployed using:
PyTorch – Model training & inference
Streamlit – Web-based UI for real-time predictions
Deployment Features
Image upload interface
Real-time inference
Confidence score display
Model caching for efficiency

# Real-World Relevance

As generative AI advances, synthetic media detection becomes critical for:
-Combating misinformation
-Digital forensics
-Content authenticity verification
-AI governance systems
-While ML-based detection analyzes pixel-level inconsistencies, it can complement metadata and provenance-based verification for stronger authenticity pipelines.

# Future Work

Cross-generator testing (Midjourney, DALL·E)
Frequency-domain analysis
Benchmark MobileNetV2 for efficiency comparison
Grad-CAM explainability
Hybrid ML + provenance tracking systems

# Tech Stack

Python
PyTorch
Torchvision
NumPy
Streamlit

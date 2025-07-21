# MNIST GAN - Synthetic Handwritten Digit Generation

A Generative Adversarial Network (GAN) implementation for generating synthetic MNIST handwritten digits using PyTorch.

## 🎯 Project Overview

This project implements a GAN to generate realistic handwritten digits (0-9) trained on the MNIST dataset. The model learns to create new synthetic digit images that are indistinguishable from real handwritten digits.

## 🏗️ Architecture

### Generator Network
- **Input**: 128-dimensional random noise vector
- **Architecture**: 
  - Linear(128 → 256) + BatchNorm + ReLU
  - Linear(256 → 512) + BatchNorm + ReLU  
  - Linear(512 → 1024) + BatchNorm + ReLU
  - Linear(1024 → 784) + Tanh
- **Output**: 28×28 grayscale images

### Discriminator Network
- **Input**: 28×28 MNIST images (flattened to 784)
- **Architecture**:
  - Linear(784 → 512) + LeakyReLU + Dropout(0.3)
  - Linear(512 → 256) + LeakyReLU + Dropout(0.3)
  - Linear(256 → 128) + LeakyReLU + Dropout(0.3)
  - Linear(128 → 1) + Sigmoid
- **Output**: Probability that input image is real (0-1)

## ⚙️ Key Features

- **Improved Training Stability**: Label smoothing and input noise
- **Different Learning Rates**: Generator (0.0001) vs Discriminator (0.0003)
- **Progress Monitoring**: Sample generation every 20 epochs
- **Digit Classification**: CNN classifier to identify generated digits
- **GPU Support**: Automatic CUDA detection and usage

## 🔧 Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Latent Dimension | 128 | Size of input noise vector |
| Batch Size | 128 | Training batch size |
| Learning Rate (G) | 0.0001 | Generator learning rate |
| Learning Rate (D) | 0.0003 | Discriminator learning rate |
| Epochs | 150 | Training epochs |
| Label Smoothing | 0.9/0.1 | Real/Fake label smoothing |

## 📊 Training Process

1. **Discriminator Training**: Learn to distinguish real MNIST images from generated fakes
2. **Generator Training**: Learn to create images that fool the discriminator
3. **Adversarial Process**: Both networks improve through competition

## 🎮 Usage

### Requirements
```bash
pip install torch torchvision matplotlib numpy
```

### Running the Code
1. **Setup & Training**: Run the first cell to initialize and train the GAN
2. **Digit Classification**: Run the second cell to train a classifier
3. **Analysis**: Run remaining cells to analyze generated digits

### Notebook Structure
- **Cell 1**: Main GAN implementation and training
- **Cell 2**: CNN classifier training for digit recognition
- **Cell 3**: Batch analysis of generated images (5×5 grid)
- **Cell 4**: NumPy import
- **Cell 5**: Detailed individual image analysis with probability distributions

## 📈 Results

### Generated Images
<!-- Add your generated image here -->
<img width="610" height="623" alt="Screenshot 2025-07-21 014927" src="https://github.com/user-attachments/assets/985f1f52-ca20-41d5-be6b-ecf2c0efd5e4" />


### Performance Metrics
The model generates diverse digits with:
- High visual quality resembling real handwritten digits
- Good digit diversity (generates all digits 0-9)
- Classifier confidence scores showing realistic digit characteristics

## 🔍 Analysis Features

- **Batch Generation**: Creates 25 images with predictions and confidence scores
- **Individual Analysis**: Detailed probability distribution for each digit (0-9)
- **Statistical Summary**: Distribution analysis of generated digits
- **Visual Monitoring**: Progress tracking during training

## 💻 Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **CPU**: Works on CPU but significantly slower
- **Memory**: 4GB+ RAM recommended
- **Storage**: ~100MB for MNIST dataset

## 🚀 Training Tips

1. **Monitor Loss Values**: Both Generator and Discriminator losses should stabilize
2. **Check Sample Images**: Quality improves gradually over epochs
3. **GPU Usage**: Verify GPU usage for faster training
4. **Experiment**: Try different hyperparameters for varied results

## 📝 Code Structure

```
MINST.ipynb
├── Imports & Setup
├── Data Loading (MNIST)
├── Generator Class
├── Discriminator Class
├── Training Loop
├── Sample Generation
├── Digit Classifier
└── Analysis Tools
```

## 🎯 Future Improvements

- [ ] Implement DCGAN with convolutional layers
- [ ] Add Wasserstein loss for better training stability
- [ ] Conditional GAN for specific digit generation
- [ ] FID/IS scores for quantitative evaluation
- [ ] Progressive growing for higher resolution

## 📚 References

- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) - Original GAN paper
- [MNIST Database](http://yann.lecun.com/exdb/mnist/) - Dataset source
- [PyTorch Documentation](https://pytorch.org/docs/) - Framework reference

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements!

---

**Note**: Training time varies based on hardware. GPU training takes ~30-60 minutes, CPU training may take several hours.

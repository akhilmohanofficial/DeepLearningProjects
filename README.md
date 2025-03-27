# Deep Learning Projects

A collection of deep learning implementations and projects by Akhil Mohan.

## Overview

This repository contains various deep learning projects implemented using popular frameworks like TensorFlow, PyTorch, and Keras. These projects demonstrate practical applications of deep learning techniques across different domains including computer vision, natural language processing, and time series analysis.

## Projects

- **Image Classification**: Implementation of CNNs for image classification tasks
- **Object Detection**: YOLO and R-CNN based object detection models
- **Natural Language Processing**: Sentiment analysis, text classification, and language modeling
- **Generative Models**: GANs and VAEs for image generation
- **Time Series Forecasting**: LSTM and Transformer models for forecasting

## Installation

```bash
# Clone the repository
git clone https://github.com/akhilmohanofficial/DeepLearningProjects.git

# Navigate to the repository
cd DeepLearningProjects

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# For Windows
venv\Scripts\activate
# For macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- PyTorch 1.x
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- CUDA (for GPU acceleration)

## Usage

Each project directory contains its own README with specific instructions. General workflow:

1. Navigate to the project directory
2. Run the training script
3. Evaluate the model
4. Make predictions with new data

Example:
```bash
cd image_classification
python train.py --epochs 50 --batch_size 32
python evaluate.py --model_path models/model_latest.h5
```

## Data

Projects use a combination of:
- Public datasets (MNIST, CIFAR-10, ImageNet, etc.)
- Custom datasets (instructions for data collection in respective project folders)

## Model Architectures

- **CNNs**: ResNet, VGG, Inception
- **RNNs**: LSTM, GRU
- **Transformers**: BERT, GPT-based
- **GANs**: DCGAN, CycleGAN, StyleGAN
- **Autoencoders**: VAE, Denoising Autoencoders

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Akhil Mohan - [@akhilmohanofficial](https://github.com/akhilmohanofficial)

Project Link: [https://github.com/akhilmohanofficial/DeepLearningProjects](https://github.com/akhilmohanofficial/DeepLearningProjects)

## Acknowledgements

- [TensorFlow](https://www.tensorflow.org/)
- [PyTorch](https://pytorch.org/)
- [Keras](https://keras.io/)
- [Papers With Code](https://paperswithcode.com/)

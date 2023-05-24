# Simple-DinoV2-Classification

Welcome to the Simple-DinoV2-Classification repository. This project provides an easy-to-use implementation of the DINOv2 model developed by Facebook, allowing you to train and classify images effectively. Harness the power of universal features for image-level and pixel-level visual tasks like image classification【7†source】. The DINOv2 model used in this project is originally developed by Facebook AI and can be found at [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2).

## Table of Contents

- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

## Getting Started

This section will guide you through the process of setting up the project on your local machine for development and testing purposes.

### Prerequisites

The following packages are required to run the code:

- Python (version 3.9)
- PyTorch (version 2.0.0)
- Torchvision (version 0.15.0)
- Torchmetrics (version 0.10.3)
- OmegaConf
- Fvcore
- IOPath
- XFormers (version 0.0.18)
- CUML (version 11)
- Pip
- SubmitIt (Install via `git+https://github.com/facebookincubator/submitit`)
  
All the dependencies can be installed using the provided conda.yml file【21†source】.

### Installation

1. Clone the repository:

```bash
git clone https://github.com/itsprakhar/Simple-DinoV2-Classification.git
```

2. Change the directory:

```bash
cd Simple-DinoV2-Classification
```

3. Create a conda environment and install dependencies:

```bash
conda env create -f conda.yml
```

4. Activate the conda environment:

```bash
conda activate dinov2
```

## Usage

1. Prepare your dataset and place it in the `data/train` directory. The data should be structured such that each class has its own subdirectory containing the respective images.

2. Run the training script:

```bash
python train.py
```

This will train the model for 100 epochs (modifiable in the script), using the DINOv2 model as a feature extractor and a custom classifier. The training process includes data augmentation, training/validation splitting, and early stopping【11†source】【12†source】【13†source】【14†source】【15†source】.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Prakhar Thakur - itsprakharthakur@gmail.com

Project Link: https://github.com/itsprakhar/Simple-DinoV2-Classification

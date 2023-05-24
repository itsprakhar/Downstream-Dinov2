Simple-DinoV2-Classification
Welcome to the Simple-DinoV2-Classification repository. This project provides an easy-to-use implementation of the DINOv2 model developed by Facebook, allowing you to train and classify images effectively. Harness the power of universal features for image-level and pixel-level visual tasks like image classification​1​. The DINOv2 model used in this project is originally developed by Facebook AI and can be found at facebookresearch/dinov2.

Table of Contents
Getting Started
Prerequisites
Installation
Usage
License
Contact
Getting Started
This section will guide you through the process of setting up the project on your local machine for development and testing purposes.

Prerequisites
The following packages are required to run the code:

Python (version 3.9)
PyTorch (version 2.0.0)
Torchvision (version 0.15.0)
Torchmetrics (version 0.10.3)
OmegaConf
Fvcore
IOPath
XFormers (version 0.0.18)
CUML (version 11)
Pip
SubmitIt (Install via git+https://github.com/facebookincubator/submitit)
All the dependencies can be installed using the provided conda.yml file​2​.

Installation
Clone the repository:
bash
Copy code
git clone https://github.com/itsprakhar/Simple-DinoV2-Classification.git
Change the directory:
bash
Copy code
cd Simple-DinoV2-Classification
Create a conda environment and install dependencies:
bash
Copy code
conda env create -f conda.yml
Activate the conda environment:
bash
Copy code
conda activate dinov2
Usage
Prepare your dataset and place it in the data/train directory. The data should be structured such that each class has its own subdirectory containing the respective images.

Run the training script:

bash
Copy code
python train.py
This will train the model for 100 epochs (modifiable in the script), using the DINOv2 model as a feature extractor and a custom classifier. The training process includes data augmentation, training/validation splitting, and early stopping​3​​4​​5​​6​​7​.

License
Distributed under the MIT License. See LICENSE for more information.

Contact
Your Name - example@example.com

Project Link: https://github.com/itsprakhar/Simple-DinoV2-Classification
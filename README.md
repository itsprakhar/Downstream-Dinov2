# Simple-DinoV2-Classification

Welcome to the Simple-DinoV2-Classification repository! This project provides an easy-to-use implementation of the DINOv2 model developed by Facebook, allowing you to train and classify images effectively. 

The DINOv2 model used in this project is originally developed by Facebook AI and can be found at facebookresearch/dinov2.

## Requirements

The following packages are required to run the code:

  * Python (version 3.9)
  * PyTorch (version 2.0.0)
  * Torchvision (version 0.15.0)
  * Torchmetrics (version 0.10.3)
  * OmegaConf
  * Fvcore
  * IOPath
  * XFormers (version 0.0.18)
  * CUML (version 11)
  * Pip
  * SubmitIt (Install via `git+https://github.com/facebookincubator/submitit`)

All the dependencies can be installed using the provided conda.yml file.

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/itsprakhar/Simple-DinoV2-Classification.git
   ```

2. Change the directory:

   ```
   cd Simple-DinoV2-Classification
   ```

3. Create a conda environment and install dependencies:

   ```
   conda env create -f conda.yml
   ```

4. Activate the conda environment:

   ```
   conda activate dinov2
   ```

## Usage

Prepare your dataset and place it in the `data/train` directory. The data should be structured such that each class has its own subdirectory containing the respective images. Run the training script with:

```
python train.py
```

This will train the model for 100 epochs (modifiable in the script), using the DINOv2 model as a feature extractor and a custom classifier. The training process includes data augmentation, training/validation splitting, and early stopping.

## Demo

A demo notebook is provided to guide you on how to use the trained model to classify images. The notebook demonstrates how to load the model, preprocess an image, and perform inference. Check out the `demo.ipynb` file in the repository.

## License

DINOv2 code and model weights are released under the CC-BY-NC 4.0 license. See LICENSE for additional details.

## Contact

Prakhar Thakur - itsprakharthakur@gmail.com

Project Link: https://github.com/itsprakhar/Simple-DinoV2-Classification.

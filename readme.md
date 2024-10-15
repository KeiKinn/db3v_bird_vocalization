# DB3V: A Dialect Dominated Dataset of Bird Vocalisation for Cross-corpus Bird Species Recognition

This repository contains the official code for the paper **"DB3V: A Dialect Dominated Dataset of Bird Vocalisation for Cross-corpus Bird Species Recognition"**. The dataset and the accompanying code are designed to facilitate research on cross-corpus bird species recognition, especially considering dialectal variations in bird vocalisations.

## Repository Structure

- **Config/**: Configuration folder that contains training and evaluation parameters.
- **train.py**: Script used for training the model.
- **evaluation.py**: Script used for evaluating the trained model.

## Setup and Configuration

1. **Configure Parameters**:

   - Modify the configuration file located at `Config/config_tdnn.yaml` with appropriate datasets and paths. Below is an example configuration template:

   ```yaml
   meta:
     train_ds: ?  # Path to the training dataset
     result: ?    # Path to save training results and model checkpoints
   hparams:
     lr: 1e-4     # Learning rate
     bs: 32       # Batch size
     log_freq: 100 # Frequency of logging metrics during training
     epoch: 100   # Number of epochs
     md_name: bird_img # Model name for saving and loading
   model:
     tdnn: 'tdnn_TN'  # Type of model
     num_classes: 10  # Number of bird species classes
   evaluation:
     ds: ?  # Path to evaluation dataset

## Training the Model
Once you've completed the configuration, you can start training the model by running:
```python
python train_tdnn.py
```
## License
This project is open-sourced under the MIT License.
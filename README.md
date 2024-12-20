# Enhancing Image Classification via Transformer Models
Utilizing Knowledge Distillation and Explainable AI for Efficient and
Improved Predictived Insights. By David B.

## Overview
In this repository you can find all the files and code snippets needed to recreate the experiments for the master thesis.

## File Structure
- `attention_rollout.py` Used to visualize the attention maps for vision transformer architectures
- `attribution.py` Used to visualize heatmaps for different attribution methods after model training
- `class_mapping.txt` Contains the names of the bird species for every class-id. This serves as a translation, for convenience.
- `clean_dataset.py` After model training, can be used to find inconsistencies in the dataset, such as duplicates, outliers etc.
- `cuda_test.py` Checks if pytorch & cuda is succesfully installed on your machine. THis is necessary to train the models on the GPU
- `dataset.py` Contains classes to load datasets and datamodules for training. This can be easily extended to use other data.
- `discord_bot.py` Sends discord notification to a discord channel. Was used to notify user, when certain events triggered (e.g. finished training a model)
- `evaluate_xai_metrics.py` Not needed necessarily. When downloading the metrics data from tensorboard, this can be used to visualize all the metrics in a convenient way. Using tensorboard is preferred tho.
- `fine_tune.py` Used to fine-tune models in a two-step way. This is the second step. Always run `pre_train.py` first before running this.
- `image_utils.py` Utility functions for visualizing images in matplotlib
- `inference_time.py` Measures the inference times of all models
- `k-fold-test.py` Tests the K-Fold Cross Validation
- `kd-train.py` Train KD Models, choose the student-teacher pairings and run the code.
- `knowledge_distillation.py`Reponse-based KD. This function is used to calculate the loss between functions (see Paper by Hinton et al)
- `models.py` All the models are implemented and presented here. Important one of them and start training/evaluating.
- `pre_train.py` Entry point to train all models. Uses pre-trained models and freezes all layers except the last one (first step of two step process). Use `fine_tune.py` After.
- `read_npz.py`
- `test.py` Use this to evaluate all the models. Will log all metrics, attribution methods for the selected model. Use log_config dict to select which parts of testing should enabled/disabled.
- `train.py` Train a selected model with the typical training scheme presented in the thesis.
- `transformations.py` The processing/augmentation pipeline before feeding the image into the models
- `utils.py` Utility functions. Also implements ROAD (Road and Debias)
- `vit_grad_rollout` For future work: Implement attention based rollout with gradients. Not tested yet. Not part of the thesis
- `vit_rollout` Class to use Attention Rollout for ViT Models
- `write_npz` Creates a compressed .npz dataset. Useful when you want to important the complete dataset in Google Colab.
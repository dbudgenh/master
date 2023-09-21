# Master

All documents and code used for the master thesis are stored here

1.  **Guidelines**
    All the guidelines for the master thesis are listed [here](https://www.uni-frankfurt.de/58323931/ContentPage_58323931)

1.  **LaTeX Template**
    The LaTeX template for the master thesis can be found [here](https://github.com/goethe-tcs/thesis-template)

1.  **List of relevant literature/documents**
    - XAI Grad-CAM_Visual_Explanations_ICCV_2017
    - HiResCAM.pdf
    - Pig face recognition and XAI.pdf
    - XAI Score-CAM_Score-Weighted_Visual_Explanat
    - XAI and Classificaiton.pdf
    - KD for Bird Classification.pdf
    - KD Review.pdf
    -
1.  **Bird Dataset**
    This dataset consists of 525 bird species. 84635 training images, 2625 test images(5 images per species) and 2625 validation images (5 images per species). For further information, see [here](https://www.kaggle.com/datasets/gpiosenka/100-bird-species)

### Checklist

- [x] Load bird data
- [x] Show data
- [x] Prepare data (transform,normalize)
- [x] Load model (naive/ResNet/state-of-the-art)
- [x] Data Augmentation
- [x] Train model
- [x] Evaluate model
- [x] Use metrics for unbalanced datasets (MCC matthew correlation coefficient)
- [x] Grad-Cam
- [x] KD review
- [x] KD implementation (try pre-trained student model and from scratch)
- [x] Compare transfer learning with model trainined from scratch
- [ ] K-Fold (4,5) Cross-Validation
- [ ] Try Triplet-network
- [ ] Try Siamese-network
- [x] Try State-of-the-art network
- [x] Fine-tuning model
- [ ] Fewshot learning
- [x] Implement state-of-the-art regularization technqiues
  - [x] LR-Optimization
    - [x] LR-Scheduler
    - [x] LR-Warmup
  - [x] Trivial Augment
  - [x] Random Erasing
  - [x] Label Smoothing
  - [x] Mixup
  - [x] Cutmix
  - [x] Fix Resolution Mitigations
  - [x] Inference Resize Tuning

# Questions

1. GPU Server? Log-in credentials? -> Ask gemma [here](https://images.cv/dataset/snail-image-classification-dataset)
2. Google Colab? Do we get TPUs? Can it run for multiple days without interruption? -> Check pro version for google colab
3. Snail dataset? need it NOW. -> Ask gemma
4. KD Implementations? https://josehoras.github.io/knowledge-distillation/
5. GradCam Implementations? https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82 https://www.coursera.org/projects/deep-learning-with-pytorch-gradcam

# Knowledge Distillation

1. Response based KD
2. Feature based KD
3. Relational KD
4. Attention KD
5. Self distillation
6. Offline distillation #David
7. Online distillation #Train two models at once
8. Collaborative distillation (Multiple teacher assistants) #Have an intermediate network teach the student

# Next steps

Prepare ppt distinction between my presentation and Leo

- Which GradCam techniques am I using? Which are Leo?
- Which KD techniques am I using? Which Leo?

# Unterschiede

- Models
  - David: ResNet, EfficientNet, VisionTransformer
  - Leo: ResNet, Siamese-network / Triplet-Network
- Regularization
  - David: Augmentation/Regularization techniques 1 (Random-Crop, Random erasing (from Paper), ...)
  - Leo : Augmentation/Regularization techniques 2 (AugMix (from Paper), ....)
- XAI
  - David: More focus on GradCam techniques (ScoreCam/HiResCAM etc.), what are the differences, what works the best etc. Less focus on KD (maybe only offline-KD)
  - Leo: More focus on different KD-techniques (Response-based/Feature-based/Relation etc..)., what are the differences, what works the best etc. Less focus on GradCAM.
  - Other XAI techniques? https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Tell_Me_Where_CVPR_2018_paper.pdf

lokal : 3.7it/s batch_size = 32 (total 2645it) -> Pro epoch 12minuten
colab : 2.01it/s batch_size = 128 (total 662it) -> Pro epoch 5.5minuten

-Disentagnling features for Species (not especially snails)
-Methods for XAI on Transformer

-Huggingfaces example for dataloading VisionTransformer
-Dataloading for ViT is different

-What techniques used for multi teacher
-Technique about weighting the different teachers
-Other XAI technique
-Compare methods to other thesis
-Visualization metrics ()
-Make the experiments similar between thesis

-KD between best student and best teacher
-XAI between best student KD and without KD
-Do different KD-techniques
-Compare different implementation families (Resnet vs efficientnet vs ViT)

-Try standard KD method as presented by [Hinton et al. 2015](https://arxiv.org/abs/1503.02531)

- alpha different values between [0.1,0.5,0.7,0.9,0.99]
- T different values between [1,3,5,7,10,20,50,100]
  -Try

# Logging

- Confusion matrix
- Grad-cam
- F1-score/Recall/Sensitivity
- ROC-Curve
- ...

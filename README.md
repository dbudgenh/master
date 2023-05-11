# Master

All documents and code used for the master thesis are stored here.

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
1.  **Bird Dataset**

    This dataset consists of 525 bird species. 84635 training images, 2625 test images(5 images per species) and 2625 validation images(5 images per species). For further information, see [here](https://www.kaggle.com/datasets/gpiosenka/100-bird-species)

### Checklist

- [x] Load bird data
- [ ] Load snail data
- [x] Show data
- [x] Prepare data (transform,normalize)
- [x] Load model (naive/ResNet/state-of-the-art)
- [x] Data Augmentation
- [x] Train model
- [x] Evaluate model
- [x] Use metrics for unbalanced datasets (MCC matthew correlation coefficient)
- [ ] Grad-Cam
- [x] KD review
- [ ] KD implementation
- [ ] Compare transfer learning with model trainined from scratch
- [ ] K-Fold (4,5) Cross-Validation
- [ ] Try Triplet-network
- [ ] Try Siamese-network
- [ ] Try State-of-the-art network
- [ ] Fine-tuning model

- [ ] Fewshot learning

# Questions

1. GPU Server? Log-in credentials?
2. Google Colab? Do we get TPUs? Can it run for multiple days without interruption?
3. Snail dataset? need it NOW.
4. KD Implementations?
5. GradCam Implementations?

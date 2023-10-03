# Russo-Ukrainian War: Prediction and explanation of Twitter suspension

In this repository we provide source code, extracted dataset and parameters described in the research paper: "Russo-Ukrainian War: Prediction and explanation of
Twitter suspension" accepted into ASONAM 2023 conference.

## Organization
Implementation is separated into the data folder, which contain the dataset description and link to Zenodo file sharing service where the data is available with the original split used for model train and test.
In the feature extraction folder we provide exact implementation for the feature extraction procedure used over the collected data stored in local MongoDB. For the pricacy issues and user annonymization we provide only extracted data separated into different feature categoreis where we remove any user identifications like user ID and tweet ID.

| Feature category | # Total features | # Model selected  |
| :---:   | :---: | :---: |
| Profile | 54   | 36   |
| Activity timing | 139   | 109   |
| Textual | 67   | 53   |
| Post embedding | 385   | 328   |
| Graph embedding | 150   | 140   |
| Combination | 1565   | 197   |


In the ml_model folder we provide implementation of the machine learning pipeline utilized for our feature selection, model fine-tuning and final model evaluation. For each separate category of features we provide list of selected features, parameters and performance stored in ml_model/parameters path.
In addition to the implemented ml model we also provide visualized results of our experiments such as roc-auc and precision vs recall curves. Furthermore, we provide the explainability of each developed models based on different feature categories.

![roc-auc curve](https://github.com/alexdrk14/TwitterSuspension/blob/main/plots/roc_curves.png?raw=true)



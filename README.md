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
This table presents the performance of various models measured during K-Fold Cross Validation. The performance is evaluated on the validation set, the initial test set (portion A data), and the second test set (portion B data).

| Model             | Validation | Validation | Validation | Test | Test | Test | Second Test | Second Test | Second Test |
|:-----------------:|:----------:|:----------:|:----------:|:----:|:----:|:----:|:-----------:|:-----------:|:-----------:|
|                   |     F1     |   ROC-AUC  |    Acc.    |  F1  |ROC-AUC| Acc. |     F1      |   ROC-AUC   |    Acc.     |
| Profile           |   0.86     |    0.93    |    0.86    | 0.86 | 0.94 | 0.87 |    0.79     |    0.90     |    0.81     |
| Activity Timing   |   0.67     |    0.76    |    0.70    | 0.68 | 0.77 | 0.71 |    0.45     |    0.62     |    0.58     |
| Textual           |   0.71     |    0.82    |    0.75    | 0.72 | 0.82 | 0.75 |    0.44     |    0.63     |    0.59     |
| Post Embedding    |   0.85     |    0.92    |    0.86    | 0.83 | 0.91 | 0.84 |    0.00     |    0.73     |    0.50     |
| Graph Embedding   |   0.77     |    0.86    |    0.79    | 0.77 | 0.87 | 0.79 |    0.21     |    0.50     |    0.50     |
| Combination       |   0.88     |    0.95    |    0.88    | 0.88 | 0.95 | 0.88 |    0.75     |    0.89     |    0.79     |


In addition to the implemented ml model we also provide visualized results of our experiments such as roc-auc and precision vs recall curves. Furthermore, we provide the explainability of each developed models based on different feature categories.

![roc-auc curve](https://github.com/alexdrk14/TwitterSuspension/blob/main/plots/roc_curves.png?raw=true)



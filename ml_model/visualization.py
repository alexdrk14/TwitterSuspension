import shap
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class curves:
    def __init__(self):
        self.FPRs = []
        self.TPRs = []
        self.precision = []
        self.recall = []
        self.roc_auc_scores = []
        self.model_categories = []
        self.dest_folder = "../plots/"

    def append_new(self, model, X_train, Y_train, X_test, Y_test, features, category_name, category):
        """Scale data"""
        scaller = StandardScaler()
        train_X = pd.DataFrame(scaller.fit_transform(X_train[features]), columns=X_train[features].columns.to_list())
        test_X = pd.DataFrame(scaller.transform(X_test[features]), columns=X_test[features].columns.to_list())

        """Fit train data into model"""
        model.fit(train_X, Y_train)

        """Get prediction probabilities of test data"""
        y_pred_proba = model.predict_proba(test_X)[:, 1]

        """Compute and store FalsePositiveRate and TruePositiveRate"""
        fpr, tpr, _ = roc_curve(Y_test, y_pred_proba)

        self.FPRs.append(fpr)
        self.TPRs.append(tpr)
        self.model_categories.append(category_name)
        """Compute and store Precision and Recall"""
        precision, recall, _ = precision_recall_curve(Y_test, y_pred_proba)
        self.precision.append(precision)
        self.recall.append(recall)

        """Compute and store ROC-AUC score"""
        self.roc_auc_scores.append(roc_auc_score(Y_test, y_pred_proba))

        self.plot_shap(model, test_X, category)


    def plot_roc(self):
        plt.clf()
        fig = plt.figure(figsize=(6, 6))
        for fpr, tpr, auc, label in zip(self.FPRs, self.TPRs, self.roc_auc_scores, self.model_categories):
            plt.plot(fpr, tpr, label="{} AUC:{:.2f}".format(label, auc))

        plt.xlabel('False Positive rate', fontsize=13)
        plt.ylabel('True Positive rate ', fontsize=13)
        plt.title('ROC curves ', fontsize=15)
        plt.legend(loc='best')
        plt.savefig(self.dest_folder + "roc_curves.png", bbox_inches='tight', dpi=600, facecolor='w')

    def plot_pr(self):
        plt.clf()
        fig = plt.figure(figsize=(6, 6))
        for precision, recall, auc, label in zip(self.precision, self.recall, self.roc_auc_scores, self.model_categories):
            plt.plot(recall, precision, label="{} AUC:{:.2f}".format(label, auc))

        plt.xlabel('Recall', fontsize=13)
        plt.ylabel('Precision', fontsize=13)
        plt.title('PR curves ', fontsize=15)
        plt.legend(loc='best')
        plt.savefig(self.dest_folder + "pr_curves.png", bbox_inches='tight', dpi=600, facecolor='w')

    def plot_shap(self, model, X_data, feature_category):
        """Create shap tree explainer based on trained model"""
        shap_values_xgb = shap.TreeExplainer(model).shap_values(X_data)

        """Store figure with shap values"""
        fig = plt.figure()
        shap.summary_plot(shap_values_xgb[1], X_data)
        fig.savefig(self.dest_folder + "shap_{}.png".format(feature_category),
                    bbox_inches='tight', dpi=600, facecolor='w')

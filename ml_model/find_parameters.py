""""####################################################################################################################
Author: Alexander Shevtsov ICS-FORTH
E-mail: shevtsov@ics.forth.gr
-----------------------------------
Parameter fine-tuning and Feature selection for each model of feature categories
####################################################################################################################"""
import sys, ast
import numpy as np
import pandas as pd

from datetime import datetime
from os import path, makedirs
from collections import defaultdict

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.model import Model
from utils.dataSplit import DataBalancer

sys.path.insert(1, '../utils/')
from progBar import PBar


class FineTuning:
    def __init__(self, shuffle=True, n_splits=5, feature_categ="",
                 model_category="xgboost", gpu=False,
                 verbose=False, first_portion=True, lasso_ReRun = False,
                 number_of_models=150):

        self.data_path = "../data/"
        self.param_path = "parameters/{}/".format(feature_categ)
        self.lasso_value_filename = "lasso_alpha.txt"
        self.selFeat_filename = "selected_features.txt"
        self.filename = "{}_features.csv".format(feature_categ)

        """Check if feature category result folder is exist. In other case create it."""
        if not path.isdir(self.param_path[:-1]):
            makedirs(self.param_path[:-1])

        self.verbose = verbose
        self.shuffle = shuffle
        self.k_fold_splits = n_splits
        self.feature_categ = feature_categ

        self.lasso_alpha = None
        self.performance_train = defaultdict(lambda: [])
        self.performance_valid = defaultdict(lambda: [])

        """read data at initialization of Class item"""
        self.balancer = DataBalancer(verbose=self.verbose, first_portion=first_portion)

        self.X_visible, self.Y_visible, X_test, self.Y_test = self.balancer.load_dataset(
                                                        self.filename)

        """Load already existing features or in case of rerun execute lasso feature selection and store alpha and selected features"""
        self.feature_selection(rerun=lasso_ReRun)

        """Use already selected features"""
        self.X_visible = self.X_visible[self.features].copy()
        self.X_test = X_test[self.features].copy()
        del(X_test)

        """In case of second data portion we also need to load data from following 21 days period. And use them as new test portion."""
        if not first_portion:
            X_second, self.Y_second = self.balancer.load_dataset(
                self.filename, first=False)
            self.X_second = X_second[self.features].copy()
            del(X_second)


        self.model_class = Model(model_type=model_category, gpu=gpu, nmbr_to_select=number_of_models, feature_category=feature_categ)
        """Progress bar parameters"""
        self.pbar = PBar(all_items=number_of_models, print_by=1)
        if self.verbose:
            if first_portion: print("Model works on first data portion between 23/Feb/2022 and 16/March/2022")
            else: print("Model works on second data portion between 16/March/2022 and 06/April/2022 and model is trained over first data portion.")

    """Data scaling function"""
    def scale(self, train, test=None):
        scaller = StandardScaler()
        train_scaled = pd.DataFrame(scaller.fit_transform(train.copy()),
                                    columns=train.columns.to_list())
        if type(test) != None:
            test_scaled = pd.DataFrame(scaller.transform(test.copy()),
                                   columns=test.columns.to_list())
            return train_scaled, test_scaled
        else:
            return train_scaled

    def get_scores(self, true_Y, predicted_Y, predicted_prob_Y):
        f1 = f1_score(true_Y, predicted_Y)
        roc_auc = roc_auc_score(true_Y, predicted_prob_Y[:, 1])
        accuracy = accuracy_score(true_Y, predicted_Y)
        return f1, roc_auc, accuracy

    """
        Function for feature selection with use of sklearn pipeline. 
        Select best alpha parameter of Lasso during K-Fold cross validation and keep best alpha.
        Based on selected alpha we fit visible dataset portion into Lasso model and 
        keep features with positive importance.
    """
    def feature_selection(self, rerun):
        if self.verbose:
            print("Fine tune Lasso feature selection ...")
        if self.verbose and rerun:
            print("\tForce re-run of lasso feature selection")
        """
            Check if lasso was already fine-tuned for this feature category
            just load lasso value from file and get best Alpha for LassoFS
        """
        if not rerun and path.isfile(self.param_path + self.lasso_value_filename):
            self.lasso_alpha = ast.literal_eval(open(self.param_path + self.lasso_value_filename, "r+").read())
        else:
            lasso_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', Lasso(max_iter=50000))
            ])
            search = GridSearchCV(lasso_pipeline,
                                  {'model__alpha': np.arange(0.00001, 0.003, 0.00001)},
                                  cv=5, scoring="neg_mean_squared_error", verbose=10, n_jobs=5
                                  )
            search.fit(self.X_visible, self.Y_visible)
            self.lasso_alpha = search.best_params_

            if self.verbose:
                print("\tLasso alpha:{}".format(self.lasso_alpha))

            f_out = open(self.param_path + self.lasso_value_filename, "w+")
            f_out.write("{}".format(self.lasso_alpha))
            f_out.close()

        """
            Get most important features from already stored file or by fitting LassoFS with selected Alpha.
        """
        if not rerun and path.isfile(self.param_path + self.selFeat_filename):
            self.features = ast.literal_eval(open(self.param_path + self.selFeat_filename, "r").read())
        else:
            lasso_model = Lasso(alpha=self.lasso_alpha['model__alpha'], max_iter=50000)
            lasso_model.fit(self.X_visible, self.Y_visible)
            importance = np.abs(lasso_model.coef_)
            self.features = self.X_visible.columns[importance > 0].to_list()
            f_out = open(self.param_path + self.selFeat_filename, "w+")
            f_out.write("{}".format(self.features))
            f_out.close()
        if self.verbose:
            print("\tDone Lasso fine-tuning with best alpha:{} with:{} of {} features".format(
                            self.lasso_alpha, len(self.features), self.X_visible.shape[1]))

    """Model fine-tuning and selection best configuration based on F1 during K-Fold cross validation"""
    def model_finetune(self):
        if self.verbose:
            print("Model FineTuning ... \n\tStart time:{}".format(datetime.now()))

        """create k_fold object"""
        k_folds = StratifiedKFold(n_splits=self.k_fold_splits,
                                  shuffle=self.shuffle)
        k_folds_scaled_data = []
        for train_index, val_index in k_folds.split(self.X_visible, self.Y_visible):
            """make train portion where K-1 folds are selected and validation portion with the fold outside of K-1"""
            train_X, val_X = self.X_visible.iloc[train_index, :], self.X_visible.iloc[val_index, :]
            train_Y, val_Y = self.Y_visible.iloc[train_index, ], self.Y_visible.iloc[val_index, ]

            """scale data"""
            train_X, val_X = self.scale(train_X, val_X)

            k_folds_scaled_data.append((train_X, val_X, train_Y, val_Y))

        for model_config in self.model_class.parameters:
            for train_X, val_X, train_Y, val_Y in k_folds_scaled_data:
                """Create model with particular parameters"""
                self.model_class.create_model(model_config)
                
                train_YP, val_YP = self.model_class.train_predict(train_X, train_Y, val_X)
                
                f1_train = f1_score(train_Y, train_YP)
                f1_val = f1_score(val_Y, val_YP)

                """Store performance of this setup in order to get best setup and number of features"""
                self.performance_train[model_config].append(f1_train)
                self.performance_valid[model_config].append(f1_val)

                if self.verbose:
                    print("\t F1 train:{} F1 val:{} for config:{}".format(f1_train, f1_val, model_config))

            """Print the progress bar"""
            self.pbar.increase_done()

        del(k_folds_scaled_data)
        self.select_best_setup()


    """Measure performance of test data portion"""
    def test_performance(self):

        if self.verbose:
            print("Measure of Model Performance... \n\tStart time:{}".format(datetime.now()))
        """Load model parameters"""
        self.model_class.load_model()

        """##########################################################################
        Measure again the average validation performance with k-fold cross validation
        over the train/validation dataset (Visible)
        At this time we store: F1, ROC-AUC and accuracy
        ##########################################################################"""
        """create k_fold object"""
        k_folds = StratifiedKFold(n_splits=self.k_fold_splits,
                                  shuffle=self.shuffle)
        train_perf = []
        val_perf = []
        
        for train_index, val_index in k_folds.split(self.X_visible, self.Y_visible):
            #make train portion where K-1 folds are selected and validation portion with the fold outside of K-1
            train_X, val_X = self.X_visible.iloc[train_index, :], self.X_visible.iloc[val_index, :]
            train_Y, val_Y = self.Y_visible.iloc[train_index,], self.Y_visible.iloc[val_index,]

            """scale data"""
            train_X, val_X = self.scale(train_X, val_X)

            train_YP, val_YP = self.model_class.train_predict(train_X, train_Y, val_X)
            
            train_perf.append( (self.get_scores(train_Y, train_YP, self.model_class.model.predict_proba(train_X))) )
            val_perf.append( (self.get_scores(val_Y, val_YP, self.model_class.model.predict_proba(val_X))) )


        f_out = open(self.param_path + "test_performance.txt", "w+")
        
        #Store performance in output file
        f_out.write("K-fold performance (avg) F1-train:{:.3f} ROC-AUC-train:{:.3f} Acc-train:{:.3f}\n".format(
            sum([item[0] for item in train_perf]) / self.k_fold_splits,
            sum([item[1] for item in train_perf]) / self.k_fold_splits,
            sum([item[2] for item in train_perf]) / self.k_fold_splits
        ))
        f_out.write("K-fold performance (avg) F1-val:{:.3f} ROC-AUC-val:{:.3f} Acc-val:{:.3f}\n".format(
            sum([item[0] for item in val_perf]) / self.k_fold_splits,
            sum([item[1] for item in val_perf]) / self.k_fold_splits,
            sum([item[2] for item in val_perf]) / self.k_fold_splits
        ))
        
        """##########################################################################
        Now measure performance of test dataset, with model trained on entire train/validation set (visible)
        ##########################################################################"""
        """scale data"""
        
        train_X, test_X = self.scale(self.X_visible, self.X_test)

        train_YP, test_YP = self.model_class.train_predict(train_X, self.Y_visible, test_X)

        f1_train, roc_auc_train, acc_train = self.get_scores(self.Y_visible, train_YP,
                                                        self.model_class.model.predict_proba(train_X))
        f1_test, roc_auc_test, acc_test = self.get_scores(self.Y_test, test_YP,
                                                        self.model_class.model.predict_proba(test_X))
    
        """Store performance in output file"""
        
        f_out.write("First test performance F1-train:{:.3f} ROC-AUC-train:{:.3f} Acc-train:{:.3f}\n".format(
            f1_train, roc_auc_train, acc_train
        ))
        f_out.write("First test performance F1-test:{:.3f} ROC-AUC-test:{:.3f} Acc-test:{:.3f}\n".format(
            f1_test, roc_auc_test, acc_test
        ))
        

        """Train model over the entire dataset of first 21 days (visible and hidden) and test it on second test (second 21 days data)
            For this reason we firstly merge visible and hidden dataset of first 21 days data.
        """
        
        """scale data"""
        train_X, second_test_X = self.scale(pd.concat([self.X_visible, self.X_test], axis=0), self.X_second)

        train_Y = pd.concat([self.Y_visible, self.Y_test], axis=0)

        train_YP, second_test_YP = self.model_class.train_predict(train_X, train_Y, second_test_X)

        f1_train, roc_auc_train, acc_train = self.get_scores(train_Y, train_YP,
                                                        self.model_class.model.predict_proba(train_X))
        f1_second_test, roc_auc_second_test, acc_second_test = self.get_scores(self.Y_second, second_test_YP,
                                                        self.model_class.model.predict_proba(second_test_X))

        if self.verbose:
            print("\tMeasuring is done.")
        """Store performance in output file"""
        f_out.write("Second test performance F1-train:{:.3f} ROC-AUC-train:{:.3f} Acc-train:{:.3f}\n".format(
            f1_train, roc_auc_train, acc_train
        ))
        f_out.write("Second test performance F1-test:{:.3f} ROC-AUC-test:{:.3f} Acc-test:{:.3f}\n".format(
            f1_second_test, roc_auc_second_test, acc_second_test
        ))

        f_out.close()



    def select_best_setup(self):
        if self.verbose:
            print("Selecting the best setup based on validation avg F1 score")
        best_avg_perf = 0.0
        best_setup = None
        for setup in self.performance_valid:
            avg_perf = sum(self.performance_valid[setup]) / len(self.performance_valid[setup])
            if avg_perf > best_avg_perf:
                best_avg_perf = avg_perf
                best_setup = setup
        best_avg_perf_train = sum(self.performance_train[best_setup]) / len(self.performance_train[best_setup])
        if self.verbose:
            print("\tBest setup params: {}\n\twith val f1:{} and train f1:{}".format(
            best_setup, best_avg_perf, best_avg_perf_train))
        """Store results in parameter path folder"""
        f_out = open(self.param_path + "fine_tuning_best_model.txt", "w+")
        f_out.write("{}\nwith val avg f1:{} and train avg f1:{}".format(
            best_setup,
            best_avg_perf,
            best_avg_perf_train))
        f_out.close()

        """Store results in log file in form of csv file with tab separator"""
        #f_out = open(self.param_path + "model_fine_tuning_logs.csv", "w+")
        #f_out.write("Setup\t")
        #for i in range(self.k_fold_splits):
        #    f_out.write("Train_Fold_{}\tVal_Fold_{}\t".format(i, i))
        #f_out.write("AVG_train_f1\tAVG_val_f1\n")

        for setup in self.performance_train:
            f_out.write(setup)
            for i in range(len(self.performance_train[setup])):
                f_out.write("\t{}\t{}".format(self.performance_train[setup][i], self.performance_valid[setup][i]))
            f_out.write("\t{}\t{}\n".format(sum(self.performance_train[setup]) / len(self.performance_train[setup]),
                                            sum(self.performance_valid[setup]) / len(self.performance_valid[setup])))
        f_out.close()

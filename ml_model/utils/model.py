import sys
import random
import ast
import numpy as np 

from xgboost import XGBClassifier
from utils.model_params import model_parameters_range


"""
   Model class that store the entier model and configurational set of parameters, 
   used in order to reduce complexity of fine-tuning method
"""
class Model:

    def __init__(self, model_type=None, gpu=False, nmbr_to_select=0, feature_category=""):
        if model_type not in model_parameters_range:
            print("Unknown model type:{} \n\t Please use the following models:{} \n\tOr define new model and params!".format(
                model_type, model_parameters_range.keys() ))
            sys.exit(-1)

        self.model_type = model_type
        self.model_params = model_parameters_range[model_type]

        self.model = None
        self.use_gpu = gpu
        self.parameters = []
        self.feature_category = feature_category

        if nmbr_to_select > 0:
            self.create_parameters_list(nmbr_to_select)

    def create_parameters_list(self, select):
        if self.model_type == "xgboost":
            """Fine tune model parameters according to selected parameters range in self.model_params"""
            for min_child_weight in self.model_params["min_child_weight"]:
                for lr in self.model_params["learning_rate"]:
                    for n_est in self.model_params["n_estimators"]:
                        for m_depth in self.model_params["max_depth"]:
                            for gamma in self.model_params["gamma"]:
                                for reg_lambda in self.model_params["reg_lambda"]:
                                    for subsample in self.model_params["subsample"]:
                                        for col_tree in self.model_params["colsample_bytree"]:
                                            self.parameters.append("{"+"'learning_rate':{}, 'n_estimators':{}, 'max_depth':{}, 'gamma':{}, 'subsample':{}, 'colsample_bytree':{}, 'min_child_weight':{}, 'reg_lambda':{}".format(
                                                lr, n_est, m_depth,gamma, subsample,col_tree,min_child_weight, reg_lambda)+"}")
        """Select at random K setup from all possible paramtere combination"""
        random.shuffle(self.parameters)
        self.parameters = random.sample(self.parameters, select)

    def create_model(self, params):
       if "xgboost" in self.model_type:
            """Free previous model"""
            if self.model != None:
                del(self.model)
            tree_method = 'auto'
            predictor = 'auto'

            if self.use_gpu:
                tree_method = "gpu_hist"
                predictor = 'gpu_predictor'


            config = ast.literal_eval(params)

            self.model = XGBClassifier(objective="multi:softprob", num_class=2,
                                       learning_rate=config['learning_rate'],
                                       n_estimators=config['n_estimators'],
                                       max_depth=config['max_depth'],
                                       gamma=config['gamma'],
                                       subsample=config['subsample'],
                                       colsample_bytree=config['colsample_bytree'],
                                       min_child_weight=config['min_child_weight'],
                                       reg_lambda=config['reg_lambda'],
                                       eval_metric='mlogloss',
                                       tree_method=tree_method, predictor=predictor,
                                       missing=None,
                                       use_label_encoder=False)

    def load_model(self):
        """Load selected parameters from fine-tuned model for particular
        feature category and create XGBoost model based on those parameters"""
        params = open("parameters/{}/fine_tuning_best_model.txt".format(self.feature_category), "r").read().split("\n")[0]
        self.create_model(params)

    def train(self, X, Y):
        self.model.fit(X, Y)

    def train_predict(self, X_train, Y_train, X_val):
        self.model.fit(X_train, Y_train)
        YP_train = self.model.predict(X_train).astype(int)
        YP_val = self.model.predict(X_val).astype(int)
        
        YP_train = YP_train[:, 1] if type(YP_train[0]) != np.int64 else YP_train
        YP_val = YP_val[:, 1] if type(YP_val[0]) != np.int64 else YP_val
        return YP_train, YP_val

    def predict(self, X):
        
        return self.model.predict(X)



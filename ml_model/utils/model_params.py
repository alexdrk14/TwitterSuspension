
model_parameters_range = {"xgboost": {
                                    'max_depth': [6, 7, 8, 9],
                                    'learning_rate': [0.005, 0.01, 0.015],
                                    'subsample': [0.65, 0.7, 0.75, 0.8],
                                    'colsample_bytree': [0.65, 0.7, 0.75],
                                    'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
                                    'gamma': [0, 0.25, 0.5, 1.0],
                                    'reg_lambda': [0.1, 1.0, 5.0],
                                    'n_estimators': [1000, 1500, 2000]
                        }
}

FCateg_translator = {"profile": 0,
                     "textual": 1,
                     "activity_timing": 2,
                     "graph_embeddings": 3,
                     "post_embeddings": 4,
                     "combination": 5}


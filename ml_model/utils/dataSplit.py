import pandas as pd
import numpy as np
from os import path
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from utils.model_params import FCateg_translator

class DataBalancer:

    def __init__(self, test_size=0.3, verbose=False, first_portion=True):
        """Define all type of features filenames (output files from feature extraction methods)"""

        self.filenames = [feature_cat + "_features.csv" for feature_cat in list(FCateg_translator.keys()) ]
        self.data_path = "../data/"
        self.mergedDF = None
        self.test_size = test_size
        self.shuffle = True
        self.verbose = verbose
        self.first_portion = first_portion
        self.main()


    """Check if features is already aligned or required pre-processing
        Return values: 0 (Don't require) 
                       1 (Required for all files)
                       2 (Required for complex df files: post embedding and combined )
    """
    def require_processing(self, first=True):
        if self.verbose:
            print("Data Loading: Required preprocecing function\n\tCheck of visible and hidden csv files for {} 21 days perdio".format("first" if first else "second"))

        filenames = self.filenames if first else [filename.replace(".csv", "_second21_balanced.csv") for filename in self.filenames]
        file_categories = [filenames[:-2], filenames[-2:]]
        for category_index in range(len(file_categories)):
            for filename in file_categories[category_index]:
                to_check = [filename] if not first else [filename.replace(".csv", "_visible.csv"),
                                                         filename.replace(".csv", "_hidden.csv")]
                if False in [path.isfile(self.data_path + file) for file in to_check]:
                    return category_index + 1
        return 0


    def prepare(self, proc_flag, first=True):
        if self.verbose:
            print("\tIn prepare function for {} 21 days".format("first" if first else "second"))
        filenames = self.filenames[:-2] if first else [filename.replace(".csv", "_second21.csv") for filename in self.filenames[:-2]]
        if proc_flag == 1:
            if self.verbose:
                print("\tPreparation of simple features")
            self.merge_files(filenames=filenames)
            self.balance()
            self.split_and_store(filenames=filenames, first=first)

        if self.verbose:
            print("\tPreparation of complex")
        self.split_and_store_complex(first=first)


    """
        Merge multiple feature categories files in single dataframe.
        Required since we extract features in separate scripts and not all users have all features categories
        and also csv files are not aligned by user_ids.
        Implementation require alignment based on user id between different models.
    """
    def merge_files(self, filenames=None):
        dataFrames = []
        self.features = []
        if filenames == None:
            filenames = self.filenames
        if self.verbose: print("\tMerge files")

        """Read all feature categories files in multiple dataframes"""
        for fileName in filenames:
            print(f'Merge with {fileName} file')
            dataFrames.append(pd.read_csv(self.data_path + fileName, sep="\t"))
            """Store feature names for each feature category"""
            self.features.append(dataFrames[-1].columns.to_list())
        """
            Remove target from all dataframe except the last one. 
            In such way we keep only one column (user_id) that is common between all DFs
        """
        self.mergedDF = dataFrames[-1].copy()
        
        for df in dataFrames[:-1]:
            df.drop(["target"], axis=1, inplace=True)
            """ Merge dataframes on user_id column. 
                That will keep only intersection of users between files and create same order 
                between all users in multiple feature files.
            """
            self.mergedDF = self.mergedDF.merge(df, how="inner", on="user_id", suffixes=(False, False))

        """shuffle the data"""
        self.mergedDF = self.mergedDF.sample(frac=1, replace=False)
        if self.verbose: print("\tMerge DF shape:{}".format(self.mergedDF.shape))


    """Make the balanced dataset by under-sampling the most frequent class of normal users"""
    def balance(self):
        if self.verbose: print("\tBefore balancing class 0:{} class 1:{}".format(
                    self.mergedDF[self.mergedDF["target"] == 0].shape[0],
                    self.mergedDF[self.mergedDF["target"] == 1].shape[0]))

        """Get all normal users"""
        normal = self.mergedDF[self.mergedDF["target"] == 0].copy()

        """Get all suspend users"""
        suspend = self.mergedDF[self.mergedDF["target"] == 1].copy()
        normal = normal.sample(suspend.shape[0], replace=False)
        del(self.mergedDF)

        self.mergedDF = pd.concat([normal, suspend], ignore_index=True)
        del(normal)
        del(suspend)
        if self.verbose: print("\tAfter balancing class 0:{} class 1:{}".format(
            self.mergedDF[self.mergedDF["target"] == 0].shape[0],
            self.mergedDF[self.mergedDF["target"] == 1].shape[0]))


    """
        Split Dataframe into two portions with *_visible.csv (train/val) and *_hidden.csv (test) postfix.
        This procedure is performed for each feature categories since 
        different model type require different category of features.
    """
    def split_and_store(self, filenames=None, first=True):
        if filenames == None:
            filenames = self.filenames
        if self.verbose: print("\tSplit and store")

        if first:
            """ Stratified split dataset into two portions
                Visible (Train/Validation) and Test portion as a hold-out
            """
            X_visible, X_test, _, _ = train_test_split(self.mergedDF,
                                                self.mergedDF["target"], test_size=self.test_size,
                                                shuffle=self.shuffle, stratify=self.mergedDF["target"])

        else:
            self.mergedDF = self.mergedDF.sample(frac=1, replace=False)

        for features, filename in zip(self.features, filenames):
            if first:
                """Store dataframes in separate files for each feature category as visible and hidden portion"""
                X_visible[features].to_csv(self.data_path + filename.replace(".csv", "_visible.csv"),
                                                sep="\t", index=False)
                X_test[features].to_csv(self.data_path + filename.replace(".csv", "_hidden.csv"),
                                             sep="\t", index=False)
            else:
                """Store dataframes in separate files for each feature category as visible and hidden portion"""
                self.mergedDF[features].to_csv(self.data_path + filename.replace(".csv", "_balanced.csv"),
                                           sep="\t", index=False)
        """Free memory"""
        del(self.mergedDF)


    """Split and store the complex feature categories"""
    def split_and_store_complex(self, first=True):
        """Split and store for post feature category"""
        filename = self.filenames[-2] #post_embeddings_features.csv
        post_file = filename.replace(".csv", "_visible.csv") if first else filename.replace(".csv", "_second21_balanced.csv")
        #"post_embeddings_features_visible.csv" if first else "post_embeddings_features_second21_balanced.csv"
        #post_file = "post_features_visible.csv" if first else "post_features_second21_balanced.csv"
        post_fix = ["_visible.csv", "_hidden.csv"] if first else ["_second21_balanced.csv"]
        user_ids = []
        if not path.isfile(self.data_path + post_file):
            """Read user ids for each category (visible hidden in case of first 21 days ) 
               or second21_balanced in other case"""
            for PFix in post_fix:
                user_ids.append(pd.read_csv(self.data_path +
                                            self.filenames[0].replace(".csv", PFix),
                                            sep="\t")["user_id"].values.tolist())

            post_data = pd.read_csv(self.data_path + (filename if first else filename.replace(".csv", "_second21.csv")), sep="\t")
            """Under-sample visible and hidden data portions and store in post files"""
            undersample = RandomUnderSampler(sampling_strategy='majority')

            for group_users, PFix in zip(user_ids, post_fix):
                X = post_data[post_data["user_id"].isin(group_users)].copy()
                X.reset_index()
                X, _ = undersample.fit_resample(X, X["target"])
                output_file = filename.replace(".csv", PFix)
                X.drop(["created_at"], axis=1, inplace=True)
                X.to_csv(self.data_path + output_file,
                                 sep="\t", index=False)
                del(X)
            del(user_ids)
            del(post_data)


        """Split and store for single feature category that contain all combination of feature categories"""
        """This data portion isn't require the balancing since it based on already balanced csv files"""
        filename = self.filenames[-1] #combination_features.csv
        single_file = filename.replace(".csv", "_visible.csv") if first else filename.replace(".csv", "_second21_balanced.csv")
        #single_file = "single_features_visible.csv" if first else "single_features_second21_balanced.csv"
        post_fix = ["_visible.csv", "_hidden.csv"] if first else ["_second21_balanced.csv"]
        if not path.isfile(self.data_path + single_file):
            for PFix in post_fix:
                """Merge files with simple features"""
                self.merge_files(
                    filenames=[dataset_filename.replace(".csv", PFix) for dataset_filename in self.filenames[:-2]])
                #X = self.mergedDF.copy()
                #del (self.mergedDF)
                self.mergedDF = self.append_post_features(self.mergedDF, first_portion=first)
                self.mergedDF.reset_index()
                output_file = filename.replace(".csv", PFix)
                self.mergedDF.to_csv(self.data_path + output_file,
                                 sep="\t", index=False)
                #del(X)
                del(self.mergedDF)

    def _load_csv_file(self, filename):
        X = pd.read_csv(self.data_path + filename, sep="\t")
        Y = X["target"].copy()
        X.drop(["target", "user_id"], axis=1, inplace=True)
        X.replace([np.inf, -np.inf], 0, inplace=True)
        return X, Y



    def load_dataset(self, filename, first=True):
        if not first:
            filename = filename.replace(".csv", "_second21.csv")

        if self.verbose:
            print("DataBalancer: Loading {} dataset for {} 21 days...".format(filename, "first" if first else "second"))

        X, Y = self._load_csv_file(filename.replace(".csv", "_visible.csv" if first else "_balanced.csv"))

        if first:
            X_test, Y_test = self._load_csv_file(filename.replace(".csv", "_hidden.csv"))

            if self.verbose:
                print("DataBalancer: Loaded dataset with:" +
                      "\n\tVisible portion, class 0:{} and 1:{}".format(
                          sum(Y == 0), sum(Y == 1)) +
                      "\n\t Hidden portiona, class 0:{} and 1:{}".format(
                          sum(Y_test == 0), sum(Y_test == 1)))
            return X, Y, X_test, Y_test

        if self.verbose:
            print("DataBalancer: Loaded dataset with:" +
                  "\n\t class 0:{} and 1:{}".format(
                                sum(Y == 0), sum(Y == 1)))
        return X, Y


    def append_post_features(self, X, first_portion=True):
        """Create dataframe with post features"""
        samples = X.shape[0]

        emb_size = len([fname for fname in open(self.data_path + self.filenames[-2].replace(".csv", "_visible.csv"), "r").readline().split("\t") if "text_emb_" in fname])
        feature_space = (emb_size + 1) * 3# we use all embeddings size + 1 feature for category of user text (retweet , mention or quote)
        feature_names = ["last_{}_post_emb_{}".format(i, j) if j != emb_size else "last_{}_post_category".format(i) for i in range(1, 4) for j in range(emb_size+1)]
        post_features = pd.DataFrame(np.zeros(( samples, feature_space)), columns=feature_names)

        post_data = pd.read_csv(self.data_path + self.filenames[-2], sep="\t")#pd.read_csv(self.data_path + "post_features_all.csv", sep="\t")
        post_data.drop(["tweet_id"], axis=1, inplace=True)

        """Store exact start and end dates for feature extraction"""
        start, end = (20220223, 20220316) if first_portion else (20220316, 20220406)

        post_data = post_data[(post_data['created_at'] >= start) & (post_data['created_at'] < end)]
        users = X['user_id'].values.tolist()
        for index in range(len(users)):
            vector = []
            user_posts = post_data[post_data['user_id'] == users[index]].sort_values(by=['created_at'], ascending=False).copy()
            user_posts.drop(["user_id", "created_at", "target"], axis=1, inplace=True)
            for i in range(user_posts.shape[0] if user_posts.shape[0] <= 3 else 3):
                vector += user_posts.iloc[i].values.tolist()
            if user_posts.shape[0] < 3:
                for i in range(3 - user_posts.shape[0]):
                    vector += [0]*(emb_size + 1) # create empty vector since user dont have i-th post information

            post_features.iloc[index] = vector
        
        post_features["user_id"] = users
        #post_features.to_csv(self.data_path + "appendix_post.csv",
        #                                 sep="\t", index=False)
            

        return X.merge(post_features, how="inner", on="user_id", suffixes=(False, False))


    def main(self):
        if self.verbose:
            print("Data Balancer initialization")

        for first in [True, False]:
            proc_flag = self.require_processing(first=first)
            if proc_flag != 0:
                if self.verbose:
                    print("\tRequire balancing for {} 21 days data portion for {} files".format(
                                                        "first" if first else "second",
                                                        "all" if proc_flag == 1 else "complex"))
                self.prepare(proc_flag=proc_flag, first=first)
                if self.verbose:
                    print("\tComplete")






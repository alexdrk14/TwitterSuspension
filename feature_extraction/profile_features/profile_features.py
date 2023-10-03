""""####################################################################################################################
Author: Alexander Shevtsov ICS-FORTH
E-mail: shevtsov@ics.forth.gr
-----------------------------------
Feature extraction based on user profile history timeline. Store collected features in csv file.
####################################################################################################################"""

import numpy as np
import argparse, sys
from datetime import datetime, timedelta
from dateutil.parser import parse
sys.path.insert(1, '../../utils/')
from mongoConnector import MongoDB
from progBar import PBar
from labels_loading import load_labels

DATA_PATH = "../../data/"


class profile_features:
    """first_weeks show if we extract feature for first 3 weeks of our data or we extract feature after first 3 weeks"""
    def __init__(self, first_weeks=True, verbose=False):
        self.labels = None
        self.verbose = verbose
        """MongoDB connection class"""
        self.connector = MongoDB()
        self.first_weeks = first_weeks
                
        if self.verbose:
           print("Extraction of profile features for {} 21 days".format("first" if first_weeks else "second"))
        
        if first_weeks:
            """Extraction of first 21 days portion. Used for train/val/test"""
            self.output_filename = DATA_PATH + "profile_features.csv"
            self.start_date = datetime(2022, 2, 23, 0, 0, 0)
            
        else:
            """Extraction of second 21 days portion. Used for future model eval"""
            self.output_filename = DATA_PATH + "profile_features_second21.csv"
            self.start_date = datetime(2022, 2, 23, 0, 0, 0) + timedelta(days=21)
        
        self.end_date = self.start_date + timedelta(days=21)

    def user_object_features(self, user_id):
        """Get user history item from collection"""
        #print(user_id)
        client, db = self.connector.connect()
        user_history = db.usersHistory.find_one({"user_id": user_id})
        client.close()
        user_data = dict()
        first_obj = None
        last_obj = None
        activity_day_start = None
        activity_day_end = None

        """
            Parse user objects history starting from starting date, until the last day of monitoring.
        """
        date_pointer = self.start_date
        while date_pointer < self.end_date:
            date = str(date_pointer.year * 10000 + date_pointer.month * 100 + date_pointer.day)
            if date in user_history:
                if first_obj == None:
                    first_obj = user_history[date]
                    activity_day_start = date_pointer
                last_obj = user_history[date]
                activity_day_end = date_pointer
            date_pointer += timedelta(days=1)

        if first_obj == None:
            return None

        """Measure account activity time in hours"""
        user_data["activity_time_range"] = (activity_day_end - activity_day_start).total_seconds() / 3600

        """Measure account age in days"""
        first_activity_day = parse(last_obj["created_at"]) if type(last_obj["created_at"]) == str else last_obj["created_at"]
        first_activity_day = first_activity_day.replace(tzinfo=None)
        user_data["age"] = float((activity_day_end - first_activity_day).days)

        for feature_name in ["favourites", "listed", "statuses", "followers", "friends"]:
            user_data[feature_name] = last_obj[feature_name + "_count"]
            """claculate activity divided by age of account in order to find growth by registered days"""
            user_data[feature_name + "_by_age"] = user_data[feature_name] / user_data["age"] if user_data["age"] != 0.0 else np.inf
            """claculate percentage growth during monitoring period"""
            user_data[feature_name + "_growth"] = ((user_data[feature_name] - first_obj[feature_name + "_count"]) /
                                                        first_obj[feature_name + "_count"] if first_obj[feature_name + "_count"] != 0
                                                        else np.inf)

        """Get len of user: name, screen_name and description"""
        for feature_name in ["name", "screen_name", "description"]:
            #print("{} {}".format(feature_name, last_obj[feature_name]))
            user_data[feature_name + "_len"] = len(last_obj[feature_name]) if last_obj[feature_name] != None else 0

        """Does user description changed during monitoring period"""
        user_data["description_changes"] = 0 if last_obj["description"] == first_obj["description"] else 1

        """Measure Jaccard similarity between screen_name and user name"""
        name_set = set(last_obj["name"].lower())
        screen_name_set = set(last_obj["screen_name"].lower())
        user_data["screen_name_sim"] = len(name_set.intersection(screen_name_set)) / \
                                            len(name_set.union(screen_name_set))

        """
            Identify and measure number , upper/lower case and special characters in 
            name, screen_name and user description
        """
        for f_categ in ["name", "screen_name", "description"]:
            feature_val = last_obj[f_categ] if last_obj[f_categ] != None else ""
            for case in ["upper", "lower", "digit", "special"]:
                if case == "upper":
                    user_data[f_categ+"_"+case+"_len"] = sum([i.isupper() for i in feature_val])
                elif case == "lower":
                    user_data[f_categ+"_"+case+"_len"] = sum([i.islower() for i in feature_val])
                elif case == "digit":
                    user_data[f_categ+"_"+case+"_len"] = sum([i.isdigit() for i in feature_val])
                else:
                    user_data[f_categ+"_"+case+"_len"] = user_data[f_categ + "_len"] - (
                            user_data[f_categ+"_upper_len"] +
                            user_data[f_categ+"_lower_len"] +
                            user_data[f_categ+"_digit_len"])
                user_data[f_categ+"_"+case+"_pcnt"] = user_data[f_categ+"_"+case+"_len"] / len(
                                                            feature_val) if len(
                                                            feature_val) != 0 else np.inf

        """Followers to  friends score"""
        user_data["foll_friends"] = (user_data["followers"] / float(user_data["friends"])) if \
                                         user_data["friends"] != 0 else np.inf

        """Boolean values from user object"""
        user_data["geo"] = 1 if last_obj["geo_enabled"] else 0
        user_data["protected"] = 1 if last_obj["protected"] else 0
        user_data["location"] = 1 if last_obj["location"] != "" else 0
        user_data["background_img"] = 1 if last_obj["profile_use_background_image"] else 0
        user_data["default_prof"] = 1 if last_obj["default_profile"] else 0
        user_data["url"] = 1 if last_obj["url"] != None else 0
        user_data["verified"] = 1 if last_obj["verified"] else 0
        user_data["user_id"] = user_id
        user_data["target"] = self.labels["label"][self.labels["user_id"] == user_id].item()
        return user_data


    def initialize_outfile(self):
        feature_names = ["activity_time_range", "age"]

        for feature_name in ["favourites", "listed", "statuses", "followers", "friends"]:
            feature_names.append(feature_name)
            feature_names.append(feature_name + "_by_age")
            feature_names.append(feature_name + "_growth")

        for feature_name in ["name", "screen_name", "description"]:
            feature_names.append(feature_name + "_len")

        feature_names.append("description_changes")

        feature_names.append("screen_name_sim")

        for f_categ in ["name", "screen_name", "description"]:
            for case in ["upper", "lower", "digit", "special"]:
                if case == "upper":
                    feature_names.append(f_categ + "_" + case + "_len")
                elif case == "lower":
                    feature_names.append(f_categ + "_" + case + "_len")
                elif case == "digit":
                    feature_names.append(f_categ + "_" + case + "_len")
                else:
                    feature_names.append(f_categ + "_" + case + "_len")
                feature_names.append(f_categ + "_" + case + "_pcnt")

        feature_names.append("foll_friends")
        feature_names.append("geo")
        feature_names.append("protected")
        feature_names.append("location")
        feature_names.append("background_img")
        feature_names.append("default_prof")
        feature_names.append("url")
        feature_names.append("verified")
        feature_names.append("user_id")
        feature_names.append("target")

        self.f_out = open(self.output_filename, "w+")
        self.f_out.write("\t".join(feature_names) + "\n")
        return feature_names

    def dump_user_vector(self, user_data, features):
        outline = ""
        for feature_name in features:
            outline += "{}\t".format(user_data[feature_name])

        self.f_out.write("{}\n".format(outline[:-1]))

    def start_extraction(self):
        if self.verbose:
            print("Starting feature extraction between: {} - {} dates.".format(self.start_date, self.end_date))
            print("Loading labels ..")

        """
            Load user labels (alive , suspended, removed and protected) and create progress bar object
        """
        self.labels, self.user_ids = load_labels(DATA_PATH, self.first_weeks)

        """Progress bar parameters"""
        self.pbar = PBar(all_items=len(self.user_ids), print_by=1000)

        if self.verbose:
            print("Find user who was active during this period.")

        """Initialize the output file with feature headers"""
        feature_names = self.initialize_outfile()

        if self.verbose:
            print("Start feature computation and extraction...")

        for user_id in self.user_ids:
            user_data = self.user_object_features(int(user_id))
            
            """Print the progress bar"""
            self.pbar.increase_done()
            if user_data == None:
                continue
            self.dump_user_vector(user_data, feature_names)

        self.f_out.close()
        if self.verbose:
            print("All features are extracted and stored in: {} file!".format(self.output_filename))

parser = argparse.ArgumentParser()
parser.add_argument('--second_portion', dest='second_weeks', default=False, action='store_true')
parser.add_argument('-v', dest='verbose', default=False, action='store_true')
args = parser.parse_args()
if __name__ == "__main__":
    first_weeks = True
    if args.second_weeks:
        first_weeks = False
    pf_extraction = profile_features(first_weeks=first_weeks, verbose=args.verbose)
    pf_extraction.start_extraction()

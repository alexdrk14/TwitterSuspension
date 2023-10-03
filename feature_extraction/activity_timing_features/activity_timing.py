""""####################################################################################################################
Author: Alexander Shevtsov ICS-FORTH
E-mail: shevtsov@ics.forth.gr
-----------------------------------
Feature extraction based on user actions in social media. Store collected features in csv file.
####################################################################################################################"""
from collections import defaultdict
import numpy as np
import argparse
import math
from dateutil.parser import parse

from datetime import datetime, timedelta
from sys import path
path.insert(1, '../../utils/')
from mongoConnector import MongoDB
from progBar import PBar
from labels_loading import load_labels

DATA_PATH = "../../data/"

class action_features:
    """first_weeks show if we extract feature for first 3 weeks of our data or we extract feature after first 3 weeks"""

    def __init__(self, first_weeks=True, verbose=False):
        self.labels = None
        self.verbose = verbose
        """MongoDB connection class"""
        self.connector = MongoDB()
        self.first_weeks = first_weeks
        self.header = list()

        if self.verbose:
            print("Extraction of ACTION features for {} 21 days".format("first" if first_weeks else "second"))

        if first_weeks:
            """Extraction of first 21 days portion. Used for train/val/test"""
            self.output_filename = DATA_PATH + "activity_timing_features.csv"
            self.start_date = datetime(2022, 2, 23, 0, 0, 0)
        else:
            """Extraction of second 21 days portion. Used for future model eval"""
            self.output_filename = DATA_PATH + "activity_timing_features_second21.csv"
            self.start_date = datetime(2022, 2, 23, 0, 0, 0) + timedelta(days=21)

        self.end_date = self.start_date + timedelta(days=21)


    def fill_post_features(self, tweet):
        tweet_type = "tw"

        if "retweeted_status" in tweet:
            # is retweet
            tweet_type = "rt"
            origin_time = parse(tweet["retweeted_status"]["created_at"]).replace(tzinfo=None) if type(
                tweet["retweeted_status"]["created_at"]) == str else tweet["retweeted_status"]["created_at"]
        elif "quoted_status" in tweet:
            tweet_type = "qt"
            origin_time = parse(tweet["quoted_status"]["created_at"]).replace(tzinfo=None) if type(
                tweet["quoted_status"]["created_at"]) == str else tweet["quoted_status"]["created_at"]

        if tweet_type != "tw":
            self.user_data_avg[tweet_type + "_time_avg"].append((tweet["created_at"] - origin_time).seconds / 60.0)

        # get week day number (Monday: 0 - Sunday: 6)
        weekDay = tweet["created_at"].weekday()
        # get hour of posted tweet/retweet (0-23)
        dayHour = tweet["created_at"].hour

        """Store user activity for week day and hour of day"""
        self.hour_day["daily_" + tweet_type][weekDay] += 1
        self.hour_day["hour_" + tweet_type][dayHour] += 1

        """Count number of seen elements (tweet, quote , retweet)"""
        self.user_data[tweet_type + "_seen"] += 1
 
    def compute_std(self, data):
        if len(data) == 0:
            return 0.0
        varience = sum([x ** 2 for x in data ]) / len(data)
        return math.sqrt(varience)


    def compute_user_times(self):
        """Compute retweet time avg/min/max/std times according to our collected data"""
        for category in ["rt_", "qt_"]:
            self.user_data[category + "time_avg"] = sum(self.user_data_avg[category + "time_avg"]) / len(self.user_data_avg[category + "time_avg"]) if len(self.user_data_avg[category + "time_avg"]) != 0 else np.inf
            self.user_data[category + "time_min"] = min(self.user_data_avg[category + "time_avg"]) if len(self.user_data_avg[category + "time_avg"]) != 0 else np.inf
            self.user_data[category + "time_max"] = max(self.user_data_avg[category + "time_avg"]) if len(self.user_data_avg[category + "time_avg"]) != 0 else np.inf
            self.user_data[category + "time_std"] = self.compute_std(self.user_data_avg[category + "time_avg"])

    def compute_daily_hourly(self):
        """Store activity (overall, tweet_only, retweet_only and mention_only by each day of week as percentage )"""
        for day in range(0, 7):
            self.user_data["daily_action_{}".format(day)] = ((self.hour_day["daily_tw"][day] +
                                                              self.hour_day["daily_rt"][day] +
                                                              self.hour_day["daily_qt"][day]) / self.all_actions)
            for category in ["rt", "tw", "qt"]:
                self.user_data["daily_{}_{}".format(category, day)] = ((self.hour_day["daily_" + category][day]) /
                                                                       self.user_data[category + "_seen"] if
                                                                       self.user_data[category + "_seen"] != 0 else np.inf)

        """Store activity (overall, tweet_only, retweet_only and mention_only by each hour of day as percentage )"""
        for hour in range(0, 24):
            self.user_data["hour_action_{}".format(hour)] = ((self.hour_day["hour_rt"][hour] +
                                                              self.hour_day["hour_tw"][hour] +
                                                              self.hour_day["hour_qt"][hour]) / self.all_actions)
            for category in ["rt", "tw", "qt"]:
                self.user_data["hour_{}_{}".format(category, hour)] = ((self.hour_day["hour_" + category][hour]) /
                                                                       self.user_data[category + "_seen"] if
                                                                       self.user_data[
                                                                           category + "_seen"] != 0 else np.inf)

    def compute_seen_perc(self):
        for category in ["rt", "tw", "qt"]:
            self.user_data[category + "_perc"] = self.user_data[category + "_seen"] / self.all_actions
        self.user_data["tw_rt_ration"] = self.user_data["tw_seen"] / self.user_data["rt_seen"] if \
            self.user_data["rt_seen"] != 0 else np.inf

    def get_user_vector(self, user_id):
        """Initialize self values and datastructures for this particular user.
           After this iteration and user vector writing we delete them in order to reduce memory usage
           and do not leak data between users.
        """
        self.user_data = defaultdict(lambda: 0)
        self.hour_day = defaultdict(lambda: defaultdict(lambda: 0))
        self.user_data_avg = defaultdict(lambda: [])
        self.all_actions = 0

        self.user_data["user_id"] = user_id
        self.user_data["target"] = self.labels["label"][self.labels["user_id"] == user_id].item()

        """Collect user posts during selected period"""
        client, db = self.connector.connect()
        for tweet in db.Tweets.find({"user.id": user_id, "created_at": {
                                    "$gte": self.start_date, "$lt": self.end_date}}, no_cursor_timeout=True):
            self.fill_post_features(tweet)
        client.close()
        """Count identified actions of user"""
        self.all_actions = sum([self.user_data[category + "_seen"] for category in ["rt", "tw", "qt"]])

        """Skip user if is there not actions in this period"""
        if self.all_actions != 0:
            """Compute user features"""
            self.compute_user_times()
            self.compute_daily_hourly()
            self.compute_seen_perc()

            """Store user vector"""
            self.dump_user_vector()

        """Clear class values of particular user"""
        del(self.user_data)
        del(self.hour_day)
        del(self.user_data_avg)

    def initialize_outfile(self):
        """Create header for retweet/tweet/quotes number of seen item during monitoring period and percentage
            Also store emoji header with min, max and average values.
        """

        for day in range(0, 7):
            self.header.append("daily_action_{}".format(day))
        for day in range(0, 7):
            for category in ["rt", "tw", "qt"]:
                self.header.append("daily_{}_{}".format(category, day))

        for hour in range(0, 24):
            self.header.append("hour_action_{}".format(hour))
        for hour in range(0, 24):
            for category in ["rt", "tw", "qt"]:
                self.header.append("hour_{}_{}".format(category, hour))

        for category in ["rt", "qt"]:
            for measure in ["avg", "min", "max", "std"]:
                self.header.append("{}_time_{}".format(category, measure))

        for group in ["seen", "perc"]:
            for category in ["rt", "tw", "qt"]:
                self.header.append("{}_{}".format(category, group))

        self.header += ["tw_rt_ration", "target", "user_id"]

        self.f_out = open(self.output_filename, "w+")
        self.f_out.write("\t".join(self.header) + "\n")


    def dump_user_vector(self):
        outline = ""
        for feature_name in self.header:
            outline += "{}\t".format(self.user_data[feature_name])
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

        if self.verbose: print("Find user who was active during this period.")

        """Initialize the output file with feature headers"""
        self.initialize_outfile()

        if self.verbose: print("Start feature computation and extraction...")

        for user_id in self.user_ids:
            self.get_user_vector(int(user_id))
            """Print the progress bar"""
            self.pbar.increase_done()


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
    pf_extraction = action_features(first_weeks=first_weeks, verbose=args.verbose)
    pf_extraction.start_extraction()

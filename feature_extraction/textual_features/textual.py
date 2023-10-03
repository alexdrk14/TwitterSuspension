""""####################################################################################################################
Author: Alexander Shevtsov ICS-FORTH
E-mail: shevtsov@ics.forth.gr
-----------------------------------
Feature extraction based on user content-actions in social media (posts). Store collected features in csv file.
####################################################################################################################"""
from collections import defaultdict
import numpy as np
from math import sqrt
import argparse

from sys import path
from datetime import datetime, timedelta
from tfidf import TFIDF
path.insert(1, '../../utils/')
from mongoConnector import MongoDB
from progBar import PBar
from labels_loading import load_labels
from utils import merge_tweet_and_retweet, text_feature

DATA_PATH = "../../data/"

class content_features:
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
            self.output_filename = DATA_PATH + "textual_features.csv"
            self.start_date = datetime(2022, 2, 23, 0, 0, 0)
        else:
            """Extraction of second 21 days portion. Used for future model eval"""
            self.output_filename = DATA_PATH + "textual_features_second21.csv"
            self.start_date = datetime(2022, 2, 23, 0, 0, 0) + timedelta(days=21)

        self.end_date = self.start_date + timedelta(days=21)

        self.tfidf = TFIDF(first_weeks=first_weeks, verbose=verbose)


    def compute_std(self, data):
        if len(data) == 0:
            return 0.0
        varience = sum([x ** 2 for x in data ]) / len(data)
        return sqrt(varience)

    def get_most_freq(self, category, action):
        """
            Get most 3 most frequent elements (in our case keys:
            hashtags/mentions/words which have highest number of usage, stored as value)
        """
        freq = [(x, self.popular["{}_{}".format(category, action )][x]) for x in self.popular["{}_{}".format(category, action)]]
        freq.sort(key=lambda x: x[1], reverse=True)
        result = [np.inf] * 3
        for i in range(0, len(freq) if len(freq) <= 3 else 3):
            result[i] = self.tfidf.compute_tf_idf(freq[i][0], category, freq[i][1])
        return result

    def post_features(self, tweet):
        tweet_type = "tw"
        """get twitter post text from object (if full_text exist use it else use just text field)"""
        tweet_text = tweet["full_text"] if "full_text" in tweet else tweet["text"]
        entities = tweet["entities"]

        if "account is temporarily unavailable because it violates the Twitter Media Policy." in tweet_text:
            """ignore this posts, it is not available"""
            return None

        if "retweeted_status" in tweet.keys():
            tweet_type = "rt"
            (tweet["retweeted_status"]["full_text"] if "full_text" in tweet["retweeted_status"] else
                                     tweet["retweeted_status"]["text"], tweet["entities"])
            tweet_text, entities = merge_tweet_and_retweet(tweet_text, (
                            tweet["retweeted_status"]["full_text"] if "full_text" in tweet["retweeted_status"] else
                                     tweet["retweeted_status"]["text"]),
                                                           tweet["entities"],
                                                           tweet["retweeted_status"]["entities"])
        elif "quoted_status" in tweet.keys():
            tweet_type = "qt"


        # get number of hashtags/mentions/urls  in user posts tweet/retweet/quote based on tweet object fields
        self.user_data_avg[tweet_type + "_hash_avg"].append(len(entities["hashtags"]))
        self.user_data_avg[tweet_type + "_ment_avg"].append(len(entities["user_mentions"]))
        self.user_data_avg[tweet_type + "_urls_avg"].append(len(entities["urls"]))


        clear_text, emojis = text_feature(tweet_text, entities)
        self.user_data_avg[tweet_type + "_emoji_avg"].append(len(emojis))

        """Get text features ..."""
        if tweet_type != "rt":
            self.vocabulary = self.vocabulary.union(set([word.lower() for word in clear_text]))

        """Store hashtags, mentions to get most popular one ..."""
        # store user hashtags used in post and user mentions
        for hs_word in entities["hashtags"]:
            self.popular["hash_" + tweet_type][hs_word["text"]] += 1
        for mnt_word in entities["user_mentions"]:
            self.popular["ment_" + tweet_type][mnt_word["screen_name"]] += 1

        self.user_data[tweet_type + "_seen"] += 1


    def user_freq_content_features(self):
        """store most frequent user hashtag/mention/word/upper_word  used in original tweets, retweets and quotes"""
        for category in ["tw", "rt", "qt"]:
            for feature_name in ["hash", "ment"]:
                result = self.get_most_freq(feature_name, category)
                for i in range(0, 3):
                    self.user_data["#{}_popular_{}_{}_tfidf".format(i+1, feature_name, category)] = result[i]


    def user_avg_std_ratio_features(self):
        """Compute avg/std on number of urls/hashtags/mentions posted by user only in their original tweet posts
           and on retweet post separately (retweet are written by other user) or quotes
        """
        for category in ["tw", "rt", "qt"]:
            for feature_name in ["urls", "hash", "ment", "emoji"]:
                categ_len = len(self.user_data_avg["{}_{}_avg".format(category,feature_name)])
                self.user_data[category + feature_name + "avg"] = (
                    sum(self.user_data_avg["{}_{}_avg".format(category, feature_name)]) / categ_len if
                    categ_len != 0 else np.inf)
                self.user_data["{}_{}_std".format(category, feature_name)] = self.compute_std(
                    self.user_data_avg["{}_{}_avg".format(category, feature_name)])
                self.user_data["{}_{}_min".format(category, feature_name)] = (
                    min(self.user_data_avg["{}_{}_avg".format(category, feature_name)]) if
                    categ_len != 0 else np.inf)
                self.user_data["{}_{}_max".format(category, feature_name)] = (
                    max(self.user_data_avg["{}_{}_avg".format(category, feature_name)]) if
                    categ_len != 0 else np.inf)




    def get_user_vector(self, user_id):
        """Initialize self values and datastructures for this particular user.
           After this iteration and user vector writing we delete them in order to reduce memory usage
           and do not leak data between users.
        """
        self.user_data = defaultdict(lambda: 0)
        self.user_data_avg = defaultdict(lambda: [])
        self.popular = defaultdict(lambda: defaultdict(lambda: 0))
        self.vocabulary = set()
        self.user_data["user_id"] = user_id
        self.user_data["target"] = self.labels["label"][self.labels["user_id"] == user_id].item()

        client, db = self.connector.connect()
        for tweet in db.Tweets.find({"user.id": user_id, "created_at": {
                                    "$gte": self.start_date, "$lt": self.end_date}}, no_cursor_timeout=True):
            self.post_features(tweet)
        client.close()

        """Count identified actions of user"""
        self.all_actions = sum([self.user_data[category + "_seen"] for category in ["rt", "tw", "qt"]])

        if self.all_actions != 0:
            self.user_data["vocabulary"] = len(self.vocabulary)
            self.user_freq_content_features()
            self.user_avg_std_ratio_features()

            """write data of this particular user"""
            self.dump_user_vector()

        """Clear class values of particular user"""
        del (self.user_data)
        del (self.user_data_avg)
        #del (self.user_freq)
        del (self.popular)
        del (self.vocabulary)

    def initialize_outfile(self):
        self.header = list()

        for category in ["tw", "rt", "qt"]:
            for ftype in ["hash", "ment", "urls", "emoji"]:
                for metric in ["min", "max", "avg", "std"]:
                    self.header.append("{}_{}_{}".format(category, ftype, metric))



        for category in ["tw", "rt", "qt"]:
            for feature_name in ["hash", "ment"]:
                for i in range(0, 3):
                    self.header.append("#{}_popular_{}_{}_tfidf".format(i + 1, feature_name, category))

        self.header += ["vocabulary", "user_id", "target"]

        self.f_out = open(self.output_filename, "w+")
        self.f_out.write("\t".join(self.header) + "\n")


    def dump_user_vector(self):
        outline = ""
        for feature_name in self.header:
            outline += "{}\t".format(self.user_data[feature_name])
        self.f_out.write("{}\n".format(outline[:-1]))

    def start_process(self):
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
    pf_extraction = content_features(first_weeks=first_weeks, verbose=args.verbose)
    pf_extraction.start_process()

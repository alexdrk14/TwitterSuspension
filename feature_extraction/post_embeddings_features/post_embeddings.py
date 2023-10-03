""""####################################################################################################################
Author: Alexander Shevtsov ICS-FORTH
E-mail: shevtsov@ics.forth.gr
-----------------------------------
Feature extraction based on user content-post (tweets, retweets and quotes ) in social media. 
Store collected features in csv file in data folder.
####################################################################################################################"""

import argparse
from dateutil.parser import parse

from sys import path
from datetime import datetime, timedelta
path.insert(1, '../../utils/')
from mongoConnector import MongoDB
from progBar import PBar
from labels_loading import load_labels
from utils import merge_tweet_and_retweet, text_feature

"""Load sentence transformer SBERT pre-trained model"""
from sentence_transformers import SentenceTransformer
#model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
model = SentenceTransformer('sentence-transformers/LaBSE')
DATA_PATH = "../../data/"

class post_features:
    """first_weeks show if we extract feature for first 3 weeks of our data or we extract feature after first 3 weeks"""

    def __init__(self, first_weeks=True, suspend_only=False, verbose=False):
        self.labels = None
        self.verbose = verbose
        """MongoDB connection class"""
        self.connector = MongoDB()
        self.first_weeks = first_weeks
        self.all_suspend = suspend_only

        if self.verbose:
            print("Extraction of user POST features for {} 21 days".format("first" if first_weeks else "second"))

        if self.all_suspend:
            """Extract features for all only suspended accounts"""
            self.start_date = datetime(2022, 2, 23, 0, 0, 0)
            self.end_date = datetime(2022, 5, 6, 0, 0, 0)
            self.output_filename = DATA_PATH + "post_embeddings_suspended.csv"  # "post_features_all_suspended.csv"
        elif first_weeks:
            """Extraction of first 21 days portion. Used for train/val/test"""
            self.output_filename = DATA_PATH + "post_embeddings_features.csv"
            self.start_date = datetime(2022, 2, 23, 0, 0, 0)
        else:
            """Extraction of second 21 days portion. Used for future model eval"""
            self.output_filename = DATA_PATH + "post_embeddings_features_second21.csv"
            self.start_date = datetime(2022, 2, 23, 0, 0, 0) + timedelta(days=21)

        if not self.all_suspend:
            """In case of not first and second 21 days only extraction we need to create end_date window 21 days 
               after the starting data extraction date
            """
            self.end_date = self.start_date + timedelta(days=21)


    def post_features(self, tweet):
        tweet_type = 0
        """get twitter post text from object (if full_text exist use it else use just text field)"""
        tweet_text = tweet["full_text"] if "full_text" in tweet else tweet["text"]
        entities = tweet["entities"]

        if "account is temporarily unavailable because it violates the Twitter Media Policy." in tweet_text:
            """ignore this posts, it is not available"""
            return None

        if "retweeted_status" in tweet.keys():
            tweet_type = 1
            tweet_text, entities = merge_tweet_and_retweet(tweet_text, (
                            tweet["retweeted_status"]["full_text"] if "full_text" in tweet["retweeted_status"] else
                                     tweet["retweeted_status"]["text"]),
                                                           tweet["entities"],
                                                           tweet["retweeted_status"]["entities"])
        elif "quoted_status" in tweet.keys():
            tweet_type = 2
   
        if "account has been withheld in" in tweet_text:
            return None
        clear_text, emojis = text_feature(tweet_text, entities)
        
        clear_text = "".join(clear_text)    
        
        if len(clear_text) != 0:
            self.user_texts.append(clear_text)
            self.user_text_type.append(tweet_type)
            self.tweet_ids.append(int(tweet["id"]))
            date = tweet["created_at"] if type(tweet["created_at"]) != str() else parse(tweet["created_at"])
            self.created_at.append((date.year*10000) + (date.month * 100) + date.day)



    def get_user_vector(self, user_id):
        """Initialize self values and datastructures for this particular user.
           After this iteration and user vector writing we delete them in order to reduce memory usage
           and do not leak data between users.
        """
        self.user_texts = []
        self.user_text_type = []
        self.user_id = user_id
        self.tweet_ids = []
        self.created_at = []
        self.target = self.labels["label"][self.labels["user_id"] == user_id].item()

        client, db = self.connector.connect()
        for tweet in db.Tweets.find({"user.id": user_id, "created_at": {
                                    "$gte": self.start_date, "$lt": self.end_date}}, no_cursor_timeout=True):
            self.post_features(tweet)
        client.close()


        if len(self.user_texts) != 0:
            """write data of this particular user"""
            self.dump_user_vector()

        """Clear class values of particular user"""
        del (self.user_texts)
        del(self.user_text_type)
        del(self.tweet_ids)
        del(self.created_at)

    def initialize_outfile(self):
        self.header = list()
        for i in range(1, 769):
            self.header.append("text_emb_{}".format(i))
        
        if not self.all_suspend:
            self.header += ["text_category", "tweet_id", "user_id", "created_at", "target"]
        else:
            self.header += ["text_category", "tweet_id", "user_id", "created_at"]
        self.f_out = open(self.output_filename, "w+")
        self.f_out.write("\t".join(self.header) + "\n")


    def dump_user_vector(self):
        encodings = model.encode(self.user_texts)
        for i in range(len(encodings)):
            if not self.all_suspend:
                text_enc = list(encodings[i]) + [self.user_text_type[i], self.tweet_ids[i], self.user_id, self.created_at[i], self.target]
            else:
                text_enc = list(encodings[i]) + [self.user_text_type[i], self.tweet_ids[i], self.user_id, self.created_at[i]]
            self.f_out.write("{}\n".format("\t".join([str(item) for item in text_enc])))

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
parser.add_argument('--suspend', dest="suspend", default=False, action='store_true')

args = parser.parse_args()
if __name__ == "__main__":
    first_weeks = True
    if args.second_weeks:
        first_weeks = False
    pf_extraction = post_features(first_weeks=first_weeks, suspend_only=args.suspend, verbose=args.verbose)
    pf_extraction.start_process()


import pickle
from math import log
from datetime import datetime, timedelta
from collections import defaultdict
#from utils import merge_entities

from sys import  path
path.insert(1, '../../utils/')
from mongoConnector import MongoDB
from progBar import PBar
from utils import merge_entities
import os

class TFIDF:

    def __init__(self, first_weeks=True, verbose=True):
        self.verbose = verbose
        """MongoDB connection class"""
        self.connector = MongoDB()
        self.first_weeks = first_weeks

        if self.verbose:
            print("Extraction of profile features for {} 21 days".format("first" if first_weeks else "second"))

        if first_weeks:
            """Extraction of first 21 days portion. Used for train/val/test"""
            self.mention_filename = "IDF_mention.pkl"
            self.hashtag_filename = "IDF_hashtag.pkl"
            self.start_date = datetime(2022, 2, 23, 0, 0, 0)
        else:
            """Extraction of second 21 days portion. Used for future model eval"""
            self.mention_filename = "IDF_mention_second21.pkl"
            self.hashtag_filename = "IDF_hashtag_second21.pkl"
            self.start_date = datetime(2022, 2, 23, 0, 0, 0) + timedelta(days=21)

        self.end_date = self.start_date + timedelta(days=21)

        self.dates_filter = {"created_at": {"$gte": self.start_date, "$lt": self.end_date}}
        self.idf_hash = None
        self.idf_ment = None
        self.load_idf()

    def load_idf(self):
        # check of IDF for hashtags/mentions is already exists
        if os.path.exists(self.hashtag_filename) and os.path.exists(self.mention_filename):
            print("IDF files are already exist!")
            in_f = open(self.hashtag_filename, "rb")
            self.idf_hash = pickle.load(in_f)
            in_f.close()
            in_f = open(self.mention_filename, "rb")
            self.idf_ment = pickle.load(in_f)
            in_f.close()
        else:
            print("TFIDF files not exist. Starting computation!")
            self.preprocess_hash_ment()

    def preprocess_hash_ment(self):
        idf_hash = defaultdict(lambda: 0)
        idf_ment = defaultdict(lambda: 0)
        hash_documents = 0
        ment_documents = 0

        client, db = self.connector.connect()
        all_cnt = db.Tweets.count(self.dates_filter, no_cursor_timeout=True)

        """Progress bar parameters"""
        if self.verbose:
            self.pbar = PBar(all_items=all_cnt, print_by=100000)

        for tweet in db.Tweets.find(self.dates_filter, no_cursor_timeout=True):
            # get twitter post text from object (if full_text exist use it else use just text field)
            tweet_text = tweet["full_text"] if "full_text" in tweet else tweet["text"]
            entities = tweet["entities"]

            if "account is temporarily unavailable because it violates the Twitter Media Policy." in tweet_text:
                """ignore this posts, it is not available"""
                continue

            if "retweeted_status" in tweet.keys():
                entities = merge_entities(tweet["entities"], tweet["retweeted_status"]["entities"])

            hs = set([entity["text"] for entity in entities["hashtags"]])
            if len(hs) > 0:
                hash_documents += 1
                for word in hs:
                    idf_hash[word] += 1
            mnt = set([entity["screen_name"] for entity in entities["user_mentions"]])
            if len(mnt) > 0:
                ment_documents += 1
                for word in mnt:
                    idf_ment[word] += 1 
            #documents_hash.append(set([entity["text"] for entity in entities["hashtags"]]))
            #documents_ment.append(set([entity["screen_name"] for entity in entities["user_mentions"]]))

            """Print the progress bar"""
            if self.verbose: self.pbar.increase_done()

        client.close()

        self.idf_hash = self.compute_idf(idf_hash, hash_documents)
        out_f = open(self.hashtag_filename, "wb+")
        pickle.dump(self.idf_hash, out_f)
        out_f.close()
        del(idf_hash)
        del(hash_documents)
        self.idf_ment = self.compute_idf(idf_ment, ment_documents)
        out_f = open(self.mention_filename, "wb+")
        pickle.dump(self.idf_ment, out_f)
        out_f.close()
        del(idf_ment)
        del(ment_documents)



    def compute_idf(self, idf_dict, number_of_doc, require_user_name=False):
        #number_of_doc = len(documents)
        #idf_dict = defaultdict(lambda: 0)
        #for document in documents:
        #    for word in document:
        #        idf_dict[word] += 1
        if require_user_name:
            idf_dict = {self.mongo.get_user_screen_name(uid): idf_dict[uid] for uid in idf_dict}

        for word, val in idf_dict.items():
            idf_dict[word] = log(float(number_of_doc) / float(val + 1))
        return dict(idf_dict)

    def compute_tf_idf(self, item, category, tf):
        if category == "hashtag" or "hash" in category:
            return self.idf_hash[item] * tf
        elif category == "mention" or "ment" in category:
            return self.idf_ment[item] * tf
        return None


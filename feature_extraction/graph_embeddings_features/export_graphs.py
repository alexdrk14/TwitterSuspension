""""####################################################################################################################
Author: Alexander Shevtsov ICS-FORTH
E-mail: shevtsov@ics.forth.gr
--------------------------------------
Parse twitter dataset collection and and store them as tsv file like (source_user_id relation destination_user_id)
(THIS SCRIPT WORK WITH NEW MONGO COLLECTION CREATED BY SCRIPT: TwitterSuspension/mongo_performance/social_relations.py)
Exported file is used by PytorchBigGraph. (Note that first21 days contain data from first 21days in dataset , but
second 21 days contain first 21 days data and second 21 days data, in total 42 days. Such export implementation is
created in this way, in order to achieve graph embedding output and node allocation stability)
####################################################################################################################"""


import sys, argparse
from collections import defaultdict
from datetime import datetime, timedelta

sys.path.insert(1, '../utils/')
from mongoConnector import MongoDB
from progBar import PBar
from os import path
import random

DATA_PATH = "../../data/"
class GraphExtractor:

    def __init__(self, first_weeks=True, verbose=False):

        self.verbose = verbose
        """MongoDB connection class"""
        self.connector = MongoDB()
        self.first_weeks = first_weeks

        if self.verbose:
            print("Graph extraction for {} 21 days".format("first" if first_weeks else "second"))
        self.output_filename = DATA_PATH + "/graph_RELATION_{}21.tsv".format("first" if first_weeks else "second")
        self.start_date = datetime(2022, 2, 23, 0, 0, 0)

        if first_weeks:
            """Extraction of first 21 days portion. Used for train/val/test"""
            self.end_date = self.start_date + timedelta(days=21)

        else:
            """Extraction of second 21 and first 21 days portion. Used for future model eval"""
            self.end_date = self.start_date + timedelta(days=42)


    def load(self, rel_category):
        if path.isfile(self.output_filename.replace("RELATION", rel_category)):
            """If file already exists, just return None"""
            print("File :{} already exists.".format(self.output_filename.replace("RELATION", rel_category)))
            return None
        if rel_category == "multy":
            graph_lines = list(self.all_edges)
        else:
            graph_rel = defaultdict(lambda: defaultdict(lambda: 0))
            client, db = self.connector.connect()
            if rel_category == "quote":
                mongo_collection = db.quotes
            elif rel_category == "mention":
                mongo_collection = db.mentions
            elif rel_category == "retweet":
                mongo_collection = db.retweets
            if self.verbose: print("\nStarting category:{}".format(rel_category))
            """Progress bar parameters"""
            self.pbar = PBar(all_items=int(mongo_collection.count({"created_at": {
                                                        "$gte": self.start_date,
                                                        "$lt": self.end_date}
                                                        })), print_by=100000)

            for item in mongo_collection.find({"created_at": {
                                                        "$gte": self.start_date,
                                                        "$lt": self.end_date}
                                                        }, no_cursor_timeout=True):
                graph_rel[item['src_user_id']][item['dst_user_id']] += 1
                self.pbar.increase_done()
            client.close()


            graph_lines = []
            for src_user in graph_rel:
                for dst_user in graph_rel[src_user]:
                    graph_lines.append("{}\t{}\t{}".format(src_user, rel_category, dst_user))

            del (dst_user)

        """Shuffle lines of graph"""
        random.shuffle(graph_lines)
        random.shuffle(graph_lines)
        random.shuffle(graph_lines)

        self.all_edges = self.all_edges.union(set(graph_lines))

        """Write data into file"""
        graph_file = open(self.output_filename.replace("RELATION", rel_category), "w+")
        graph_file.write("\n".join(graph_lines))
        graph_file.close()
        del(graph_lines)

    def main(self):
        self.all_edges = set()
        for category in ["quote", "mention", "retweet", "multy"]:
            self.load(category)
        del(self.all_edges)

parser = argparse.ArgumentParser()
parser.add_argument('--second_portion', dest='second_weeks', default=False, action='store_true')
parser.add_argument('-v', dest='verbose', default=False, action='store_true')

args = parser.parse_args()

if __name__ == "__main__":
    first_weeks = True
    if args.second_weeks:
        first_weeks = False
    GE = GraphExtractor(first_weeks=first_weeks, verbose=args.verbose)
    GE.main()

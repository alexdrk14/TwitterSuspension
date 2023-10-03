""""####################################################################################################################
Author: Alexander Shevtsov ICS-FORTH
E-mail: shevtsov@ics.forth.gr
--------------------------------------
Parse the tsv graph relation file and train the PytorchBigGraph model in order to store the graph embeddings
####################################################################################################################"""

import os, sys, argparse
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from pathlib import Path
from os import path
from parser import ParseEmbeddings


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from torchbiggraph.converters.importers import TSVEdgelistReader, convert_input_data
from torchbiggraph.config import parse_config
from torchbiggraph.train import train
from torchbiggraph.util import SubprocessInitializer, set_logging_verbosity, setup_logging


class SocialGraphEmbedding:
    def __init__(self, graph_type, out_dimmensions, first):
        self.dims = out_dimmensions
        self.DATA_DIR = '../../data'
        self.GRAPH_PATH = self.DATA_DIR + "/graph_{}_{}21.tsv".format(graph_type, "first" if first else "second")

        self.name = "{}_{}_{}21".format(graph_type, self.dims, "first" if first else "second")
        self.MODEL_DIR = 'model_{}'.format(self.name)
        self.parser = ParseEmbeddings(graph_type, out_dimmensions, first)
        if not path.isfile(self.GRAPH_PATH):
            print("Graph :{} not exists!")
            sys.exit(-1)
        self.main()

        self.parser.store_embeddings()

    """Generates configuration file in form of pytorchBigGraph"""
    def create_config(self):
        return dict(
            # I/O data
            entity_path="entity_{}_json/".format(self.name),
            edge_paths=["edges_{}_json/".format(self.name)],
            checkpoint_path="checkpoint_{}_json/".format(self.name),
            # Graph structure
            entities={"user": {"num_partitions": 1}},
            relations=[
                {
                    "name": "all_edges",
                    "lhs": "user",
                    "rhs": "user",
                    "operator": "complex_diagonal"
                }
            ],
            dynamic_relations=True,
            dimension=self.dims,
            global_emb=False,
            comparator="dot",
            num_epochs=50,
            num_edge_chunks=10,
            batch_size=10000,
            num_batch_negs=500,
            num_uniform_negs=500,
            loss_fn="softmax",
            lr=0.1,
            relation_lr=0.01,
            eval_fraction=0.0005,
            eval_num_batch_negs=10000,
            eval_num_uniform_negs=0,
            # GPU
            num_gpus=1,
        )

    def main(self):
        if path.exists(self.parser.entity_file) and path.exists(self.parser.embeddings_file):
            print("Embedding already exists")
            return None

        setup_logging()
        config = parse_config(self.create_config())
        subprocess_init = SubprocessInitializer()
        input_edge_paths = [Path(self.GRAPH_PATH)]

        convert_input_data(
            config.entities,
            config.relations,
            config.entity_path,
            config.edge_paths,
            input_edge_paths,
            TSVEdgelistReader(lhs_col=0, rel_col=1, rhs_col=2),
            dynamic_relations=config.dynamic_relations,
        )

        # train embeddings
        train(config, subprocess_init=subprocess_init)


parser = argparse.ArgumentParser(description='Parse extracted Graph files(.tsv) into PytorchBigGraph model and generate node embeddings')
parser.add_argument('--dims', type=int, dest='dims', required=True, help="Number of output embedding dimensions. (Required!)")
parser.add_argument('--second_portion', dest='second_weeks', default=False, action='store_true', help="Flag parameter describe if should be used for second 21 days data extraction")
parser.add_argument('--multy', dest='multy', default=False, action='store_true', help="Flag that describe graph relation that would be utilized (Multy)")
parser.add_argument('--quote', dest='quote', default=False, action='store_true', help="Flag that describe graph relation that would be utilized (Quote)")
parser.add_argument('--mention', dest='mention', default=False, action='store_true', help="Flag that describe graph relation that would be utilized (Mention)")
parser.add_argument('--retweet', dest='retweet', default=False, action='store_true', help="Flag that describe graph relation that would be utilized (Retweet)")


args = parser.parse_args()

if __name__ == "__main__":
    graph_type = ""
    if args.multy: graph_type = "multy"
    elif args.quote: graph_type = "quote"
    elif args.mention: graph_type = "mention"
    elif args.retweet: graph_type = "retweet"
    if graph_type == "":
        print("Select the graph relation (--multy or --retweet or --quote or --mention).")
        sys.exit(-1)

    PTBG = SocialGraphEmbedding(graph_type, args.dims, first=False if args.second_weeks else True)
    print("Done")
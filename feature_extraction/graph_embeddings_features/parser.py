import json
import h5py
import pandas as pd 

class ParseEmbeddings:
    def __init__(self, graph_type, out_dims, first):
        DATA_DIR = '../../data'

        self.graph_type = graph_type
        self.dims = out_dims
        self.first = first
        name = "{}_{}_{}21".format(graph_type, out_dims, "first" if first else "second")
        self.dataset_file = DATA_DIR + "/profile_features{}.csv".format("_second21" if not first else "")
        self.entity_file = "entity_{}_json/entity_names_user_0.json".format(name)
        self.embeddings_file = "model_{}/embeddings_user_0.v500.h5".format(name)
        self.output_file = DATA_DIR + "/graph_embeddings_features{}.csv".format("_second21" if not first else "")



    def store_embeddings(self):
        """Read profile feature file in order to get selected user_ids and do not store data for entire 6M dataset"""

        dataset = pd.read_csv(self.dataset_file, sep="\t")
        """Keep user_id and target in two separate list"""
        uid = dataset["user_id"].values.tolist()
        target = dataset["target"].values.tolist()

        """Read node names from neural embedding json file"""
        with open(self.entity_file, "rt") as tf:
            names = json.load(tf)
        """Cast to dictionary making much faster search process"""
        names = {names[i]: i for i in range(len(names))}

        """Create list of vectors based on user ids from profile features, containing graph embeddings"""
        emb = []
        with h5py.File(self.embeddings_file, "r") as hf:
            for i in range(len(uid)):
                try:
                    position = names[str(uid[i])]
                    emb.append(list(hf["embeddings"][position, :]) + [uid[i], target[i]])
                except:
                    """In case when user from profile features is not found in graph we just skip this user"""
                    continue
        """Create dataframe column name header and store it as csv file"""
        columns = ["graph_dim_{}".format(i) for i in range(1, 151)]
        columns.append("user_id")
        columns.append("target")
        df = pd.DataFrame(emb, columns=columns)
        df.to_csv(self.output_file, sep="\t", index=False)
        print("Store of embedding {} is done.".format(self.output_file))

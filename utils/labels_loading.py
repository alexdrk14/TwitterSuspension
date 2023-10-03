import pandas as pd
import ast

def load_labels(DATA_PATH, first_weeks):
    """read user labels from file"""
    labels = pd.read_csv(DATA_PATH + "user_labels.csv", header=0, sep=',')

    suspend = set()

    """Keep only suspended accounts that was suspended in first 21 days"""
    for line in open(DATA_PATH + "compliance_results_2022-03-15_23:03:24.117973.txt", "r").read().split("\n"):
        if line == '':
            continue
        line = ast.literal_eval(line)
        if line["reason"] == "suspended":
            suspend.add(int(line["id"]))

    """In case of second data portion we keep only users who was suspended during the second 21 days"""
    if not first_weeks:
        new_suspend = set()
        """Keep only suspended accounts that was suspended in first 21 days"""
        for line in open(DATA_PATH + "compliance_results_2022-04-05_17:43:42.515385.txt", "r").read().split("\n"):
            if line == '':
                continue
            line = ast.literal_eval(line)
            if line["reason"] == "suspended":
                new_suspend.add(int(line["id"]))
        suspend = new_suspend - suspend

    """Keep all normal users"""
    norm_ind = labels[labels["label"] == 0].index.tolist()
    """keep suspended that reveals in selected file"""
    df_susp = labels[labels["user_id"].isin(suspend)]
    susp_ind = df_susp[df_susp["label"] == 1].index.tolist()

    """store final df into self.labels"""
    labels = labels.iloc[norm_ind + susp_ind]
    user_ids = set(labels["user_id"].values)
    return labels, user_ids
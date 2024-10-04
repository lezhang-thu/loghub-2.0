import json
import os
import sys
import pandas as pd
import re
import string
from sklearn.utils import shuffle

from logppt.sampling import adaptive_random_sampling

datasets = [
    "Proxifier",
    "Linux",
    "Apache",
    "Zookeeper",
    "Hadoop",
    "HealthApp",
    "OpenStack",
    "HPC",
    "Mac",
    "OpenSSH",
    #"Spark",
    #"Thunderbird",
    #"BGL",
    #"HDFS",
]


def clean(s):
    """ Preprocess log message
    Parameters
    ----------
    s: str, raw log message
    Returns
    -------
    str, preprocessed log message without number tokens and special characters
    """
    s = re.sub(r':|\(|\)|=|,|"|\{|\}|@|$|\[|\]|\||;|\.', ' ', s)
    s = " ".join([
        word.lower() if word.isupper() else word for word in s.strip().split()
    ])
    s = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', s))
    s = " ".join(
        [word for word in s.split() if not bool(re.search(r'\d', word))])
    trantab = str.maketrans(dict.fromkeys(list(string.punctuation)))
    s = s.translate(trantab)
    s = " ".join([word.lower().strip() for word in s.strip().split()])
    return s


if __name__ == '__main__':
    shot_dir = "datasets-fs-full-wo-vl"
    brain_dir = sys.argv[1]
    for dataset in datasets:
        print(dataset)
        brain_df = pd.read_csv(
            os.path.join(brain_dir,
                         '{}_full.log_structured.csv'.format(dataset)))
        gt_df_path = os.path.join(
            '../../full_dataset/{}/{}_full.log_structured.csv'.format(
                dataset, dataset))
        gt_df = pd.read_csv(gt_df_path)
        # hacker - start
        gt_df["Content"] = brain_df["Content"]

        x_gt = os.path.basename(gt_df_path)
        if x_gt == 'HPC_full.log_structured.csv':
            print('Fix the ERROR in {}! Good!!!'.format(x_gt))
            gt_df["EventTemplate"] = gt_df["EventTemplate"].apply(
                lambda x: 'PSU status ( <*> <*> )'
                if x == 'PSU status (<*> <*>)' else x)
        # hacker - end

        content = [(clean(x), i, len(x))
                   for i, x in enumerate(gt_df['Content'].tolist())]
        content = [x for x in content if len(x[0].split()) > 1]
        content = shuffle(content)

        for shot in [32]:
            samples_ids = adaptive_random_sampling(content, shot)

            labeled_samples = [(row['Content'], row['EventTemplate'])
                               for _, row in gt_df.take(samples_ids).iterrows()]
            labeled_samples = [{
                "text": x[0],
                "label": x[1],
                "type": 1
            } for x in labeled_samples]
            os.makedirs("{}/{}/{}shot".format(shot_dir, dataset, shot),
                        exist_ok=True)
            with open("{}/{}/{}shot/{}.json".format(shot_dir, dataset, shot, 4),
                      "w") as f:
                for s in labeled_samples:
                    f.write(json.dumps(s) + "\n")

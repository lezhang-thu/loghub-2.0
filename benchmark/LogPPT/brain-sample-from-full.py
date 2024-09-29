import sys
import os
import json
import pandas as pd
import numpy as np
import copy
#from sentence_transformers import SentenceTransformer
from sklearn_extra.cluster import KMedoids
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils import shuffle

from fewshot_sampling import clean
from logppt.sampling import jaccard_distance
from logppt.sampling import adaptive_random_sampling

datasets = [
    #"Proxifier",
    #"Linux",
    "Apache",
    #"Zookeeper",
    #"Hadoop",
    #"HealthApp",
    #"OpenStack",
    #"HPC",
    #"Mac",
    #"OpenSSH",
    #"Spark",
    #"Thunderbird",
    #"BGL",
    #"HDFS",
]


def generate_train_data(df):
    grouped = df.groupby('EventTemplate').size().reset_index(name='counts')
    #print(grouped)
    df_with_probs = df.merge(grouped,
                             left_on='EventTemplate',
                             right_on='EventTemplate',
                             how='left')
    #print(df_with_probs)
    #exit(0)
    df_with_probs['probability'] = 1 / (len(grouped) * df_with_probs['counts'])
    return df_with_probs['probability'], len(grouped)


def sample_data(df, weights=None, num_groups=None):
    # init shot
    samples_ids = np.random.choice(
        len(df),
        size=max(32 * 10, num_groups * 10),
        #size=int(1e2),
        replace=True,
        p=weights)
    return df.loc[samples_ids][['Content', 'EventTemplate']]


def y_reproduce(df, model):
    content = [
        (clean(x), i, len(x)) for i, x in enumerate(df['Content'].tolist())
    ]
    content = [x for x in content if len(x[0].split()) > 1]
    content = shuffle(content)
    #only_clean = [_[0] for _ in content]

    samples_ids = adaptive_random_sampling(content, 32)
    labeled_samples = [(row['Content'], row['EventTemplate'])
                       for _, row in df.take(samples_ids).iterrows()]
    return labeled_samples

    #dist = pairwise_distances(
    #    only_clean,
    #    metric=lambda x, y: jaccard_distance(x, y),
    #)
    #x = KMedoids(
    #    n_clusters=32,
    #    metric='precomputed',
    #    init='k-medoids++',
    #    method='pam',
    #)

    #x.fit(dist)
    #x_list = []
    #for _ in x.medoid_indices_:
    #    x_list.append(content[_][1])

    #labeled_samples = [(row['Content'], row['EventTemplate'])
    #                   for _, row in df.take(x_list).iterrows()]
    #return labeled_samples


def x_reproduce(df, model):
    content = [
        (clean(x), i, len(x)) for i, x in enumerate(df['Content'].tolist())
    ]
    content = [x for x in content if len(x[0].split()) > 1]
    content = shuffle(content)

    vectors = np.asarray([_[0] for _ in content])
    center = max(range(0, len(content)),
                 key=lambda x: (len(content[x][0].split()), content[x][2]))

    ret = [None for _ in range(len(content))]
    not_chosen = np.full(len(content), True, dtype=bool)
    not_chosen_idx = np.arange(len(content), dtype=np.int32)

    not_chosen[center] = False
    ret[0] = content[center][0]

    for k in range(1, 32):
        dist = pairwise_distances(
            vectors[not_chosen],
            ret[:k],
            metric=lambda x, y: jaccard_distance(x, y),
        )
        dist = np.min(dist, axis=-1)

        max_vertex = np.argmax(dist)
        max_vertex = not_chosen_idx[not_chosen][max_vertex]

        ret[k] = vectors[max_vertex]
        not_chosen[max_vertex] = False
    chosen = ~not_chosen
    x_list = []
    for k, x in enumerate(chosen):
        if x:
            x_list.append(content[k][1])
    #print('len(x_list): {}'.format(len(x_list)))
    #exit(0)

    labeled_samples = [(row['Content'], row['EventTemplate'])
                       for _, row in df.take(x_list).iterrows()]
    return labeled_samples


def reproduce(df, model):
    content = [
        (clean(x), i, len(x)) for i, x in enumerate(df['Content'].tolist())
    ]
    content = [x for x in content if len(x[0].split()) > 1]
    content = shuffle(content)

    embeddings = model.encode([x[0] for x in content])
    vectors = np.asarray(embeddings)
    center = max(range(0, len(content)),
                 key=lambda x: (len(content[x][0].split()), content[x][2]))

    ret = np.zeros((32, vectors.shape[1]), dtype=vectors.dtype)
    not_chosen = np.full(len(vectors), True, dtype=bool)
    not_chosen_idx = np.arange(len(vectors), dtype=np.int32)

    not_chosen[center] = False
    ret[0] = vectors[center]

    for k in range(1, 32):
        dist = pairwise_distances(vectors[not_chosen], ret[:k], metric='cosine')
        dist = np.min(dist, axis=-1)

        max_vertex = np.argmax(dist)
        max_vertex = not_chosen_idx[not_chosen][max_vertex]

        ret[k] = vectors[max_vertex]
        not_chosen[max_vertex] = False
    chosen = ~not_chosen
    x_list = []
    for k, x in enumerate(chosen):
        if x:
            x_list.append(content[k][1])
    #print('len(x_list): {}'.format(len(x_list)))
    #exit(0)

    labeled_samples = [(row['Content'], row['EventTemplate'])
                       for _, row in df.take(x_list).iterrows()]
    return labeled_samples


def k_medoids(df, model):
    sentences = df['Content'].to_list()
    #sentences = [clean(_) for _ in sentences]
    # shot - 32
    embeddings = model.encode(sentences)
    if False:
        pass
        #x = pairwise_distances(embeddings, embeddings, metric='cosine')
        #x = x.sum(-1)
        #center = x.argmin()
    if False:
        kmedoids = KMedoids(
            n_clusters=32,
            metric='cosine',
            init='k-medoids++',
            method='pam',
        ).fit(embeddings)
        return df.iloc[kmedoids.medoid_indices_][['Content', 'EventTemplate']]
    if True:
        kmedoids = KMedoids(
            n_clusters=1,
            metric='cosine',
            init='k-medoids++',
            max_iter=0,
        ).fit(embeddings)
        center = kmedoids.medoid_indices_[0]

        vectors = np.asarray(embeddings)
        ret = np.zeros((32, vectors.shape[1]), dtype=vectors.dtype)
        not_chosen = np.full(len(vectors), True, dtype=bool)
        not_chosen_idx = np.arange(len(vectors), dtype=np.int32)

        not_chosen[center] = False
        ret[0] = vectors[center]
        for k in range(1, 32):
            dist = pairwise_distances(vectors[not_chosen],
                                      ret[:k],
                                      metric='cosine')
            dist = np.min(dist, axis=-1)

            max_vertex = np.argmax(dist)
            max_vertex = not_chosen_idx[not_chosen][max_vertex]

            ret[k] = vectors[max_vertex]
            not_chosen[max_vertex] = False

        return df[~not_chosen][['Content', 'EventTemplate']]


if __name__ == '__main__':
    #model = SentenceTransformer('all-mpnet-base-v2')
    model = None
    shot = 32
    shot_dir = "datasets-brain-help-sample-from-full"
    brain_dir = sys.argv[1]
    print('brain_dir: {}'.format(brain_dir))
    for dataset in datasets:
        print(dataset)
        os.makedirs("{}/{}".format(shot_dir, dataset), exist_ok=True)
        brain_df = pd.read_csv(
            os.path.join(brain_dir,
                         '{}_full.log_structured.csv'.format(dataset)))
        gt_df = pd.read_csv(
            os.path.join(
                '../../full_dataset/{}/{}_full.log_structured.csv'.format(
                    dataset, dataset)))
        # hacker - start
        gt_df["Content"] = brain_df["Content"]
        # hacker - end
        z, num_groups = generate_train_data(brain_df)
        x = sample_data(gt_df, weights=z.to_list(), num_groups=num_groups)
        x = x.reset_index(drop=True)
        json_ret = [
            {
                "text":  #z.iloc[0],
                    z[0],
                "label":  #z.iloc[1],
                    z[1],
                "type": 1
                #} for _, z in k_medoids(x, model).iterrows()]
            } for z in y_reproduce(x, model)
        ]

        os.makedirs("{}/{}/{}shot".format(shot_dir, dataset, shot),
                    exist_ok=True)
        with open("{}/{}/{}shot/{}.json".format(shot_dir, dataset, shot, 3),
                  "w") as f:
            for s in json_ret:
                f.write(json.dumps(s) + "\n")

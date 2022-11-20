import argparse

import numpy as np
from tqdm import tqdm

from spikenet import dataset
from spikenet.deepwalk import DeepWalk

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="?", default="DBLP",
                    help="Datasets (DBLP, Tmall, Patent). (default: DBLP)")
parser.add_argument('--normalize', action='store_true',
                    help='Whether to normalize output embedding. (default: False)')


args = parser.parse_args()
if args.dataset.lower() == "dblp":
    data = dataset.DBLP()
elif args.dataset.lower() == "tmall":
    data = dataset.Tmall()
elif args.dataset.lower() == "patent":
    data = dataset.Patent()
else:
    raise ValueError(
        f"{args.dataset} is invalid. Only datasets (dblp, tmall, patent) are available.")


model = DeepWalk(80, 10, 128, window_size=10, negative=1, workers=16)
xs = []
for g in tqdm(data.adj):
    model.fit(g)
    x = model.get_embedding(normalize=args.normalize)
    xs.append(x)


file_path = f'{data.root}/{data.name}/{data.name}.npy'
np.save(file_path, np.stack(xs, axis=0)) # [T, N, F]
print(f"Generated node feautures saved at {file_path}")

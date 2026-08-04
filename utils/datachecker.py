import numpy as np
from pathlib import Path

# load one file
datapath = Path("/home/bibarel/workspace/maskdp_data/maskdp_train")
taskpath = Path("walker/expert/walker_walk/train")
episode = Path ("episode_000000_1000.npz")

data = np.load(datapath / taskpath / episode)

for ep in range(10):
    episode = Path("episode_{}_1000.npz".format(str(ep).zfill(6)))
    data = np.load(datapath / taskpath / episode)
    print("Episode", ep)

    # check shapes of arrays
    for k, v in data.items():
        print("{}: {}".format(k, v.shape))
        # print("\t", v[0])
    print()

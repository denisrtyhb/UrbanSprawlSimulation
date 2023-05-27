# Import necessary packages

import lib
import os
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
import config

os.system(f"mkdir -p {config.results_folder}")

def read_data():
    all_files = os.listdir(config.data_folder)

    tif_files = []
    for name in all_files:
        if name[-4:] == ".tif":
            tif_files.append(name[:-4])

    before_name = config.before_name
    after_name = config.after_name

    if before_name not in tif_files:
        print(f"file {before_name}.tif not found")
        assert False

    if after_name not in tif_files:
        print(f"file {before_name}.tif not found")
        assert False

    def get_feature_names():
        res = []
        for name in tif_files:
            if name not in (before_name, after_name):
                res.append(name)
        return res

    feature_names = get_feature_names()

    all_data = dict()

    for name in feature_names + [before_name, after_name]:
        all_data[name] = rio.open(os.path.join(config.data_folder, name + ".tif")).read()[0]

    before = all_data[before_name]
    after = all_data[after_name]

    features = np.zeros((len(feature_names), *before.shape))
    for i in range(len(feature_names)):
        features[i] = all_data[feature_names[i]]
    return before, after, features

before, after, features = read_data()

# Data preprocessing
before[before < 0] = 3
after[after < 0] = 3
map_height, map_width = before.shape

# model fit
model_name = config.model_name
model = None
if model_name == "GWR":
    from models import GWR
    model = GWR(LAP=0.7, TIP=0.1, bandwidth=0.31, neigh_radius=5, threshold=0.3, seed=0x2143122)
    model.fit(before, after, features, samples_count = 4805)
else:
    from models import LogisticRegression
    model = LogisticRegression(LAP=0.5, TIP=0.1, neigh_radius=5, threshold=0.485)
    model.fit(before, after, features)

# Prediction
pr = model.predict(before)
print(lib.score(pr, before=before, after=after))
# lib.write_arr(f"{model_name}_simulation_after", pr)
lib.save_result(f"{model_name}_simulation_present", pr)

pr2 = model.predict(after, t=1)
lib.save_result(f"{model_name}_simulation_future", pr2)


im, ax= plt.subplots(1, 3, figsize=(20, 20))
ax[0].imshow(before, cmap=lib.cmap)
ax[0].axis('off')
ax[0].set_title("data before simulation")
ax[1].imshow(pr, cmap=lib.cmap)
ax[1].axis('off')
ax[1].set_title("Simulated present map")
ax[2].imshow(pr2, cmap=lib.cmap)
ax[2].axis('off')
ax[2].set_title("Simulated future map")
plt.show()
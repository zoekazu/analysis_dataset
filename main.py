# %%
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
from src.read_dir_images import ImgsInDirAsBool, ImgsInDirAsGray
import cv2
from IPython.display import display, Image

# %%


def display_cv(image, format='.bmp', bool_switch=False):
    if bool_switch:
        image = image.astype(np.uint8)*255
    decoded_bytes = cv2.imencode(format, image)[1].tobytes()
    display(Image(data=decoded_bytes))


# %%
fish_files = ImgsInDirAsGray('./images/pin/fish')
ref_files = ImgsInDirAsBool('./images/pin/ref',  bool_switch=True)
true_files = ImgsInDirAsBool('./images/pin/true', bool_switch=True)
false_files = ImgsInDirAsBool('./images/pin/false', bool_switch=True)
# %%


def display_label(nlabels, labels, img):
    img = np.zeros([true.shape[0], true.shape[1], 3])
    cols = []
    for i in range(1, nlabels):
        cols.append(np.array(
            [random.randint(0, 255),
             random.randint(0, 255),
             random.randint(0, 255)]))

    for i in range(1, nlabels):
        img[labels == i, ] = cols[i - 1]
    display_cv(img)

# %%


df_connected = pd.DataFrame(index=[], columns=['image_No', 'x_start',
                                               'y_start', 'width', 'height',
                                               'area', 'center_x', 'center_y',
                                               'pixle_mean', 'pixel_std',
                                               'pixel_var', 'pixel_min',
                                               'pixel_max', 'pixel_median',
                                               'true'])

for num, (fish, ref, true, false) in enumerate(zip(fish_files.read_files(), ref_files.read_files(),
                                                   true_files.read_files(), false_files.read_files()), start=1):
    true_or_false = np.logical_or(true, false)
    nlabels, labels, labels_status, center_object = cv2.connectedComponentsWithStats(
        true_or_false.astype(np.uint8)*255, connectivity=8)
    # display_cv(true_or_false.astype(np.uint8)*255)
    # display_label(nlabels, labels, true)
    # display_cv(true, bool_switch=True)

    labels_bool = np.zeros([labels.shape[0], labels.shape[1], nlabels], dtype=bool)
    nlabels_true = []
    pixel_mean = []
    pixel_std = []
    pixel_var = []
    pixel_min = []
    pixel_max = []
    pixel_median = []

    for i in range(nlabels):
        labels_bool[:, :, i] = np.where(labels == i, True, False)
        nlabels_true.append(np.any(np.logical_and(labels_bool[:, :, i], true)))
        pixel_mean.append(np.mean(fish[labels_bool[:, :, i] == True]))
        pixel_std.append(np.std(fish[labels_bool[:, :, i] == True]))
        pixel_var.append(np.var(fish[labels_bool[:, :, i] == True]))
        pixel_min.append(np.min(fish[labels_bool[:, :, i] == True]))
        pixel_max.append(np.max(fish[labels_bool[:, :, i] == True]))
        pixel_median.append(np.median(fish[labels_bool[:, :, i] == True]))

    for i in range(1, nlabels):
        status_series = pd.Series([num, labels_status[i, 0],
                                   labels_status[i, 1],
                                   labels_status[i, 2],
                                   labels_status[i, 3],
                                   labels_status[i, 4], ],
                                  dtype=np.int32,
                                  index=df_connected.columns[:6])
        center_series = pd.Series([center_object[i, 0],
                                   center_object[i, 1],
                                   pixel_mean[i],
                                   pixel_std[i],
                                   pixel_var[i],
                                   pixel_min[i],
                                   pixel_max[i],
                                   pixel_median[i]],
                                  dtype=np.float32,
                                  index=df_connected.columns[6:14])
        true_series = pd.Series(nlabels_true[i],
                                dtype=bool,
                                index=[df_connected.columns[14]])
        all_series = pd.concat([status_series, center_series, true_series])

        df_connected = df_connected.append(all_series, ignore_index=True)
# %%
df_connected['true'] = df_connected['true'].astype(bool)
df_connected.head()


# %%
x_train = df_connected.loc[:, ('width', 'height', 'area', 'pixle_mean', 'pixel_std',
                               'pixel_var', 'pixel_min',
                               'pixel_max', 'pixel_median',)].values
x_train

# %%
y_train = df_connected.loc[:, 'true'].values
y_train

# %%
rf = RandomForestClassifier(n_estimators=200, max_features='auto')
rf.fit(x_train, y_train)


# %%
ranking = np.argsort(-rf.feature_importances_)
# f, ax = plt.subplot(figsize=(11,9))
sns.barplot(x=rf.feature_importances_[ranking], y=df_connected.loc[:, ('width', 'height', 'area', 'pixle_mean', 'pixel_std',
                                                                       'pixel_var', 'pixel_min',
                                                                       'pixel_max', 'pixel_median',)].columns.values[ranking], orient='h')
# ax.set_xlabel('feature importance')
plt.tight_layout()
plt.show()


# %%

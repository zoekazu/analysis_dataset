# %%
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
                                               'y_start', 'width', 'height', 'area', 'center_x', 'center_y', 'true'])

for num, (fish, ref, true, false) in enumerate(zip(fish_files.read_files(), ref_files.read_files(),
                                                   true_files.read_files(), false_files.read_files()), start=1):
    true_or_false = np.logical_or(true, false)
    nlabels, labels, labels_status, center_object = cv2.connectedComponentsWithStats(
        true_or_false.astype(np.uint8)*255, connectivity=8)
    # display_cv(true_or_false.astype(np.uint8)*255)
    # display_label(nlabels. labels, true)
    # display_cv(true, bool_switch=True)
    nlabels_true = []

    labels_bool = np.zeros([labels.shape[0], labels.shape[1], nlabels], dtype=bool)
    for i in range(nlabels):
        labels_bool[:, :, i] = np.where(labels == i, True, False)
        nlabels_true.append(np.any(np.logical_and(labels_bool[:, :, i], true)))

    for i in range(1, nlabels):
        status_series = pd.Series([num, labels_status[i, 0],
                                   labels_status[i, 1],
                                   labels_status[i, 2],
                                   labels_status[i, 3],
                                   labels_status[i, 4]],
                                  dtype=np.int32,
                                  index=df_connected.columns[:6])
        center_series = pd.Series([center_object[i, 0],
                                   center_object[i, 1]],
                                  dtype=np.float32,
                                  index=df_connected.columns[6:8])
        true_series = pd.Series(nlabels_true[i],
                                dtype=bool,
                                index=[df_connected.columns[8]])
        all_series = pd.concat([status_series, center_series, true_series])

        df_connected = df_connected.append(all_series, ignore_index=True)
# %%
df_connected['true'] = df_connected['true'].astype(bool)
df_connected.head()
for (bool_type, color, maker_size) in [(False, 'b', 5), (True, 'r', 15)]:
    plt.scatter(df_connected.width[df_connected['true'] == bool_type],
                df_connected.height[df_connected['true'] == bool_type],
                c=color, label=bool_type, s=maker_size)
plt.xlabel('Width')
plt.ylabel('Height')
plt.legend()


# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('Width')
ax.set_ylabel('Height')
ax.set_zlabel('Area')


df_connected['true'] = df_connected['true'].astype(bool)
df_connected.head()
for (bool_type, color) in [(False, 'b'), (True, 'r')]:
    ax.plot(df_connected.width[df_connected['true'] == bool_type],
            df_connected.height[df_connected['true'] == bool_type],
            df_connected.area[df_connected['true'] == bool_type],
            "o", c=color, label=bool_type)
plt.show()


# %%
plt.figure()
df_area_sorted = df_connected.sort_values(by=['area'], ascending=True)
df_area_sorted = df_area_sorted.reset_index()
for (bool_type, color, maker_size) in [(False, 'b', 5), (True, 'r', 15)]:
    plt.scatter(df_area_sorted.index[df_area_sorted['true'] == bool_type],
                df_area_sorted.area[df_area_sorted['true'] == bool_type],
                c=color, label=bool_type, s=maker_size)
plt.xlabel('Number')
plt.ylabel('Area')
# %%
plt.figure()
df_width_sorted = df_connected.sort_values(by=['width'], ascending=True)
df_width_sorted = df_width_sorted.reset_index()
for (bool_type, color, maker_size) in [(False, 'b', 5), (True, 'r', 15)]:
    plt.scatter(df_width_sorted.index[df_width_sorted['true'] == bool_type],
                df_width_sorted.width[df_width_sorted['true'] == bool_type],
                c=color, label=bool_type, s=maker_size)
plt.xlabel('Number')
plt.ylabel('Width')


# %%
plt.figure()
df_height_sorted = df_connected.sort_values(by=['height'], ascending=True)
df_height_sorted = df_height_sorted.reset_index()
for (bool_type, color, maker_size) in [(False, 'b', 5), (True, 'r', 15)]:
    plt.scatter(df_height_sorted.index[df_height_sorted['true'] == bool_type],
                df_height_sorted.height[df_height_sorted['true'] == bool_type],
                c=color, label=bool_type, s=maker_size)
plt.xlabel('Number')
plt.ylabel('Height')

# %%
df_connected.to_pickle('./pandas_df_connected.pkl')

#%%

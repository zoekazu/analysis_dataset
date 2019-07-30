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
                                               'pixel_mean', 'pixel_std',
                                               'pixel_var', 'pixel_min',
                                               'pixel_max', 'pixel_median',
                                               'pixel_mean_ar3', 'pixel_std_ar3',
                                               'pixel_var_ar3', 'pixel_min_ar3',
                                               'pixel_max_ar3', 'pixel_median_ar3',
                                               'pixel_mean_ar5', 'pixel_std_ar5',
                                               'pixel_var_ar5', 'pixel_min_ar5',
                                               'pixel_max_ar5', 'pixel_median_ar5',
                                               'pixel_mean_ar3_diff', 'pixel_std_ar3_diff',
                                               'pixel_var_ar3_diff', 'pixel_min_ar3_diff',
                                               'pixel_max_ar3_diff', 'pixel_median_ar3_diff',
                                               'pixel_mean_ar5_diff', 'pixel_std_ar5_diff',
                                               'pixel_var_ar5_diff', 'pixel_min_ar5_diff',
                                               'pixel_max_ar5_diff', 'pixel_median_ar5_diff',
                                               'true'])
# %%
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

    pixel_mean_ar3 = []
    pixel_std_ar3 = []
    pixel_var_ar3 = []
    pixel_min_ar3 = []
    pixel_max_ar3 = []
    pixel_median_ar3 = []

    pixel_mean_ar5 = []
    pixel_std_ar5 = []
    pixel_var_ar5 = []
    pixel_min_ar5 = []
    pixel_max_ar5 = []
    pixel_median_ar5 = []

    kernel_ar3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_ar5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for i in range(nlabels):
        labels_bool[:, :, i] = np.where(labels == i, True, False)
        nlabels_true.append(np.any(np.logical_and(labels_bool[:, :, i], true)))
        pixel_mean.append(np.mean(fish[labels_bool[:, :, i] == True]))
        pixel_std.append(np.std(fish[labels_bool[:, :, i] == True]))
        pixel_var.append(np.var(fish[labels_bool[:, :, i] == True]))
        pixel_min.append(np.min(fish[labels_bool[:, :, i] == True]))
        pixel_max.append(np.max(fish[labels_bool[:, :, i] == True]))
        pixel_median.append(np.median(fish[labels_bool[:, :, i] == True]))

        labels_bool_3 = cv2.dilate(labels_bool[:, :, i].astype(
            np.uint8)*255, kernel_ar3, iterations=1)
        labels_bool_3 = np.where(labels_bool_3 > 0, True, False)
        labels_bool_ar3 = np.logical_and(labels_bool_3, np.logical_not(labels_bool[:, :, i]))

        pixel_mean_ar3.append(np.mean(fish[labels_bool_ar3 == True]))
        pixel_std_ar3.append(np.std(fish[labels_bool_ar3 == True]))
        pixel_var_ar3.append(np.var(fish[labels_bool_ar3 == True]))
        pixel_min_ar3.append(np.min(fish[labels_bool_ar3 == True]))
        pixel_max_ar3.append(np.max(fish[labels_bool_ar3 == True]))
        pixel_median_ar3.append(np.median(fish[labels_bool_ar3 == True]))

        labels_bool_5 = cv2.dilate(labels_bool[:, :, i].astype(
            np.uint8)*255, kernel_ar5, iterations=1)
        labels_bool_5 = np.where(labels_bool_5 > 0, True, False)
        labels_bool_ar5 = np.logical_and(labels_bool_5, np.logical_not(labels_bool[:, :, i]))

        pixel_mean_ar5.append(np.mean(fish[labels_bool_ar5 == True]))
        pixel_std_ar5.append(np.std(fish[labels_bool_ar5 == True]))
        pixel_var_ar5.append(np.var(fish[labels_bool_ar5 == True]))
        pixel_min_ar5.append(np.min(fish[labels_bool_ar5 == True]))
        pixel_max_ar5.append(np.max(fish[labels_bool_ar5 == True]))
        pixel_median_ar5.append(np.median(fish[labels_bool_ar5 == True]))

    pixel_mean_ar3_diff = [i - j for (i,j)in zip(pixel_mean, pixel_mean_ar3)]
    pixel_std_ar3_diff = [i - j for (i,j)in zip(pixel_std, pixel_std_ar3)]
    pixel_var_ar3_diff = [i - j for (i,j)in zip(pixel_var, pixel_var_ar3)]
    pixel_min_ar3_diff = [i - j for (i,j)in zip(pixel_min, pixel_min_ar3)]
    pixel_max_ar3_diff = [i - j for (i,j)in zip(pixel_max, pixel_max_ar3)]
    pixel_median_ar3_diff = [i - j for (i,j)in zip(pixel_median, pixel_median_ar3)]

    pixel_mean_ar5_diff = [i - j for (i,j)in zip(pixel_mean, pixel_mean_ar5)]
    pixel_std_ar5_diff = [i - j for (i,j)in zip(pixel_std, pixel_std_ar5)]
    pixel_var_ar5_diff = [i - j for (i,j)in zip(pixel_var, pixel_var_ar5)]
    pixel_min_ar5_diff = [i - j for (i,j)in zip(pixel_min, pixel_min_ar5)]
    pixel_max_ar5_diff = [i - j for (i,j)in zip(pixel_max, pixel_max_ar5)]
    pixel_median_ar5_diff = [i - j for (i,j)in zip(pixel_median, pixel_median_ar5)]

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
                                   pixel_median[i],
                                   pixel_mean_ar3[i],
                                   pixel_std_ar3[i],
                                   pixel_var_ar3[i],
                                   pixel_min_ar3[i],
                                   pixel_max_ar3[i],
                                   pixel_median_ar3[i],
                                   pixel_mean_ar5[i],
                                   pixel_std_ar5[i],
                                   pixel_var_ar5[i],
                                   pixel_min_ar5[i],
                                   pixel_max_ar5[i],
                                   pixel_median_ar5[i],
                                   pixel_mean_ar3[i],
                                   pixel_std_ar3_diff[i],
                                   pixel_var_ar3_diff[i],
                                   pixel_min_ar3_diff[i],
                                   pixel_max_ar3_diff[i],
                                   pixel_median_ar3_diff[i],
                                   pixel_mean_ar5_diff[i],
                                   pixel_std_ar5_diff[i],
                                   pixel_var_ar5_diff[i],
                                   pixel_min_ar5_diff[i],
                                   pixel_max_ar5_diff[i],
                                   pixel_median_ar5[i]],
                                  dtype=np.float32,
                                  index=df_connected.columns[6:38])
        true_series = pd.Series(nlabels_true[i],
                                dtype=bool,
                                index=[df_connected.columns[38]])
        all_series = pd.concat([status_series, center_series, true_series])

        df_connected = df_connected.append(all_series, ignore_index=True)
# %%
df_connected['true'] = df_connected['true'].astype(bool)
df_connected.head()
#%%
df_connected.to_pickle('./pandas_df_connected_diff_pin.pkl')



# %%
df_connected[df_connected['true']==True]

#%%
df_connected_ignore = df_connected.copy()
df_connected_ignore.head()
#%%
drop_list = list(df_connected_ignore.index[[181,182,184,187,195,196,197,207,209,212,213,214]])
drop_list
#%%
df_connected_ignore=df_connected_ignore.drop(index=drop_list)
df_connected_ignore[df_connected_ignore['true']==True]
#%%
df_connected_ignore = df_connected_ignore.reset_index()
df_connected_ignore

#%%
df_connected_ignore.to_pickle('./pandas_df_connected_diff_ignore_pin.pkl')


#%%

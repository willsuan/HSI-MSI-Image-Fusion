from motion_code.motion_code import MotionCode
from motion_code.data_processing import load_data, process_data_for_motion_codes
import cv2
from datasets.jasper_ridge import input_processing
import numpy as np
import spectral as sp

def load_envi_data(pix_path, hdr_path):
    img = sp.envi.open(hdr_path, pix_path)
    cube = np.asarray(img.load()) 
    return cube

start_band = 380; end_band = 2500
rgb_width = 64; rgb_height = 64
hsi_width = 32; hsi_height = 32

# img_sri, gt = input_processing(img_path, gt_path)
# img_hsi = img_sri #cv2.pyrDown(img_sri, dstsize=(50, 50))

hsi_path = "./data/GRSS/ImageryAndTrainingGT/2018IEEE_Contest/Phase2/Aligned/hsi_aligned"
hsi_hdr = "./data/GRSS/ImageryAndTrainingGT/2018IEEE_Contest/Phase2/Aligned/hsi_aligned.hdr"

gt_path = "./data/GRSS/ImageryAndTrainingGT/2018IEEE_Contest/Phase2/Aligned/gt_aligned"
gt_hdr = "./data/GRSS/ImageryAndTrainingGT/2018IEEE_Contest/Phase2/Aligned/gt_aligned.hdr"

img_hsi = load_envi_data(hsi_path, hsi_hdr)
gt = load_envi_data(gt_path, gt_hdr)

print("hsi shape: ", img_hsi.shape)
img_hsi_reshaped = img_hsi.reshape(-1, img_hsi.shape[-1])
print("hsi reshaped: ", img_hsi_reshaped.shape)

print("gt shape: ", gt.shape)
gt_reshaped = gt.reshape(-1, gt.shape[-1])
print("gt reshaped: ", gt_reshaped.shape)

size_each_class = 50
color = 10
colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'brown', 'pink', 'gray', 'olive', 'teal', 'navy', 'maroon', 'lime', 'coral', 'gold', 'silver', 'slategray']
label_names = ['road', 'tree', 'water', 'dirt', 'shade', 'building', 'vehicle', 'person', 'animal', 'other', 'sky', 'grass', 'rock', 'sand', 'snow', 'cloud', 'shadow', 'fence', 'sign', 'xyz', 'unknown']

num_classes = 21
indices = None


all_labels = gt_reshaped[:, 0]
for c in range(num_classes):
    print('Class', c)
    indices_in_class = np.where(all_labels == c)[0]
    print("indices in class: ", len(indices_in_class))
    current_choices = np.random.choice(indices_in_class, size=size_each_class)
    if indices is None:
        indices = current_choices
    else:
        indices = np.append(indices, current_choices)
num_series = indices.shape[0]

all_num_series = img_hsi_reshaped.shape[0]
Y_train_all = img_hsi_reshaped.reshape(all_num_series, 1, -1)
Y_train = img_hsi_reshaped[indices, :].reshape(num_series, 1, -1)
labels_train_all = np.argmax(gt_reshaped, axis=1)
labels_train = np.argmax(gt_reshaped[indices, :], axis=1)
print(Y_train.shape, labels_train.shape)

# Then we process the data for motion code model and generate X-variable, which is needed for training.
X_train, Y_train, labels_train = process_data_for_motion_codes(Y_train, labels_train)
X_train_all, Y_train_all, labels_train_all = process_data_for_motion_codes(Y_train_all, labels_train_all)
print(X_train.shape, Y_train.shape, labels_train.shape)
print(X_train_all.shape, Y_train_all.shape, labels_train_all.shape)

model = MotionCode(m=12, Q=1, latent_dim=2, sigma_y=0.1)

print("X_train shape: ", X_train.shape)

model_path = 'motion_code/saved_models/grss'

# Then we train model on the given X_train, Y_train, label_train set and saved it to a file named test_model.
model.fit(X_train, Y_train, labels_train, model_path)





model = MotionCode(m=12, Q=1, latent_dim=2, sigma_y=0.1)

model.load(model_path)
from motion_code.utils import plot_motion_codes

plot_motion_codes(X_train, Y_train, test_time_horizon=None, labels=labels_train, label_names=label_names, \
                           model=model, output_dir='motion_code/out/multiple/', additional_data=None)




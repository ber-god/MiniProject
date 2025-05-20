import os
import glob
import torch
from monai.data import list_data_collate
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Spacingd, ScaleIntensityRanged,
    CropForegroundd, RandFlipd, RandRotate90d, ToTensord, Compose
)
from monai.transforms import (
    ResizeWithPadOrCropd, RandCropByPosNegLabeld,
    EnsureTyped, ConvertToMultiChannelBasedOnBratsClassesd, Orientationd, RandSpatialCropd, NormalizeIntensityd, RandScaleIntensityd, RandShiftIntensityd, Activations, AsDiscrete
)
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.transforms import MapTransform
from monai.networks.nets import SegResNet
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
from monai.utils import set_determinism
logging.basicConfig(level=logging.INFO)
torch.autograd.set_detect_anomaly(True)

# Set the random seed for reproducibility
set_determinism(seed=0)

# Helper function to collect files from nested folders
def get_data_list(root_dir):
    subjects = sorted(os.listdir(root_dir))
    data_list = []
    for subj in subjects:
        preRT_dir = os.path.join(root_dir, subj, "preRT")
        if os.path.isdir(preRT_dir):
            image = glob.glob(os.path.join(preRT_dir, "*_preRT_T2.nii.gz"))
            label = glob.glob(os.path.join(preRT_dir, "*_preRT_mask.nii.gz"))
            if image and label:
                data_list.append({"image": image[0], "label": label[0]})
    return data_list

# Load file paths
train_files = get_data_list("C:/Users/ber/Downloads/Dataset/train")
val_files = get_data_list("C:/Users/ber/Downloads/Dataset/test")

# Update the label conversion to use the provided class
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            #result.append(torch.logical_or(d[key] ==0, d[key] == 1))
            result.append((d[key] == 0))
            result.append((d[key] == 1))
            d[key] = torch.stack(result, axis=0).float()
        return d

train_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        RandSpatialCropd(keys=["image", "label"], roi_size=(64, 64, 32), random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ]
)


val_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
)

# Create datasets and loaders
train_ds = CacheDataset(data=train_files, transform=train_transform, cache_rate=0)
train_loader = DataLoader(
    train_ds,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    collate_fn=list_data_collate,
)


val_ds = CacheDataset(data=val_files, transform=val_transform, cache_rate=0)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=0, pin_memory=False, collate_fn=list_data_collate)
'''
# pick one image from DecathlonDataset to visualize and check the 4 channels
val_data_example = val_ds[0]
print(f"image shape: {val_data_example['image'].shape}")
plt.figure("image", (6, 6))
plt.title("image")
plt.imshow(val_data_example["image"][0, :, :, 60].detach().cpu(), cmap="gray")
plt.savefig("example_image.png")
plt.close()

# Visualize the 3 channels label corresponding to this image
print(f"label shape: {val_data_example['label'].shape}")
plt.figure("label", (18, 6))
for i in range(3):  # Adjust to match the number of output channels in your model
    plt.subplot(1, 3, i + 1)
    plt.title(f"label channel {i}")
    plt.imshow(val_data_example["label"][i, :, :, 60].detach().cpu())
plt.savefig("example_label.png")
plt.close()
'''
    
# Setup model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SegResNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,  # ← set to 2 or 3 depending on your number of label classes
    init_filters=32,
    dropout_prob=0.2,
).to(device)

def inference(input):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(240, 240, 160),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if True:
        with torch.autocast("cuda"):
            return _compute(input)
    else:
        return _compute(input)
    
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

model.load_state_dict(torch.load(os.path.join("C:/Users/ber/Downloads/MiniProject", "best_metric_model.pth"), weights_only=True))
model.eval()
with torch.no_grad():
    # select one image to evaluate and visualize the model output
    val_input = val_ds[0]["image"].unsqueeze(0).to(device)
    roi_size = (128, 128, 64)
    sw_batch_size = 4
    val_output = inference(val_input)
    val_output = post_trans(val_output[0])
    
    # Save the input image
    plt.figure("image", (24, 6))
    plt.title(f"image channel {1}")
    plt.imshow(val_ds[0]["image"][0, :, :, 60].detach().cpu(), cmap="gray")
    plt.savefig("val_image_channel_1.png")
    plt.close()
    
    # Save the label channels
    plt.figure("label", (18, 6))
    plt.subplot(1, 2, 1)
    plt.title(f"output channel {0}")
    plt.imshow(val_ds[0]["label"][0, :, :, 60].detach().cpu())
    plt.subplot(1, 2, 2)
    plt.title(f"output channel {1}")
    plt.imshow(val_ds[0]["label"][1, :, :, 60].detach().cpu())
    plt.savefig("val_label_channels.png")
    plt.close()
    
    # Save the output channels
    plt.figure("output", (18, 6))
    plt.subplot(1, 2, 1)
    plt.title(f"output channel {0}")
    plt.imshow(val_output[0, :, :, 60].detach().cpu())
    plt.subplot(1, 2, 2)
    plt.title(f"output channel {1}")
    plt.imshow(val_output[1, :, :, 60].detach().cpu())
    plt.savefig("val_output_channels.png")
    plt.close()
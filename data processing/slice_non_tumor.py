import os
import numpy as np
import nibabel as nib
from PIL import Image

image_dir = r".\Tumor_FLAIR"  # Update this path to your FLAIR/T1/T2 images directory and run the code to generate the non tumor slices.
mask_dir = r".\Segmentation_masks" # Update this path to your segmentation masks directory.
output_dir = r".\FLAIR\No_tumor" # Update this path to your desired output directory for non tumor slices. The code will save the non tumor slices in this directory.

os.makedirs(output_dir, exist_ok=True)

def normalize_to_uint8(slice_2d):
    slice_2d = slice_2d.astype(np.float32)
    slice_2d -= slice_2d.min()
    if slice_2d.max() > 0:
        slice_2d /= slice_2d.max()
    return (slice_2d * 255).astype(np.uint8)

mask_lookup = {
    fname[:20]: fname
    for fname in os.listdir(mask_dir)
}

for img_name in os.listdir(image_dir):
    prefix = img_name[:20]

    if prefix not in mask_lookup:
        continue

    img_path = os.path.join(image_dir, img_name)
    mask_path = os.path.join(mask_dir, mask_lookup[prefix])

    image_nii = nib.load(img_path)
    mask_nii = nib.load(mask_path)

    image = image_nii.get_fdata()
    mask = mask_nii.get_fdata()
    mask = (mask > 0).astype(np.uint8)
    

    
    mask_slices = np.where(mask.sum(axis=(0, 1)) > 0)[0]
    
    top_slice = mask_slices.min()
    bottom_slice = mask_slices.max()

    if len(mask_slices) == 0:
        continue

    slices_to_save = set()
    slices_to_save.update([top_slice - 10, top_slice - 6, top_slice - 2, bottom_slice + 2, bottom_slice + 6, bottom_slice + 10])
    
    slices_to_save = [
        z for z in slices_to_save
        if 0 <= z < image.shape[2]
    ]

    for z in sorted(slices_to_save):
        slice_2d = image[:, :, z]
        if z > 130 or z < 20:
            continue
        slice_2d = normalize_to_uint8(slice_2d)

        png = Image.fromarray(slice_2d)
        png = png.resize((256, 256), resample=Image.BILINEAR)

        out_name = f"{prefix}_slice_{z:03d}.png"
        png.save(os.path.join(output_dir, out_name))
        
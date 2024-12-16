import os
import datetime
import json
from PIL import Image
import numpy as np
from glob import glob
from PIL import Image, ImageDraw
from scipy import ndimage, misc
import scipy.misc
import imageio

# http://scikit-image.org/docs/dev/auto_examples/filters/plot_denoise.html
from skimage.util import random_noise

'''
IMAGE PROCESSING
- read_dataset_image_path
- read_dataset_images
- read_lst_images
- read_image
- get_noisy_data
'''
import os
import numpy as np
from glob import glob
from skimage.util import random_noise
from skimage import io, transform



# IMAGE PROCESSING FUNCTIONS

def get_noisy_data(data, sigma=0.155):
    """Add Gaussian noise to a dataset of images."""
    return np.array([random_noise(image, var=sigma**2) for image in data])

def read_dataset_image_path(dataset_url, number_count=None):
    """Fetch all image paths from the dataset directory."""
    image_paths = []
    for dir_path in glob(os.path.join(dataset_url, '*')):
        for image_path in glob(os.path.join(dir_path, '*')):
            image_paths.append(image_path)
            if number_count is not None and len(image_paths) >= number_count:
                return np.array(image_paths)
    return np.array(image_paths)

def read_image_with_noise(image_path, sigma=0.155):
    """Read an image, add Gaussian noise, and return it."""
    image = read_image(image_path)
    noisy_image = random_noise(image, var=sigma**2)
    return np.array(noisy_image)

def read_image(image_path):
    """Read and normalize an image."""
    image = io.imread(image_path)
    # Crop and normalize
    cropped_image = image[100:240, 0:360]
    normalized_image = cropped_image / 127.5 - 1.0
    return np.array(normalized_image)

def read_images_with_noise(image_paths, patch_size, patch_step):
    """Read images, add noise, and return patches."""
    slices = []
    locations = []
    for image_path in image_paths:
        noisy_image = read_image_with_noise(image_path)
        patches, locations_slice = get_image_patches([noisy_image], patch_size, patch_step)
        slices.extend(patches)
        locations.extend(locations_slice)
    return np.array(slices), locations

def read_images(image_paths, patch_size=None, patch_step=None, work_on_patches=True):
    """Read images, optionally return patches."""
    if work_on_patches:
        slices = []
        locations = []
        for image_path in image_paths:
            image = read_image(image_path)
            patches, locations_slice = get_image_patches([image], patch_size, patch_step)
            slices.extend(patches)
            locations.extend(locations_slice)
        return slices, locations
    else:
        images = [read_image(image_path) for image_path in image_paths]
        return np.array(images)

def get_image_patches(images, patch_size, stride):
    """Extract image patches with specified size and stride."""
    patches = []
    locations = []
    for img in images:
        for i in range(0, img.shape[0] - patch_size[0] + 1, stride[0]):
            for j in range(0, img.shape[1] - patch_size[1] + 1, stride[1]):
                patch = img[i:i + patch_size[0], j:j + patch_size[1]]
                patches.append(patch)
                locations.append((i, j))
    return np.array(patches), locations

def get_video_patches(image_paths, patch_size, stride, depth):
    """Extract patches from video slices."""
    video_slices = []
    video_locations = []
    num_videos = len(image_paths) // depth

    for i in range(num_videos):
        video_images = read_images(image_paths[i * depth:(i + 1) * depth])
        patches, locations = get_image_patches(video_images, patch_size, stride)
        video_slices.extend(patches)
        video_locations.extend(locations)

    print(f'Video patches ready: {len(video_slices)} patches')
    return np.array(video_slices), video_locations
import os
import numpy as np
from glob import glob
from skimage.util import random_noise
from skimage import io, transform

# IMAGE PROCESSING FUNCTIONS

def get_noisy_data(data, sigma=0.155):
    """Add Gaussian noise to a dataset of images."""
    return np.array([random_noise(image, var=sigma**2) for image in data])

def read_dataset_image_path(dataset_url, number_count=None):
    """Fetch all image paths from the dataset directory."""
    image_paths = []
    for dir_path in glob(os.path.join(dataset_url, '*')):
        for image_path in glob(os.path.join(dir_path, '*')):
            image_paths.append(image_path)
            if number_count is not None and len(image_paths) >= number_count:
                return np.array(image_paths)
    return np.array(image_paths)

def read_image_with_noise(image_path, sigma=0.155):
    """Read an image, add Gaussian noise, and return it."""
    image = read_image(image_path)
    noisy_image = random_noise(image, var=sigma**2)
    return np.array(noisy_image)

def read_image(image_path):
    """Read and normalize an image."""
    image = io.imread(image_path)
    # Crop and normalize
    cropped_image = image[100:240, 0:360]
    normalized_image = cropped_image / 127.5 - 1.0
    return np.array(normalized_image)

def read_images_with_noise(image_paths, patch_size, patch_step):
    """Read images, add noise, and return patches."""
    slices = []
    locations = []
    for image_path in image_paths:
        noisy_image = read_image_with_noise(image_path)
        patches, locations_slice = get_image_patches([noisy_image], patch_size, patch_step)
        slices.extend(patches)
        locations.extend(locations_slice)
    return np.array(slices), locations

def read_images(image_paths, patch_size=None, patch_step=None, work_on_patches=True):
    """Read images, optionally return patches."""
    if work_on_patches:
        slices = []
        locations = []
        for image_path in image_paths:
            image = read_image(image_path)
            patches, locations_slice = get_image_patches([image], patch_size, patch_step)
            slices.extend(patches)
            locations.extend(locations_slice)
        return slices, locations
    else:
        images = [read_image(image_path) for image_path in image_paths]
        return np.array(images)

def get_image_patches(image_src, nd_patch_size, nd_stride):
    """Extract patches and their locations from an image with stride and size."""
    image_src = np.array(image_src)

    lst_patches = []
    lst_locations = []

    n_stride_h, n_stride_w = nd_stride
    tmp_frame = image_src[0].shape
    n_frame_h, n_frame_w = tmp_frame

    i = 0
    while i < n_frame_h:
        start_h = i
        end_h = i + nd_patch_size[0]
        if end_h > n_frame_h:
            start_h = n_frame_h - nd_patch_size[0]
            end_h = n_frame_h
        
        j = 0
        while j < n_frame_w:
            start_w = j
            end_w = j + nd_patch_size[1]
            if end_w > n_frame_w:
                start_w = n_frame_w - nd_patch_size[1]
                end_w = n_frame_w

            tmp_slices = np.array(image_src[:, start_h:end_h, start_w:end_w])
            lst_patches.append(tmp_slices)
            lst_locations.append([start_h, start_w])

            j += n_stride_w
        i += n_stride_h

    return np.array(lst_patches), lst_locations

def get_video_patches(image_paths, patch_size, stride, depth):
    """Extract patches from video slices."""
    video_slices = []
    video_locations = []
    num_videos = len(image_paths) // depth

    for i in range(num_videos):
        video_images = read_images(image_paths[i * depth:(i + 1) * depth])
        patches, locations = get_image_patches(video_images, patch_size, stride)
        video_slices.extend(patches)
        video_locations.extend(locations)

    print(f'Video patches ready: {len(video_slices)} patches')
    return np.array(video_slices), video_locations

def kh_is_dir_exist(path):
    """Check if a directory exists, and create it if not."""
    if not os.path.exists(path):
        os.makedirs(path)
        print('Directory created:', path)
    return

def kh_crop(img, n_start_x, n_end_x, n_start_y, n_end_y):
    """Crop an image to the specified coordinates."""
    return img[n_start_y:n_end_y, n_start_x:n_end_x]

def kh_extract_patches(s_img, n_stride=1, nd_slice_size=(10, 10), save_images=False):
    """Extract patches from a list of images."""
    i = 0
    j = 0
    img_array = np.zeros([io.imread(s_img[0]).shape[0], io.imread(s_img[0]).shape[1], 3])

    while i < len(s_img):
        img_tmp1 = io.imread(s_img[i])
        img_tmp2 = io.imread(s_img[i + 1])

        img_array1 = (img_tmp1 - np.mean(img_tmp1)) / np.std(img_tmp1)
        img_array2 = (img_tmp2 - np.mean(img_tmp2)) / np.std(img_tmp2)

        img_array[:, :, j] = (img_array1 + img_array2) / 2
        i += 2
        j += 1

    n_img_array_h, n_img_array_w = img_array.shape[:2]
    best_rg = img_array[100:n_img_array_h - 14, 0:n_img_array_w]
    slice_size_width, slice_size_height = nd_slice_size

    lst_fnames_tmp = []
    lst_patches = []
    for y in range(0, n_img_array_h - slice_size_height + 1, n_stride):
        for x in range(0, n_img_array_w - slice_size_width + 1, n_stride):
            mx = min(x + slice_size_width, n_img_array_w)
            my = min(y + slice_size_height, n_img_array_h)

            crp = kh_crop(img_array, x, mx, y, my)
            tile = transform.resize(crp, (slice_size_width, slice_size_height, 3))

            if save_images:
                save_dir = './'
                kh_is_dir_exist(save_dir)
                save_path = os.path.join(save_dir, f'patch_{x}_{y}.jpg')
                io.imsave(save_path, tile)

            lst_patches.append(tile)
            lst_fnames_tmp.append((x, y))

    print(f'{s_img} => Patches extracted')
    return lst_patches, lst_fnames_tmp
def extract_patches_one(image_path, stride=1, slice_size=(10, 10), save_images=False):
    # Read image using TensorFlow
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)

    img_height, img_width, _ = img.shape

    # Extract the region of interest
    best_rg = img[100:img_height-14, :]
    best_rg_height, best_rg_width, _ = best_rg.shape

    patch_height, patch_width = slice_size

    patches = []
    file_names = []

    for y in range(0, best_rg_height - patch_height + 1, stride):
        for x in range(0, best_rg_width - patch_width + 1, stride):
            patch = best_rg[y:y+patch_height, x:x+patch_width]

            # Optional: Save patch images if required
            if save_images:
                base_path = os.path.dirname(image_path)
                base_name = os.path.basename(image_path).split('.')[0]
                save_dir = os.path.join(base_path, "patches")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"patch_{y}_{x}.png")
                tf.keras.utils.save_img(save_path, patch)

            # Append the patch to the list
            patches.append(patch.numpy())
            file_names.append(f"patch_{y}_{x}")

    return patches, file_names

def get_sliced_images(image_paths, slice_size=(10, 10), stride=1, save_images=False):
    all_patches = []
    all_names = []

    for image_path in image_paths:
        patches, names = extract_patches_one(image_path, stride=stride, slice_size=slice_size, save_images=save_images)
        all_patches.extend(patches)
        all_names.extend(names)

    return all_patches, all_names

def get_sliced_images_simple(image_paths, slice_size=(10, 10), stride=1, save_images=False):
    all_patches = []
    all_names = []

    for image_path in image_paths:
        patches, names = extract_patches_one(image_path, stride=stride, slice_size=slice_size, save_images=save_images)
        all_patches.extend(patches)
        all_names.extend(names)

    return all_patches, all_names

def get_images(image_paths, get_slices=True, slice_size=(10, 10), stride=1, save_images=False):
    if get_slices:
        return get_sliced_images(
            image_paths=image_paths,
            slice_size=slice_size,
            stride=stride,
            save_images=save_images,
        )
    return [], []

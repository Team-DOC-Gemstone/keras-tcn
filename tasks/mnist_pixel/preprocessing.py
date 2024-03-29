import numpy as np
from scipy import ndimage
import os

def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 512
    desired_height = 512
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def process_scan(path):
    """Read and resize volume"""
    print("Processing scan " + str(path))
    # Read scan
    volume = np.load(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume

def get_normal():
    normal_path = "/fs/class-projects/spring2024/gems497/ge497g00/normal"
    preprocessed_file_path = "/fs/class-projects/spring2024/gems497/ge497g00/normal_scans_preprocessed_tcn.npy"
    preprocessed_labels = "/fs/class-projects/spring2024/gems497/ge497g00/normal_labels_tcn.npy"
    if (os.path.isfile(preprocessed_file_path) and os.path.isfile(preprocessed_labels)):
        normal_scans = np.load(preprocessed_file_path)
        normal_labels = np.load(preprocessed_labels)
        return (normal_scans, normal_labels)

    normal_scan_paths = [
        os.path.join(os.getcwd(), normal_path, x)
        for x in os.listdir(normal_path)
    ]

    print("CT scans with normal lung tissue: " + str(len(normal_scan_paths)))
    print("normal scan processing")
    normal_scans = np.array([process_scan(path) for path in normal_scan_paths])
    with open(preprocessed_file_path, 'wb') as f:
        np.save(f, normal_scans)
    normal_labels = np.array([0 for _ in range(len(normal_scans))])
    with open(preprocessed_labels, 'wb') as s:
        np.save(s, normal_labels)
    return (normal_scans, normal_labels)

def get_abnormal():
    preprocessed_file_path = "/fs/class-projects/spring2024/gems497/ge497g00/abnormal_scans_preprocessed_tcn.npy"
    preprocessed_labels = "/fs/class-projects/spring2024/gems497/ge497g00/abnormal_labels_tcn.npy"
    if (os.path.isfile(preprocessed_file_path)):
        abnormal_scans = np.load(preprocessed_file_path)
        abnormal_labels = np.load(preprocessed_labels)
        return (abnormal_scans, abnormal_labels)

    cancerous_path = "/fs/class-projects/spring2024/gems497/ge497g00/usable-cancerous"
    abnormal_scan_paths = [
        os.path.join(os.getcwd(), cancerous_path, x)
        for x in os.listdir(cancerous_path)
    ]
    print("CT scans with cancerous lung tissue: " + str(len(abnormal_scan_paths)))
    print("abnormal scans processing")
    abnormal_scans = np.array([process_scan(path) for path in abnormal_scan_paths])
    with open(preprocessed_file_path, 'wb') as f:
        np.save(f, abnormal_scans)
    abnormal_labels = np.array([0 for _ in range(len(abnormal_scans))])
    with open(preprocessed_labels, 'wb') as s:
        np.save(s, abnormal_labels)
    return (abnormal_scans, abnormal_labels)

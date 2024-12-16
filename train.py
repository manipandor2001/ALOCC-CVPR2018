import os
import numpy as np
from models import ALOCC_Model
from utils import pp, visualize, to_json, show_all_variables
import tensorflow as tf


# Replace TensorFlow Flags with argparse for better compatibility in TF2
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train or Test ALOCC Model")
    parser.add_argument("--epoch", type=int, default=40, help="Epoch to train")
    parser.add_argument("--learning_rate", type=float, default=0.002, help="Learning rate for Adam")
    parser.add_argument("--beta1", type=float, default=0.5, help="Momentum term of Adam")
    parser.add_argument("--attention_label", type=int, default=1, help="Conditioned label that growth attention of training label")
    parser.add_argument("--r_alpha", type=float, default=0.2, help="Refinement parameter")
    parser.add_argument("--train_size", type=float, default=np.inf, help="The size of train images")
    parser.add_argument("--batch_size", type=int, default=128, help="The size of batch images")
    parser.add_argument("--input_height", type=int, default=45, help="The size of image to use")
    parser.add_argument("--input_width", type=int, default=None, help="The size of image to use")
    parser.add_argument("--output_height", type=int, default=45, help="The size of the output images to produce")
    parser.add_argument("--output_width", type=int, default=None, help="The size of the output images to produce")
    parser.add_argument("--dataset", type=str, default="UCSD", help="The name of the dataset")
    parser.add_argument("--dataset_address", type=str, default="./dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train", help="The path of dataset")
    parser.add_argument("--input_fname_pattern", type=str, default="*", help="Glob pattern of input filenames")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint", help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="log", help="Directory to save logs")
    parser.add_argument("--sample_dir", type=str, default="samples", help="Directory to save image samples")
    parser.add_argument("--train", type=bool, default=True, help="True for training, False for testing")






    return parser.parse_args()

def check_some_assertions(flags):
    if flags.input_width is None:
        flags.input_width = flags.input_height
    if flags.output_width is None:
        flags.output_width = flags.output_height

    os.makedirs(flags.checkpoint_dir, exist_ok=True)
    os.makedirs(flags.log_dir, exist_ok=True)
    os.makedirs(flags.sample_dir, exist_ok=True)

def main():
    flags = parse_args()
    pp.pprint(vars(flags))

    n_per_itr_print_results = 100
    kb_work_on_patch = True

    # Dataset and slicing parameters
    nd_input_frame_size = (240, 360)  # Example for UCSD
    nd_slice_size = (45, 45)
    n_stride = 25
    n_fetch_data = 600

    flags.input_width = nd_slice_size[0]
    flags.input_height = nd_slice_size[1]
    flags.output_width = nd_slice_size[0]
    flags.output_height = nd_slice_size[1]

    flags.sample_dir = f'export/{flags.dataset}_{nd_slice_size[0]}.{nd_slice_size[1]}'

    check_some_assertions(flags)

    # Configure GPUs in TF2
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError as e:
            print(e)

    tmp_model = ALOCC_Model(
        input_width=flags.input_width,
        input_height=flags.input_height,
        output_width=flags.output_width,
        output_height=flags.output_height,
        batch_size=flags.batch_size,
        sample_num=flags.batch_size,
        attention_label=flags.attention_label,
        r_alpha=flags.r_alpha,
        dataset_name=flags.dataset,
        dataset_address=flags.dataset_address,
        input_fname_pattern=flags.input_fname_pattern,
        checkpoint_dir=flags.checkpoint_dir,
        is_training=flags.train,
        log_dir=flags.log_dir,
        sample_dir=flags.sample_dir,
        nd_patch_size=nd_slice_size,
        n_stride=n_stride,
        n_per_itr_print_results=n_per_itr_print_results,
        kb_work_on_patch=kb_work_on_patch,
        nd_input_frame_size=nd_input_frame_size,
        n_fetch_data=n_fetch_data
    )

    if flags.train:
        print('Program is in Train Mode')
        tmp_model.train(flags)
    else:
        if not tmp_model.load(flags.checkpoint_dir):
            print('Program is in Test Mode')
            raise Exception("[!] Train a model first, then run test mode from file test.py")

if __name__ == "__main__":
    main()

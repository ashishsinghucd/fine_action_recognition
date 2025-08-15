import os
import argparse
import logging
import json
from collections import defaultdict

# --- 1. Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


def build_class_mapping(splits_folder, file_names):
    """
    Scans all split files to find unique class names and create a mapping to integers.

    Args:
        splits_folder (str): The folder containing the split files.
        file_names (list): A list of filenames to scan (e.g., ['train.txt', 'val.txt']).

    Returns:
        dict: A dictionary mapping class names to integer labels, or None if an error occurs.
    """
    logging.info("Building class mapping from all split files...")
    unique_classes = set()
    try:
        for file_name in file_names:
            file_path = os.path.join(splits_folder, file_name)
            if not os.path.exists(file_path):
                logging.warning(f"File {file_path} not found, skipping for class mapping.")
                continue

            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Assumes class name is everything before the last underscore
                    class_name = line.rsplit('_', 1)[0]
                    unique_classes.add(class_name)

        if not unique_classes:
            logging.error("No classes found in any of the provided files. Cannot create mapping.")
            return None

        # Sort for consistent mapping and create the dictionary
        sorted_classes = sorted(list(unique_classes))
        class_to_idx = {name: i for i, name in enumerate(sorted_classes)}

        logging.info(f"Found {len(class_to_idx)} unique classes.")
        return class_to_idx

    except Exception as e:
        logging.error(f"Failed to build class mapping. Error: {e}")
        return None


def process_split_file(input_path, output_path, class_mapping, path_prefix):
    """
    Converts a single split file to the new format using the provided class mapping.

    Args:
        input_path (str): Path to the source split file (e.g., 'train.txt').
        output_path (str): Path to save the new formatted file.
        class_mapping (dict): The dictionary mapping class names to labels.
        path_prefix (str): The path to prepend to each video filename.
    """
    logging.info(f"Processing {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
    lines_processed = 0
    try:
        with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue

                video_id = line
                class_name = video_id.rsplit('_', 1)[0]

                if class_name in class_mapping:
                    class_label = class_mapping[class_name]
                    # Create the full video path with the prefix
                    full_video_path = os.path.join(path_prefix, f"{video_id}.mp4")
                    # Write in the format: path/to/video_name.mp4 class_label
                    f_out.write(f"{full_video_path} {class_label}\n")
                    lines_processed += 1
                else:
                    logging.warning(f"Class '{class_name}' from file {input_path} not found in mapping. Skipping line.")

        logging.info(f"Successfully processed {lines_processed} lines.")

    except Exception as e:
        logging.error(f"Failed to process file {input_path}. Error: {e}")


def main(args):
    """
    Main function to orchestrate the file processing.
    """
    splits_folder = args.input_folder
    path_prefix = args.path_prefix
    split_files = ['train.txt', 'val.txt', 'test.txt']

    # --- 1. Build Class Mapping ---
    class_mapping = build_class_mapping(splits_folder, split_files)
    if class_mapping is None:
        return

    # --- 2. Save Class Mapping to JSON ---
    mapping_path = os.path.join(splits_folder, 'class_mapping.json')
    try:
        with open(mapping_path, 'w') as f:
            json.dump(class_mapping, f, indent=4)
        logging.info(f"Class mapping saved to {mapping_path}")
    except Exception as e:
        logging.error(f"Could not save class mapping JSON file. Error: {e}")
        return

    # --- 3. Process each split file ---
    for file_name in split_files:
        input_path = os.path.join(splits_folder, file_name)
        if os.path.exists(input_path):
            # Standardize output filenames to use '_split_1_'
            output_name = f"{file_name.split('.')[0]}_split_1_videos.txt"
            output_path = os.path.join(splits_folder, output_name)
            process_split_file(input_path, output_path, class_mapping, path_prefix)
        else:
            logging.warning(f"Input file not found, cannot generate output for: {file_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Format action recognition annotation files for MMAction2.")
    parser.add_argument('--input_folder', type=str, required=True,
                        help="Path to the folder containing train.txt, val.txt, and test.txt.")
    parser.add_argument('--path_prefix', type=str, required=True,
                        help="Path to prepend to each video file in the output annotations (e.g., 'videos/train').")

    # Example usage:
    # python your_script_name.py --input_folder /path/to/splits --path_prefix videos

    args = parser.parse_args()
    main(args)

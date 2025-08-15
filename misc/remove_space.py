import os
import argparse
import logging

# --- 1. Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


def clean_annotation_file(file_path):
    """
    Reads an annotation file, removes spaces from the file path in each line,
    and overwrites the file with the cleaned content.

    Args:
        file_path (str): The full path to the annotation file to clean.
    """
    if not os.path.exists(file_path):
        logging.warning(f"File not found, skipping: {file_path}")
        return

    logging.info(f"Cleaning file: {os.path.basename(file_path)}...")
    cleaned_lines = []
    lines_changed = 0

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Split the line at the last space to separate the path from the label
            parts = line.rsplit(' ', 1)
            if len(parts) == 2:
                file_path_part, label_part = parts

                # Remove all spaces from the file path part
                cleaned_path = file_path_part.replace(' ', '')

                # Reconstruct the line
                new_line = f"{cleaned_path} {label_part}\n"
                cleaned_lines.append(new_line)

                # Track if a change was made
                if cleaned_path != file_path_part:
                    lines_changed += 1
            else:
                # If the line doesn't have a space, add it back as is
                cleaned_lines.append(line + '\n')

        # Overwrite the original file with the cleaned content
        with open(file_path, 'w') as f:
            f.writelines(cleaned_lines)

        logging.info(f"Finished cleaning. {lines_changed} lines were modified.")

    except Exception as e:
        logging.error(f"Failed to clean file {os.path.basename(file_path)}. Error: {e}")


def main(args):
    """
    Main function to find and clean all relevant annotation files.
    """
    splits_folder = args.input_folder

    # Define the files to be cleaned
    files_to_clean = [
        'train_split_1_videos.txt',
        'val_split_1_videos.txt',
        'test_split_1_videos.txt'
    ]

    for file_name in files_to_clean:
        full_path = os.path.join(splits_folder, file_name)
        clean_annotation_file(full_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clean annotation files by removing spaces from file paths.")
    parser.add_argument('--input_folder', type=str, required=True,
                        help="Path to the folder containing the annotation text files.")

    # Example usage:
    # python your_script_name.py --input_folder /path/to/your/splits

    args = parser.parse_args()
    main(args)

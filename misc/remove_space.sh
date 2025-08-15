#!/bin/bash

# This script renames .mp4 files in a specified directory by removing all spaces
# from their filenames. Example: 'play_ otamatone_017.mp4' -> 'play_otamatone_017.mp4'

# --- 1. Check for Input ---
# Exit with a usage message if no directory path is provided.
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/your_videos_folder"
    exit 1
fi

TARGET_DIR="$1"

# --- 2. Validate Input ---
# Check if the provided path is a valid directory.
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: '$TARGET_DIR' is not a valid directory."
    exit 1
fi

echo "Searching for .mp4 files with spaces in '$TARGET_DIR'..."

# --- 3. Find and Rename Files ---
# Loop through all .mp4 files in the target directory that contain a space.
# The `find` command is the safest way to handle filenames with special characters.
find "$TARGET_DIR" -maxdepth 1 -type f -name "* *.mp4" | while IFS= read -r file_path; do
    # Get just the filename from the full path
    filename=$(basename "$file_path")

    # Create the new filename by removing all spaces
    # This uses bash's built-in string replacement for efficiency
    new_filename="${filename// /}"

    # Construct the full path for the new filename
    new_path="$TARGET_DIR/$new_filename"

    # Rename the file using the -n (no-clobber) flag to prevent overwriting
    mv -n "$file_path" "$new_path"
    echo "Renamed: '$filename' -> '$new_filename'"
done

echo "Renaming complete."

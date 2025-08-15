#!/bin/bash

# This script moves all files from subdirectories of a given
# parent directory into the parent directory itself.

# --- 1. Check for Input ---
# Exit with a usage message if no directory path is provided.
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/parent_directory"
    exit 1
fi

PARENT_DIR="$1"

# --- 2. Validate Input ---
# Check if the provided path is a valid directory.
if [ ! -d "$PARENT_DIR" ]; then
    echo "Error: '$PARENT_DIR' is not a valid directory."
    exit 1
fi

echo "Moving all files from sub-folders into '$PARENT_DIR'..."

# --- 3. Find and Move Files ---
# -mindepth 2: Starts searching two levels deep (i.e., inside the sub-folders).
# -type f: Finds only files, not directories.
# -exec mv -n '{}' "$PARENT_DIR" \; : Executes the move command for each file found.
#   - The '-n' (no-clobber) flag prevents overwriting if a file with the same name already exists in the parent directory.
#   - '{}' is a placeholder for the file path found by the find command.
find "$PARENT_DIR" -mindepth 2 -type f -exec mv -n '{}' "$PARENT_DIR" \;

echo "Move complete."
echo "Note: This script does not overwrite files with the same name."

# --- 4. Optional: Clean Up Empty Directories ---
echo
echo "To remove the now-empty subdirectories, you can run the following command:"
echo "find \"$PARENT_DIR\" -mindepth 1 -type d -empty -delete"
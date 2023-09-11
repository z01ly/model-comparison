#!/bin/bash


destination_dir="../infoVAE/tsne/selected_sdss"

image_paths_file="../infoVAE/tsne/selected_sdss_filenames.txt"

# Check if the destination directory exists
if [ ! -d "$destination_dir" ]; then
    mkdir -p "$destination_dir"
fi

echo "Start.........."

while IFS= read -r full_image_path; do
    # Extract the filename from the full path
    filename=$(basename "$full_image_path")

    # Check if the image file exists
    if [ -f "$full_image_path" ]; then
        cp "$full_image_path" "$destination_dir/$filename"
    else
        echo "File not found: $filename"
    fi
done < "$image_paths_file"

echo "Copy process completed."

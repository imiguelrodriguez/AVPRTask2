import os
import shutil
import xml.etree.ElementTree as ET

# Directories
image_dir = "Stanford40/JPEGImages"  # Directory containing the images
splits_dir = "Stanford40/ImageSplits"  # Directory containing the .txt splits
output_dir = "data"  # Output directory

# Create the output directory and subdirectories
os.makedirs(os.path.join(output_dir, 'train', 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'test', 'images'), exist_ok=True)


def process_action_splits():
    # Iterate through the split files (action-specific ones)
    for split_file in os.listdir(splits_dir):
        # We want to filter out the general split files like 'train.txt' and 'test.txt'
        if split_file.endswith('_train.txt') or split_file.endswith('_test.txt'):
            split_type = 'train' if 'train' in split_file else 'test'

            # Read the action from the file name, like 'applauding' from 'applauding_train.txt'
            action = '_'.join(split_file.split('_')[:-1])  # Join all parts except for 'train' or 'test'

            # Create directory for the action (label) in the corresponding split directory
            split_action_dir = os.path.join(output_dir, split_type, action)
            os.makedirs(split_action_dir, exist_ok=True)

            # Read the list of image filenames from the split file
            with open(os.path.join(splits_dir, split_file), 'r') as f:
                img_names = [line.strip() for line in f.readlines()]

            # Process each image in the split file
            for img_name in img_names:
                img_path = os.path.join(image_dir, img_name)

                # Ensure the image exists in the images folder
                if os.path.exists(img_path):
                    # Copy the image to the appropriate action subdirectory
                    shutil.copy(img_path, os.path.join(split_action_dir, img_name))


# Main function
def main():
    # Process the splits (train/test) for each action
    process_action_splits()
    print(f"Data has been organized into '{output_dir}/train' and '{output_dir}/test'.")

# Run the script
if __name__ == '__main__':
    main()

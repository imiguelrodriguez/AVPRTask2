import os

from PIL import Image
from torch.utils.data import Dataset


class Stanford40Dataset(Dataset):
    def __init__(self, data_dir, split, transform=None):
        """
        Args:
            data_dir (str): Directory where 'train' and 'test' subdirectories are stored.
            split (str): 'train' or 'test' to specify which split to load.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.images = self._load_images()

        # Create a mapping from class names to integer labels
        self.classes = {class_name for class_name in
                             sorted(os.listdir(os.path.join(data_dir, split)))}
        self.class_to_idx = {class_name: idx for idx, class_name in
                             enumerate(sorted(os.listdir(os.path.join(data_dir, split))))}

    def _load_images(self):
        images = []
        split_dir = os.path.join(self.data_dir, self.split)  # Data directory for the specified split (train/test)

        # Iterate over the directories (each directory corresponds to an action)
        for action in os.listdir(split_dir):
            action_dir = os.path.join(split_dir, action)
            if os.path.isdir(action_dir):  # Check if it's a directory (action)
                # Get all the image filenames for the action
                for img_name in os.listdir(action_dir):
                    img_path = os.path.join(action_dir, img_name)
                    if os.path.isfile(img_path):
                        images.append((img_path, action))

        print(f"Loaded {len(images)} images from {self.split} split.")
        return images

    def __len__(self):
        # Return the total number of samples
        return len(self.images)

    def __getitem__(self, idx):
        # Get the image file path and label from the images list
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')  # Open image and convert to RGB

        if self.transform:
            image = self.transform(image)

        # Convert the label from string to integer using class_to_idx
        label_idx = self.class_to_idx[label]

        return image, label_idx

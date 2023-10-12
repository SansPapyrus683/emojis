import os
import pathlib
from torch.utils.data import Dataset
from torchvision import io, transforms


class EmojiDataset(Dataset):
    def __init__(self, image_folder_path: str, desired_image_size: tuple[int, int]):
        # initialize the data from the path
        self.data = []
        start = pathlib.Path(image_folder_path)
        resize = transforms.Resize(desired_image_size)
        for domain in os.listdir(start):
            if domain in ["KDDI", "DoCoMo", "SoftBank", "Gmail"]:
                continue
            for img in os.listdir(start / domain):
                path = start / domain / img
                tensor = io.read_image(str(path), io.ImageReadMode.RGB)
                self.data.append(resize(tensor))

    def __getitem__(self, i):
        # return the ith image as a tensor
        return self.data[i]

    def __len__(self):
        # return the length of the dataset
        return len(self.data)

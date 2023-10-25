import os
import pathlib

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class EmojiDataset(Dataset):
    def __init__(self, image_folder_path: str, desired_image_size: tuple[int, int]):
        # initialize the data from the path
        self.paths = []
        start = pathlib.Path(image_folder_path)
        self.ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(desired_image_size, antialias=True)
        ])
        for domain in os.listdir(start):
            for img in os.listdir(start / domain):
                self.paths.append(start / domain / img)

    def __getitem__(self, i):
        # return the ith image as a tensor
        path = self.paths[i]
        im = Image.open(path)
        rgb_im = im.convert("RGBA").convert("RGB")
        return self.ops(rgb_im)

    def __len__(self):
        # return the length of the dataset
        return len(self.paths)


if __name__ == "__main__":
    data = EmojiDataset("data/image", (64, 64))
    sample = data[0]

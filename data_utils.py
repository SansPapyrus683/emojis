import os
import pathlib

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class EmojiDataset(Dataset):
    def __init__(self, image_folder_path: str, desired_image_size: tuple[int, int]):
        # initialize the data from the path
        self.data = []
        start = pathlib.Path(image_folder_path)
        ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(desired_image_size)
        ])
        for domain in os.listdir(start):
            for img in os.listdir(start / domain):
                path = start / domain / img
                im = Image.open(path)
                rgb_im = im.convert("RGBA")
                self.data.append(ops(rgb_im))

    def __getitem__(self, i):
        # return the ith image as a tensor
        return self.data[i]

    def __len__(self):
        # return the length of the dataset
        return len(self.data)


if __name__ == "__main__":
    data = EmojiDataset("data/image", (64, 64))
    print(data[0].shape)

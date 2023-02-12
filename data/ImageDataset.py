import data.util as Util
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, dataroot, resolution=256, split='train', data_len=-1):
        self.res = resolution
        self.data_len = data_len
        self.split = split
        self.path = []

        # support combine data_folders to train
        dataroots = []
        if isinstance(dataroot, str):
            dataroots.append(dataroot)
        else:
            dataroots = dataroot
        for every_folder in dataroots:
            self.path.extend(Util.get_paths_from_images(every_folder))

        self.dataset_len = len(self.path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img = Image.open(self.path[index]).convert("RGB")

        img = Util.transform_augment(img, split=self.split, min_max=(-1, 1), res=self.res)

        return {'img': img, 'Index': index}

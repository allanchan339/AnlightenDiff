import pytorch_lightning as pl
from torchvision import transforms as T
from torchvision.transforms.functional import hflip, vflip, rotate
from os.path import join
from torch.utils.data import DataLoader, Dataset
import os
from os import listdir
import sys
from PIL import Image, ImageOps
import random
from utils.module_util import identity, default
# import dill as pickle # dont know what it is but to fix bug 
# extra
# os.environ["DEEPLAKE_DOWNLOAD_PATH"] = './data/LOL/fromDeepLake'
path = os.getcwd()
sys.path.append(path)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".bmp", ".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    # y, _, _ = img.split()
    return img


def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

def random_crop_identical(img_in, img_tar, patch_size):
    # replace with torchvision
    # for train, random crop

    # Define the transform to perform the random crop
    crop_transform = T.RandomCrop(size=(patch_size, patch_size))

    # Get the random cropping parameters for the first image
    i, j, h, w = crop_transform.get_params(img_tar, output_size=(patch_size, patch_size))

    # Apply the same cropping parameters to both images
    img_tar = T.functional.crop(img_tar, i, j, h, w)
    img_in = T.functional.crop(img_in, i, j, h, w)

    return img_in, img_tar

def get_patch(img_in, img_tar, patch_size, train=False, padding=False):
    if train:
        # XXX: consider RandomResizedCrop
        img_in, img_tar = random_crop_identical(img_in, img_tar, patch_size)

    elif not padding:
        mul = 32
        # for test, center crop with largest possible space
        # condition: must be divisible by 2^5 and >160
        (w, h) = img_tar.size #still in PIL, but return (w,h) instead of (h,w)
        h = (h // mul) * mul
        w = (w // mul) * mul

        centercrop = T.CenterCrop((h, w))
        img_in = centercrop(img_in)
        img_tar = centercrop(img_tar)
    
    return img_in, img_tar

def augment(img_in, img_tar, flip_h = True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if flip_h and random.random() > 0.5:
        img_in = hflip(img_in)
        img_tar = hflip(img_tar)
        info_aug['flip_h'] = True

    if rot and random.random() > 0.5:
        # either 180 or 90 
        if random.random() > 0.5: 
            img_in = vflip(img_in)
            img_tar = vflip(img_tar)
            info_aug['flip_v'] = True

        else:
            img_in = rotate(img_in, 90)
            img_tar = rotate(img_tar, 90)
            info_aug['trans'] = True

    return img_in, img_tar, info_aug

class DatasetFromFolder(Dataset):
    def __init__(self, folders, scale_factor=1.0, transform= identity):
        super(DatasetFromFolder, self).__init__()
        self.hr_image_filenames = []
        self.lr_image_filenames = []

        for folder in folders:
            HR_dir = folder + "/high"
            LR_dir = folder + '/low'

            hr_image_filenames = [join(HR_dir, x)
                                for x in listdir(HR_dir) if is_image_file(x)]
            # take HR_dir list to generate LR_dir list
            lr_image_filenames = [join(LR_dir, x)
                                for x in listdir(HR_dir) if is_image_file(x)]
            lr_image_filenames = [x.replace('normal', 'low') for x in lr_image_filenames]

            self.hr_image_filenames += hr_image_filenames
            self.lr_image_filenames += lr_image_filenames
            
        self.scale_factor = scale_factor
        self.transform = transform
    def __getitem__(self, index):

        target = load_img(self.hr_image_filenames[index])
        # name = self.hr_image_filenames[index]
        # lr_name = name[:25]+'LR/'+name[28:-4]+'x4.png'
        # lr_name = name[:18] + 'LR_4x/' + name[21:]
        input = load_img(self.lr_image_filenames[index])

        target = self.transform(target)
        input = self.transform(input)

        input_name = self.lr_image_filenames[index]
        input_name = os.path.basename(input_name)
        return input, target, input_name

    def __len__(self):
        return len(self.hr_image_filenames)

class PairImageDataset(Dataset):
    def __init__(self, ds, config, transform = None, upscale_factor=1.0, data_augmentation = False, padding=True) -> None:
        super().__init__()
        self.ds = ds
        self.transform = transform
        self.config = config
        self.upscale_factor = upscale_factor  # it is not SR task
        self.data_augmentation = data_augmentation
        self.padding = padding

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        input, target, input_name = self.ds[idx]
        # TODO: accelerate by using transforms
        if self.data_augmentation: # should work only on train
            img_in, img_tar, = get_patch(
            input, target, self.config.image_size, train = self.data_augmentation, padding=self.padding)
            img_in, img_tar, _ = augment(img_in, img_tar)
            # return [3,200,200] image
        else:
            # in valid mode
            # return [3, 200*n, 200*n] image
            img_in, img_tar, = get_patch(
            input, target, self.config.image_size, train = self.data_augmentation, padding=self.padding)

        if self.transform:
            img_in = self.transform(img_in)
            img_tar = self.transform(img_tar)

        return img_in.contiguous(), img_tar.contiguous(), input_name

class LitLOLDataModule(pl.LightningDataModule):
    def __init__(self, config, train_folders, test_folders) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.transform = T.Compose([
            T.ToTensor(), 
            # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
             
        self.train_folders = train_folders
        self.test_folders = test_folders

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = None):
        if stage == 'fit' :
            ds = DatasetFromFolder(self.train_folders, default(self.config.scale_factor, 1.0))
            self.ds_train = PairImageDataset(ds, self.config, self.transform, data_augmentation=True, padding=self.config.padding)

        ds_test = DatasetFromFolder(self.test_folders, self.config.scale_factor)
        self.ds = PairImageDataset(ds_test, self.config, self.transform, data_augmentation=False, padding=self.config.padding)

    def train_dataloader(self):
        dl = DataLoader(self.ds_train, batch_size=self.config.train_batch_size, shuffle=True,
                        pin_memory=self.config.pin_memory, num_workers=self.config.num_workers, persistent_workers=self.config.persistent_workers)
        return dl

    def val_dataloader(self):
        dl = DataLoader(self.ds, batch_size=self.config.batch_size, shuffle=False,
                        pin_memory=self.config.pin_memory, num_workers=self.config.num_workers)
        return dl

    def test_dataloader(self):
        dl = DataLoader(self.ds, batch_size=self.config.batch_size, shuffle=False,
                        pin_memory=self.config.pin_memory, num_workers=self.config.num_workers)
        return dl

if __name__ == '__main__':
    from types import SimpleNamespace

    config_LOL = SimpleNamespace(
        train_folders='data/LOL/our485',
        test_folders = 'data/LOL/eval15',
        scale_factor = 1.0,
        image_size = 160
    ) # at most have batch size = 15... 

    config_LOL4K = SimpleNamespace(
        train_folders='data/LOL4K/train',
        test_folders = 'data/LOL4K/test',
        scale_factor = 2.0
    )

    import random
    random.seed(42)
    ds = DatasetFromFolder([config_LOL.test_folders], 1)
    # img_lr, img_hr = ds[0]
    # l1, h1 = random_crop_identical(img_lr, img_hr, 160)
    # l2, h2 = random_crop_identical(img_lr, img_hr, 160)
    ds = PairImageDataset(ds, config_LOL, T.ToTensor(), data_augmentation=True, padding=True)
    l1, h1, l1name = ds[0]
    print()
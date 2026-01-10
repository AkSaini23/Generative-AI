from collections.abc import Callable
import os

import torch
from torch.utils.data import DataLoader, default_collate
from torchvision import datasets
from torchvision.transforms.v2 import (CenterCrop, Compose, CutMix, MixUp, RandomChoice, RandomHorizontalFlip,
                                       ToDtype, ToImage, Transform)

from .transforms import PadTransform, Transpose
from ..types import DataBatchFloat, DataFloat, LabelBatchFloat, LabelFloat
from ..visualization import plot_image_grid


# needed for cutmix/mixup
N_CLASSES = {"mnist": 10, "cifar10": 10, "fashion": 10, "svhn": 10, "emnist": 47, "cifar100": 100}


def get_datasets_and_loaders(dataset: str,
                             batch_size: int,
                             num_workers: int = 0,
                             root: str = "data",
                             augment_transforms: list[Transform] | None = None,
                             no_flip: bool = False,
                             cutmixup: bool = False,
                             additional_transforms: list[Transform] | None = None,
                             verbose: bool = True,
                             plot_descale: Callable[[DataBatchFloat], DataBatchFloat] | None = None) \
                                -> tuple[datasets.VisionDataset,
                                         datasets.VisionDataset,
                                         DataLoader[tuple[DataBatchFloat, LabelBatchFloat]],
                                         DataLoader[tuple[DataBatchFloat, LabelBatchFloat]]]:
    """Standard preparation of datasets (train/validation) and data loaders.

    Only for vision datasets.

    Parameters:
        dataset: Name of the dataset. Currently allowed are mnist, emnist, fashion, svhn, cifar10, cifar100, stl10,
                 flickr and celeba.
                 mnist: You know; MNIST. Handwritten digits 0-9. Gets padded to 32x32.
                 emnist: Like MNIST, but also includes letters.
                 fashion: FashionMNIST, structure like MNIST, but data is fashion items like shirts, shoes, bags...
                 svhn: Street View House Numbers. Not much more difficult than MNIST, but in color!
                 cifar10: Should know this one; 32x32 color images of various animals and vehicles.
                 cifar100: CIFAR but with 100 classes.
                 stl10: 96x96 color images. I don't know how many classes or what kind of objects we have. There are
                        very few labeled images for 10 classes, but many more varied unlabeled images. If 96 pixels is
                        too much, you could add a Resize transform to go to 48 or 64. Still more than CIFAR!
                 flickr: Color images of human faces, 128x128 assuming you downloaded the "thumbnails" version.
                         NOTE this dataset does come with torchvision. You have to download it here:
                         https://github.com/NVlabs/ffhq-dataset
                         Be sure to download the thumbnails version!! There is no explicit train/test separation. The
                         website says there should be 70,000 images, but there seem to be fewer, around 65,000. After
                         downloading and unpacking, I propose the following:
                         - Take the last five folders, i.e. named '65000' to '69000'.
                         - Put them in a separate directory flickr_test.
                         - Rename the other folder (originally thumbnails128x128) to flickr_train.
                         This gives us around 5000 test images and 60000 training images. Also NOTE, the dataset will
                         return labels as they are inferred by the folder names. But these are meaningless! This dataset
                         does not have labels.
                 celeba: Faces, but FAMOUS. We crop to square images, as the original ones are not. Since the images are
                         218x178, this leaves us with 178x178 square images.
        batch_size: Guess what!
        num_workers: Used by DataLoader.
        root: Base path where datasets should be stored/looked for.
        augment_transforms: Transforms you want to be applied only to the training data, i.e. data augmentation. Note
                            that most augmentations negatively affect data quality, and would make your generative model
                            learn to generate such data (e.g. rotated or transposed images). As such, this is only
                            really useful for training classifiers for evaluation.
        no_flip: An exception to the above comment is the random horizontal flip transform. This does not compromise
                 image quality, and is applied *by default* to all training data, unless it doesn't make sense for the
                 dataset, e.g. for digits. If you *don't* want data to be flipped randomly for whatever reason, you can
                 turn it off by setting this argument to True.
        cutmixup: If True, apply cutmix and mixup. DO NOT USE for generative models. Only used for training classifiers.
                  It's another augmentation, but goes into the dataloader, so unfortunately it's a separate argument.
        additional_transforms: Any transforms you want on BOTH training and testing data besides standard ToTensor() 
                               and possibly padding/flipping. For example, re-scaling images to range [-1, 1] instead of
                               [0, 1]. Check out the Normalize transform below. Another common use case could be
                               resizing. For example, resize flickr images to 32x32 or 64x64 to make training more
                               manageable. For that, use the Resize transform from torchvision.
        verbose: If True, print some info about the dataset elements (shape and dtype), as well as plotting some example
                 images.
        plot_descale: If you have supplied additional transforms that change the scale of the data, you likely want to
                      undo that (scale back to [0, 1]) before plotting. This function should do that. Only needed if
                      verbose is True.
    """
    # torch keeps telling me to use this instead of ToTensor...
    to_tensor = Compose([ToImage(), ToDtype(torch.float32, scale=True)])
    train_transforms = [to_tensor]
    test_transforms = [to_tensor]
    if dataset in ["mnist", "emnist", "fashion"]:
        # mnist is padded to 32 x 32 for convenience
        pad = PadTransform(2)
        train_transforms.append(pad)
        test_transforms.append(pad)
    elif dataset != "svhn" and not no_flip:
        # mnist/svhn should not be flipped, since flipping digits is not something that makes sense
        train_transforms.append(RandomHorizontalFlip())
    if dataset == "emnist":
        # for some reason EMNIST has height/width transposed...
        transpose = Transpose()
        train_transforms.append(transpose)
        test_transforms.append(transpose)
    if dataset == "celeba":
        crop = CenterCrop(178)
        train_transforms.append(crop)
        test_transforms.append(crop)

    if augment_transforms is not None:
        train_transforms += augment_transforms
    if additional_transforms is not None:
        train_transforms += additional_transforms
        test_transforms += additional_transforms
    train_transforms = Compose(train_transforms)
    test_transforms = Compose(test_transforms)

    if dataset == "flickr":
        train_data = datasets.ImageFolder(root=os.path.join(root, "flickr_train"), transform=train_transforms)
        test_data = datasets.ImageFolder(root=os.path.join(root, "flickr_test"), transform=test_transforms)
    elif dataset == "celeba":
        train_data = datasets.CelebA(root=root, split="train", transform=train_transforms, download=True)
        test_data = datasets.CelebA(root=root, split="valid", transform=test_transforms, download=True)
    elif dataset == "svhn":
        train_data_base = datasets.SVHN(root=root, split="train", transform=train_transforms, download=True)
        extra_data = datasets.SVHN(root=root, split="extra", transform=train_transforms, download=True)
        train_data = torch.utils.data.ConcatDataset((train_data_base, extra_data))
        test_data = datasets.SVHN(root=root, split="test", transform=test_transforms, download=True)
    elif dataset == "emnist":
        train_data = datasets.EMNIST(root=root, split="balanced", train=True, transform=train_transforms, download=True)
        test_data = datasets.EMNIST(root=root, split="balanced", train=False, transform=test_transforms, download=True)
    elif dataset == "stl10":
        train_data = datasets.STL10(root=root, split="train+unlabeled", transform=train_transforms, download=True)
        test_data = datasets.STL10(root=root, split="test", transform=test_transforms, download=True)
    else:
        if dataset == "mnist":
            constructor = datasets.MNIST
        elif dataset == "fashion":
            constructor = datasets.FashionMNIST
        elif dataset == "cifar10":
            constructor = datasets.CIFAR10
        elif dataset == "cifar100":
            constructor = datasets.CIFAR100
        train_data = constructor(root=root, train=True, transform=train_transforms, download=True)
        test_data = constructor(root=root, train=False, transform=test_transforms, download=True)

    if cutmixup:
        n_classes = N_CLASSES[dataset]
        cutmixup = RandomChoice([CutMix(num_classes=n_classes, alpha=0.2), MixUp(num_classes=n_classes, alpha=0.2)])

        def collate_fn(batch: list[tuple[DataFloat, LabelFloat]]) -> tuple[DataBatchFloat, LabelBatchFloat]:
            return cutmixup(*default_collate(batch))
    else:
        collate_fn = None
        
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True,
                                  drop_last=True, num_workers=num_workers, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, pin_memory=True,
                                 num_workers=num_workers)
    if verbose:
        images, y = next(iter(train_dataloader))
        print(f"Shape/dtype of batch X [N, C, H, W]: {images.shape}, {images.dtype}")
        print(f"Shape/dtype of batch y: {y.shape}, {y.dtype}")
        plot_image_grid(images[:128],
                        figure_size=(14, 7), title="Example images", n_rows=8, n_cols=16, plot_descale=plot_descale)
    return train_data, test_data, train_dataloader, test_dataloader

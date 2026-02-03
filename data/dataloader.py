from torchvision import datasets, transforms

from data.data_transformer import get_train_transforms, get_test_transforms
from  utils.random_config import set_seed

root = "./data"
aug_dir = "./data/aug_dir"

def get_datasets():
    train_dataset = datasets.CIFAR10(
                        root=root,
                        train= True,
                        download=True,
                        transform=get_train_transforms()
                        )

    test_dataset = datasets.CIFAR10(
                        root = root,
                        train=False,
                        download=True,
                        transform=get_test_transforms()
                        )

    return train_dataset , test_dataset

#aug_dataset = augment_data(train_dataset, aug_dir,num_aug_images,aug_exist)
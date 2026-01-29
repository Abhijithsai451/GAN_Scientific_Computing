from torchvision import datasets, transforms
from  utils.random_config import set_seed

set_seed(42)
TIN_MEAN = (0.5, 0.5, 0.5)
TIN_STD = (0.5, 0.5, 0.5)
root = "./data"
def get_train_transforms():
    """
    Transforms the training Images.
    1. Normalizes the image
    2.

    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = TIN_MEAN, std = TIN_STD)
    ])
    return transform

def get_test_transforms():
    pass

def augment_data(dataset):
    return dataset


train_transforms = get_train_transforms()
test_transforms = get_test_transforms()

def get_datasets():
    train_dataset = datasets.CIFAR10(
                        root=root,
                        train= True,
                        download=True,
                        transform=train_transforms
                        )

    test_dataset = datasets.CIFAR10(
                        root = root,
                        train=False,
                        download=True,
                        transform=test_transforms
                        )

    return train_dataset , test_dataset
#
#aug_dataset = augment_data(train_dataset)
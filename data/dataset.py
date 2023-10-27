import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

normalize = transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )

dataset_CIFAR10_train = datasets.CIFAR10(
    root='./data/cifar10', train=True,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop( 32, 4 ),
        transforms.ToTensor(),
        normalize,
    ]),
    download=True
)

dataset_CIFAR10_val = datasets.CIFAR10(
    root='./data/cifar10', train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
)

dataset_CIFAR100_train = datasets.CIFAR100(
    root='./data/cifar100', train=True,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop( 32, 4 ),
        transforms.ToTensor(),
        normalize,
    ]),
    download=True
)

dataset_CIFAR100_val = datasets.CIFAR100(
    root='./data/cifar100', train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
)
evens = list( range( 0, len( dataset_CIFAR100_train ), 2 ) )
odds = list( range( 1, len( dataset_CIFAR100_train ), 2 ) )
dataset_CIFAR100_sub1 = torch.utils.data.Subset( dataset_CIFAR100_train, evens )
dataset_CIFAR100_sub2 = torch.utils.data.Subset( dataset_CIFAR100_train, odds )

dataset_Pub = datasets.ImageFolder(
    root='./data/imagenet/val',
    transform=transforms.Compose([
        transforms.Resize( 36 ),
        transforms.CenterCrop( 32 ),
        transforms.ToTensor(),
        normalize,
    ])
)

dataset_ImageNet_train = datasets.ImageFolder(
    root='./data/imagenet/train',
    transform=transforms.Compose([
        transforms.RandomResizedCrop( 224 ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
)

dataset_ImageNet_val = datasets.ImageFolder(
    root='./data/imagenet/val',
    transform=transforms.Compose([
        transforms.Resize( 256 ),
        transforms.CenterCrop( 224 ),
        transforms.ToTensor(),
    ])
)

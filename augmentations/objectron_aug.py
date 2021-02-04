import torchvision.transforms as T
from torchvision.transforms.transforms import RandomResizedCrop, Scale
try:
    from torchvision.transforms import GaussianBlur
except ImportError:
    from .gaussian_blur import GaussianBlur
    T.GaussianBlur = GaussianBlur
    
imagenet_mean_std = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]

def get_objectron_gpu_transform(image_size, horizontal_flip=False, mean_std=imagenet_mean_std):
    hflip = T.RandomHorizontalFlip()
    p_blur = 0.5 if image_size > 32 else 0 
    transform_list = [
        #T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        #hflip,
        #T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
        #T.RandomGrayscale(p=0.2),
        #T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=p_blur),
        #T.ToTensor(),
        T.Normalize(*mean_std)]
    #if not horizontal_flip:
    #    transform_list.remove(hflip) 
    transform = T.Compose(transform_list)
    return transform

class ObjectronTransform():
    def __init__(self, image_size, mean_std=imagenet_mean_std, simsiam_transform=False, train=True, horizontal_flip=True):
        image_size = 224 if image_size is None else image_size # by default simsiam use image size 224
        p_blur = 0.5 if image_size > 32 else 0 
        # the paper didn't specify this, feel free to change this value
        # I use the setting from simclr which is 50% chance applying the gaussian blur
        # the 32 is prepared for cifar training where they disabled gaussian blur


        #torchvision.transforms.Pad(padding, fill=0, padding_mode='constant')
        if simsiam_transform:
            hflip = T.RandomHorizontalFlip()
            transform_list = [
                T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                hflip,
                T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=p_blur),
                T.ToTensor(),
                T.Normalize(*mean_std)]
            if not horizontal_flip:
                transform_list.remove(hflip) 
            self.transform = T.Compose(transform_list)
        else:
            #No transforms, except normalisation.
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(*mean_std)
            ])

    def __call__(self, x):
        return self.transform(x) 


class ObjectronVideoTransform():
    def __init__(self, image_size, mean_std=imagenet_mean_std, train=True):
        image_size = 224 if image_size is None else image_size # by default simsiam use image size 224
        p_blur = 0.5 if image_size > 32 else 0 
        # the paper didn't specify this, feel free to change this value
        # I use the setting from simclr which is 50% chance applying the gaussian blur
        # the 32 is prepared for cifar training where they disabled gaussian blur

        transform_list = [
            T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            T.RandomResizedCrop(image_size, scale=(0.8,1), ratio=(0.95,1.1)), #Small variation.
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=p_blur),
            T.ToTensor(),
            T.Normalize(*mean_std),
            T.RandomErasing(), #T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            ]
        self.transform = T.Compose(transform_list)


    def __call__(self, x):
        return self.transform(x) 
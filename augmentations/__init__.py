from .simsiam_aug import SimSiamTransform
from .eval_aug import Transform_single
from .byol_aug import BYOL_transform
from .simclr_aug import SimCLRTransform
from .objectron_aug import ObjectronTransform, ObjectronVideoTransform

def get_aug(name, image_size, train, train_classifier=True, horizontal_flip=True):
    if name != 'objectron_aug' and not horizontal_flip:
        raise ValueError("Disabling Horizontal Flips is only supported for Objectron Training")
    if train==True:
        if name == 'simsiam':
            augmentation = SimSiamTransform(image_size)
        elif name == 'byol':
            augmentation = BYOL_transform(image_size)
        elif name == 'simclr':
            augmentation = SimCLRTransform(image_size)
        elif name == 'objectron':
            augmentation = ObjectronTransform(image_size)
        elif name == 'objectron_aug':
            augmentation = ObjectronTransform(image_size,simsiam_transform=True, horizontal_flip=horizontal_flip)
        elif name =='objectron_video':
            augmentation = ObjectronVideoTransform(image_size)
        else:
            raise NotImplementedError
    elif train==False:
        if name=="simsiam" or name=="byol" or name=="simclr":
            augmentation = Transform_single(image_size, train=train_classifier)
        else:
            augmentation = ObjectronTransform(image_size, train=train_classifier)
    else:
        raise Exception
    
    return augmentation









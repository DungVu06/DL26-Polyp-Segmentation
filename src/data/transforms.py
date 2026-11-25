import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def _get_norm(use_pretrained: bool = False):
    if use_pretrained:
        return A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        )
    return A.Normalize(
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        max_pixel_value=255.0,
    )


def _parse_size(image_size):
    if isinstance(image_size, (list, tuple)):
        return image_size[0], image_size[1]
    return image_size, image_size


def build_unet_train_transform(image_size=(256, 256), use_pretrained: bool = False):
    h, w = _parse_size(image_size)
    norm = _get_norm(use_pretrained)
    return A.Compose([
        A.Resize(h, w, interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=30,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.3,
        ),
        norm,
        ToTensorV2(),
    ])


def build_unet_eval_transform(image_size=(256, 256), use_pretrained: bool = False):
    h, w = _parse_size(image_size)
    norm = _get_norm(use_pretrained)
    return A.Compose([
        A.Resize(h, w, interpolation=cv2.INTER_LINEAR),
        norm,
        ToTensorV2(),
    ])


def get_transforms(img_size: int = 256, use_pretrained: bool = False):
    return (
        build_unet_train_transform(img_size, use_pretrained),
        build_unet_eval_transform(img_size, use_pretrained),
    )

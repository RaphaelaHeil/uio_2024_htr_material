from math import ceil
from typing import Tuple, Iterable, Union, List, Callable

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as tvF
from PIL import Image
from skimage.morphology import square, disk, dilation, erosion
from skimage.transform import rescale
from torchvision.transforms.v2 import Resize


class PadSequence(object):

    def __init__(self, length: int, padValue: int = 0):
        self.length = length
        self.padValue = padValue

    def __call__(self, sequence: torch.Tensor):
        sequenceLength = sequence.shape[0]
        if sequenceLength == self.length:
            return sequence
        targetLength = self.length - sequenceLength
        return F.pad(sequence, pad=(0, targetLength), mode="constant", value=self.padValue)


class ResizeToHeight(Resize):

    def __init__(self, size: int):
        super().__init__(size)
        if isinstance(size, Tuple):
            self.height = size[0]
        else:
            self.height = size

    def forward(self, img: Image):
        oldWidth, oldHeight = img.size
        if oldHeight > oldWidth:
            scaleFactor = self.height / oldHeight
            intermediateWidth = ceil(oldWidth * scaleFactor)
            return tvF.resize(img, [self.height, intermediateWidth], self.interpolation, self.max_size, self.antialias)
        else:
            return super().forward(img)


class ResizeAndPad(object):
    """
    Custom transformation that maintains the image's original aspect ratio by scaling it to the given height and padding
    it to achieve the desired width.
    """

    def __init__(self, height: int, width: int, padwith: int = 1):
        self.width = width
        self.height = height
        self.padwith = padwith

    def __call__(self, img: Image):
        oldWidth, oldHeight = img.size
        if oldWidth == self.width and oldHeight == self.height:
            return img
        else:
            scaleFactor = self.height / oldHeight
            intermediateWidth = ceil(oldWidth * scaleFactor)
            if intermediateWidth > self.width:
                intermediateWidth = self.width
            resized = img.resize((intermediateWidth, self.height), resample=Image.BICUBIC)
            if img.mode == "RGB":
                padValue = (self.padwith, self.padwith, self.padwith)
            else:
                padValue = self.padwith
            preprocessed = Image.new(img.mode, (self.width, self.height), padValue)
            preprocessed.paste(resized)
            return preprocessed

    @classmethod
    def invert(cls, image: np.ndarray, targetShape: Tuple[int, int]) -> np.ndarray:
        # resize so that the height matches, then cut off at width ...
        originalHeight, originalWidth = image.shape
        scaleFactor = targetShape[0] / originalHeight
        resized = rescale(image, scaleFactor)
        return resized[:, :targetShape[1]]

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class _Morph(object):

    def __init__(self, method: Callable, seShape: str = "square", size: Union[int, Tuple[int, int], List[int]] = 3):
        self.method = method
        if seShape == "square":
            self.selem = square
        elif seShape == "disk":
            self.selem = disk

        if isinstance(size, Iterable):
            lower = min(size[0], size[1])
            upper = max(size[0], size[1])
            if lower == upper:
                self.shape = lower
                self.getSelem = self.__fixed_shape__
            else:
                self.shape = [lower, upper]
                self.getSelem = self.__random_shape__
        else:
            self.shape = size
            self.getSelem = self.__fixed_shape__

    def __random_shape__(self):
        return self.selem(np.random.randint(self.shape[0], self.shape[1]))

    def __fixed_shape__(self):
        return self.selem(self.shape)

    def __call__(self, img: Image):
        img = np.array(img)
        out = self.method(img, self.getSelem())
        return Image.fromarray(out)

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'({self.method.__name__})'


class Dilation(_Morph):

    def __init__(self, seShape: str = "square", size: Union[int, Tuple[int, int], List[int]] = 3):
        super().__init__(dilation, seShape, size)


class Erosion(_Morph):

    def __init__(self, seShape: str = "square", size: Union[int, Tuple[int, int], List[int]] = 3):
        super().__init__(erosion, seShape, size)

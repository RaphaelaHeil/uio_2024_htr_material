import string
from typing import List, Tuple

import torch
from torchvision import transforms
from torchvision.transforms.v2 import Compose, Grayscale, ToDtype, Transform, RandomInvert, RandomApply, ToImage

from htr.utils.config import Configuration
from htr.utils.transforms import PadSequence, ResizeAndPad

PAD_TOKEN = 0


class Encoder:

    def __init__(self):
        self.alphabet = list("ÄäÅåÖöØøÆæüéè–$«¬´»½")
        self.alphabet.extend(string.punctuation)
        self.alphabet.extend(string.digits)
        self.alphabet.extend(string.ascii_letters)
        self.alphabet.extend("�")
        self.alphabet.sort()
        self.alphabet.insert(0, "")  # blank
        self.alphabet.append(" ")  # space

    def encode(self, text: str) -> List[int]:
        encoded = []
        for char in text:
            if char in self.alphabet:
                encoded.append(self.alphabet.index(char))
            else:
                encoded.append(self.alphabet.index("�"))
        return encoded

    def decode(self, text: List[int]) -> str:
        return "".join([self.alphabet[c] for c in text if c != 0])

    @property
    def alphabetSize(self) -> int:
        return len(self.alphabet)


def composeTextTransformations(config: Configuration) -> Compose:
    if config.batchSize > 1:
        return transforms.Compose([PadSequence(length=config.transcriptionLength, padValue=PAD_TOKEN)])
    return transforms.Compose([])


def composeImageTransformations(config: Configuration) -> Tuple[Transform, Transform]:
    preAug = Compose([Grayscale(num_output_channels=1), RandomInvert(p=1.0)])

    postAug = Compose(
        [ResizeAndPad(height=config.padHeight, width=config.padWidth, padwith=config.padValue), ToImage(),
         ToDtype(torch.float32, scale=True)])

    # no padding:
    # postAug = Compose([ResizeToHeight(size=config.padHeight), ToTensor()])
    return preAug, postAug


def composeAugmentations(config: Configuration) -> Compose:
    # fill in your choice of augmentations
    augmentations = [Grayscale(num_output_channels=1)]

    # to randomly apply augmentations, use:
    # return RandomApply(transforms=augmentations, p=0.5)

    # to always apply augmentations, use:
    return None  # Compose(augmentations)

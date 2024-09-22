from pathlib import Path

from PIL import Image
from PIL.ImageOps import invert
from torchvision.transforms.v2 import ElasticTransform, RandomRotation, RandomAffine

from htr.utils.transforms import Dilation, Erosion


def show(original: Image.Image, augmented: Image.Image):
    width, height = original.size
    augWidth, augHeight = augmented.size

    joinedImage = Image.new('L', (max(width, augWidth), height + augHeight + 10))

    joinedImage.paste(original, (0, 0))
    joinedImage.paste(augmented, (0, height + 10))

    joinedImage.show()


def main():
    imagePath = Path("demo/sample_line.png")

    image = Image.open(imagePath).convert("L")
    augmented = invert(image)

    augmentations = [
        # Dilation/Erosion: seShape: "disk" or "square"; size: width, resp. diameter of structuring element, in pixels
        # if size is a tuple, value will be resampled from [min, max) for each image
        Dilation("disk", 5),
        # Erosion("square", 10),

        # ElasticTransform: alpha: magnitude, sigma: smoothness of displacement, respectively
        # ElasticTransform(alpha=50.0, sigma=5.0),

        # RandomRotation: degree of rotation, sampled from (min, max)
        # RandomRotation((5, 5)),

        # RandomAffine:
        # degree of rotation, sampled from (min, max)
        # fraction of translation (a,b) , horizontal sampled from (-a,a), vertical sampled from (-b,b)
        # scale, sampled from (min, max)
        # shear (xMin, xMax, yMin, yMax), x-shear sampled from (xMin, xMax), y-shear sampled from (yMin, yMax)
        # RandomAffine(degrees=(0, 0), translate=(0, 0), scale=(0.5, 0.5), shear=(0, 0, 0, 0)),
    ]

    for aug in augmentations:
        augmented = aug(augmented)

    show(image, invert(augmented))


if __name__ == '__main__':
    main()

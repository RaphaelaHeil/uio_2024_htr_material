import csv
from enum import Enum
from typing import Dict, Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from htr.utils.config import Configuration
from htr.utils.run_utils import composeImageTransformations, composeTextTransformations, Encoder


class DatasetMode(Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3
    INFER = 4


class LineDataset(Dataset):

    def __init__(self, config: Configuration, mode: DatasetMode, encoder: Encoder, augmentations=None):
        self.augmentations = augmentations

        self.encoder = encoder

        self.data = []
        if mode == DatasetMode.INFER:
            self.data = [{"path": p, "transcription": ""} for p in config.dataDir.glob("*.png")]
        else:
            dataDir = config.dataDir / mode.name.lower()
            with (dataDir / "index.tsv").open("r") as inFile:
                csvReader = csv.reader(inFile, delimiter='\t')
                self.data = [{"path": dataDir / row[0], "transcription": row[1]} for row in csvReader]

        self.preAugTransforms, self.postAugTransforms = composeImageTransformations(config)
        self.textTransforms = composeTextTransformations(config)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        lineData = self.data[index]

        transcription = lineData["transcription"]

        transcriptionEncoding = self.encoder.encode(transcription)
        transcriptionEncoding = torch.tensor(transcriptionEncoding)

        lineImage = Image.open(lineData["path"]).convert("L")

        lineImage = self.preAugTransforms(lineImage)

        if self.augmentations:
            lineImage = self.augmentations(lineImage)

        lineImage = self.postAugTransforms(lineImage)

        unpaddedLength = transcriptionEncoding.shape[0]

        transcriptionEncoding = self.textTransforms(transcriptionEncoding)

        return {"imageName": lineData["path"].name, "image": lineImage, "plaintext": transcription,
                "transcription": transcriptionEncoding, "t_len": unpaddedLength}

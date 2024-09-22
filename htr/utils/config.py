"""
Contains all code related to the configuration of experiments.
"""
import configparser
import random
from argparse import Namespace
from configparser import SectionProxy
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import List
from shutil import copy2

import torch


class Configuration:
    """
    Holds the configuration for the current experiment.
    """

    def __init__(self, filename: Path, fileSection: str = "DEFAULT", mode: str = "train",
                 inferenceDataPath: Path = None):

        configParser = configparser.ConfigParser()
        configParser.read(filename)

        if mode in ["test", "infer"] or "finetune" in mode:
            if len(configParser.sections()) > 0:
                parsedConfig = configParser[configParser.sections()[0]]
            else:
                parsedConfig = configParser["DEFAULT"]
        else:
            parsedConfig = configParser[fileSection]
            sections = configParser.sections()
            for s in sections:
                if s != fileSection:
                    configParser.remove_section(s)

        self.parsedConfig = parsedConfig

        self.inferenceModelFile = self.getSetStr("test_model_filename", "best_val.pth")

        if mode in ["test", "infer"]:
            self.outDir = filename.parent
        else:
            outDirName = f"{fileSection}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{random.randint(0, 100000)}"

            if "finetune" in mode:
                outDirName = f"finetune_{outDirName}"
                self.outDir = Path(self.parsedConfig.get("out_dir")).resolve().parent / outDirName
                self.outDir.mkdir(parents=True, exist_ok=True)
                copy2(Path(self.parsedConfig.get("out_dir")).resolve() / self.inferenceModelFile,
                      self.outDir / self.inferenceModelFile)
            else:
                self.outDir = Path(self.parsedConfig.get("out_dir")).resolve() / outDirName
                self.outDir.mkdir(parents=True, exist_ok=True)

            self.parsedConfig["out_dir"] = str(self.outDir)

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.epochs = self.getSetInt("epochs", 100)
        self.learningRate = self.getSetFloat("learning_rate", 0.001)
        self.earlyStoppingEpochCount = self.getSetInt("early_stopping_epoch_count", -1)

        self.batchSize = self.getSetInt("batch_size", 4)
        self.modelSaveEpoch = self.getSetInt("model_save_epoch", 10)
        self.validationEpoch = self.getSetInt("validation_epoch", 1)
        self.warmup = self.getSetInt("warmup_epochs", 0)
        if mode == "infer":
            self.dataDir = inferenceDataPath
        else:
            self.dataDir = Path(self.getSetStr("data_dir")).resolve()

        self.transcriptionLength = self.getSetInt("transcription_length", 80)
        self.padHeight = self.getSetInt('pad_height', 64)
        self.padWidth = self.getSetInt('pad_width', 1700)
        self.padValue = self.getSetInt("pad_value", 0)

        self.workerCount = self.getSetInt("workers", 1)

        if "train" in mode or "finetune" in mode:
            configOut = self.outDir / "config.cfg"
            with configOut.open("w+") as cfile:
                parsedConfig.parser.write(cfile)

    def getSetInt(self, key: str, default: int = None):
        value = self.parsedConfig.getint(key, default)
        self.parsedConfig[key] = str(value)
        return value

    def getSetFloat(self, key: str, default: float = None):
        value = self.parsedConfig.getfloat(key, default)
        self.parsedConfig[key] = str(value)
        return value

    def getSetBoolean(self, key: str, default: bool = None):
        value = self.parsedConfig.getboolean(key, default)
        self.parsedConfig[key] = str(value)
        return value

    def getSetStr(self, key: str, default: str = None):
        value = self.parsedConfig.get(key, default)
        self.parsedConfig[key] = str(value)
        return value

    @staticmethod
    def parseCSList(configString: str) -> List[str]:
        split = configString.split(",")
        result = [s.strip() for s in split]
        return result


def getConfiguration(args: Namespace) -> Configuration:
    fileSection = "DEFAULT"
    filename = "config.cfg"
    mode = "train"
    if "section" in args:
        fileSection = args.section
    if "config" in args:
        filename = args.config.resolve()
    if "mode" in args:
        mode = args.mode

    if mode == "infer":
        if args.eval.exists():
            return Configuration(filename, fileSection=fileSection, mode=mode, inferenceDataPath=args.eval.resolve())
        else:
            raise FileNotFoundError(args.eval)
    else:
        return Configuration(filename, fileSection=fileSection, mode=mode)

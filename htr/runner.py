import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import torch
from torch.nn import CTCLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics.text import CharErrorRate, WordErrorRate

from htr.dataset import LineDataset, DatasetMode
from htr.model import LoghiModel9
from htr.utils.config import Configuration
from htr.utils.log import initInfoLogger, initProgressLogger
from htr.utils.run_utils import composeAugmentations, Encoder


class Runner:

    def __init__(self, config: Configuration):
        self.config = config

        self.transcriptionEncoder = Encoder()

        self.model = LoghiModel9(self.transcriptionEncoder.alphabetSize).to(self.config.device)

        self.loss = CTCLoss(zero_infinity=True)

        self.optimiser = AdamW(self.model.parameters(), lr=self.config.learningRate)

        initInfoLogger(config.outDir, "info")

        self.infoLogger = logging.getLogger("info")

        self.cerMetric = CharErrorRate()
        self.werMetric = WordErrorRate()

        self.bestValidationCER = float("inf")
        self.bestValidationEpoch = 0

        self.evalLogger = None
        self.evalDataloader = None

    def __loadModelCheckpoint(self):
        state_dict = torch.load(self.config.outDir / self.config.inferenceModelFile,
                                map_location=torch.device(self.config.device))
        if 'model_state_dict' in state_dict.keys():
            state_dict = state_dict['model_state_dict']
        self.model.load_state_dict(state_dict)

    def __initTrain(self):
        initProgressLogger(self.config.outDir, "train")
        initProgressLogger(self.config.outDir, "validation")
        augmentations = composeAugmentations(self.config)

        trainDataset = LineDataset(self.config, DatasetMode.TRAIN, self.transcriptionEncoder, augmentations)
        self.trainDataloader = DataLoader(trainDataset, batch_size=self.config.batchSize, shuffle=True,
                                          num_workers=self.config.workerCount)
        evalDataset = LineDataset(self.config, DatasetMode.VALIDATION, self.transcriptionEncoder)
        self.evalDataloader = DataLoader(evalDataset, batch_size=self.config.batchSize, shuffle=False,
                                         num_workers=self.config.workerCount)
        self.evalLogger = logging.getLogger("validation")

    def finetune(self):
        self.infoLogger.info(f"Preparing model for finetuning.")
        self.__loadModelCheckpoint()
        self.train()

    def train(self):
        self.__initTrain()
        trainLogger = logging.getLogger("train")
        trainLogger.info("epoch,loss")
        validationLogger = logging.getLogger("validation")
        validationLogger.info("epoch,loss,cer,wer")

        self.infoLogger.info(f"Setup complete, commencing training with {self.config.device} device")

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            batchLosses = []
            epochStartTime = time.time()
            for batchId, data in enumerate(self.trainDataloader):
                lineImage = data["image"].to(self.config.device)
                predicted = self.model(lineImage)
                predicted = predicted.log_softmax(2)

                input_lengths = torch.full(size=(predicted.shape[1],), fill_value=predicted.shape[0], dtype=torch.long)

                encodedTranscription = data["transcription"].to(self.config.device)
                loss = self.loss(predicted, encodedTranscription, input_lengths, data["t_len"])
                loss.backward()

                self.optimiser.step()
                self.optimiser.zero_grad()

                batchLosses.append(loss.item())

            meanBatchLoss = np.mean(batchLosses)
            trainLogger.info(f"{epoch},{meanBatchLoss}")
            self.infoLogger.info(
                f"[{epoch}/{self.config.epochs}] - loss: {meanBatchLoss}, time: {time.time() - epochStartTime}")
            if epoch > 0 and self.config.modelSaveEpoch > 0 and epoch % self.config.modelSaveEpoch == 0:
                torch.save(self.model.state_dict(), self.config.outDir / Path(f'epoch_{epoch}.pth'))
                self.infoLogger.info(f'Epoch {epoch}: model saved')
            if self.config.validationEpoch > 0 and epoch % self.config.validationEpoch == 0:
                valLoss, cer, wer, results = self.__evaluate()
                self.infoLogger.info(f"Validation - loss: {valLoss}, CER: {cer * 100:0.2f}, WER: {wer * 100:0.2f}")
                validationLogger.info(f"{epoch},{valLoss},{cer},{wer}")
                if cer < self.bestValidationCER:
                    self.bestValidationCER = cer
                    self.bestValidationEpoch = epoch
                    torch.save(self.model.state_dict(), self.config.outDir / Path('best_val.pth'))
                    with (self.config.outDir / "best_val_results.json").open("w") as outFile:
                        json.dump(results, outFile, ensure_ascii=False, indent=4)
                    self.infoLogger.info(f'Epoch {epoch}: best model checkpoint updated')
            if self.config.earlyStoppingEpochCount > 0 and epoch > self.config.warmup:
                if epoch - self.bestValidationEpoch >= self.config.earlyStoppingEpochCount:
                    self.infoLogger.info(
                        f'No validation CER improvement in {epoch - self.bestValidationEpoch} epochs, stopping training.')
                    break

        self.infoLogger.info(
            f"Best validation CER: {self.bestValidationCER * 100:0.2f} (epoch {self.bestValidationEpoch})")

    def __decode(self, predicted) -> List[str]:
        decodedTranscriptions = []
        _, max_index = torch.max(predicted, dim=2)
        for i in range(predicted.shape[1]):
            raw_prediction = list(max_index[:, i].detach().cpu().numpy())

            previous = raw_prediction[0]
            output = [previous]
            for char in raw_prediction[1:]:
                if char == output[-1]:
                    continue
                else:
                    output.append(char)

            result = self.transcriptionEncoder.decode(output)
            decodedTranscriptions.append(result)
        return decodedTranscriptions


    def __evaluate(self) -> Tuple[float, float, float, List[Dict[str, str]]]:
        batchLosses = []
        predictedTranscriptions = []
        expectedTranscriptions = []
        filenames = []

        self.model.eval()
        with torch.no_grad():
            for batchId, data in enumerate(self.evalDataloader):
                lineImage = data["image"].to(self.config.device)
                encodedTranscription = data["transcription"].to(self.config.device)
                plaintextTranscription = data["plaintext"]
                filenames.extend(data["imageName"])

                rawPrediction = self.model(lineImage)
                predicted = rawPrediction.log_softmax(2)

                input_lengths = torch.full(size=(predicted.shape[1],), fill_value=predicted.shape[0], dtype=torch.long)
                target_lengths = data["t_len"]

                loss = CTCLoss(zero_infinity=True)(predicted, encodedTranscription, input_lengths, target_lengths)
                batchLosses.append(loss.item())

                predictedText = self.__decode(rawPrediction)
                predictedTranscriptions.extend(predictedText)
                expectedTranscriptions.extend(plaintextTranscription)

            cer = self.cerMetric(predictedTranscriptions, expectedTranscriptions)
            wer = self.werMetric(predictedTranscriptions, expectedTranscriptions)

            meanBatchLoss = np.mean(batchLosses)
            results = [{"line": file, "transcription": pred, "groundTruth": gt} for (file, pred, gt) in
                       zip(filenames, predictedTranscriptions, expectedTranscriptions)]
        return meanBatchLoss, cer, wer, results

    def test(self):

        self.infoLogger.info("Testing ...")
        evalDataset = LineDataset(self.config, DatasetMode.TEST, self.transcriptionEncoder)
        self.evalDataloader = DataLoader(evalDataset, batch_size=self.config.batchSize, shuffle=False,
                                         num_workers=self.config.workerCount)
        self.__loadModelCheckpoint()
        _, cer, wer, results = self.__evaluate()
        with (self.config.outDir / f"test_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json").open(
                "w") as outFile:
            json.dump(results, outFile, ensure_ascii=False, indent=4)
        self.infoLogger.info(f"Test performance: CER: {cer * 100:0.2f}, WER: {wer * 100:0.2f}")

    def infer(self):
        self.infoLogger.info("Preparing inference.")
        evalDataset = LineDataset(self.config, DatasetMode.INFER, self.transcriptionEncoder)
        inferenceDataLoader = DataLoader(evalDataset, batch_size=self.config.batchSize, shuffle=False,
                                         num_workers=self.config.workerCount)
        self.__loadModelCheckpoint()
        self.model.eval()
        results = []
        self.infoLogger.info("Inference in progress")
        with torch.no_grad():
            for batchId, data in enumerate(inferenceDataLoader):
                lineName = data["imageName"]
                lineImage = data["image"].to(self.config.device)
                rawPrediction = self.model(lineImage)
                predictedText = self.__decode(rawPrediction)
                for (line, transcription) in zip(lineName, predictedText):
                    results.append({"line": line, "transcription": transcription})

        filename = f"inference_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        with (self.config.outDir / filename).open("w") as outFile:
            json.dump(results, outFile, ensure_ascii=False, indent=4)
        self.infoLogger.info(f"Inference complete, output in {self.config.outDir / filename}")

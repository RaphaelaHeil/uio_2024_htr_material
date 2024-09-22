import logging
from pathlib import Path


def initInfoLogger(outDir: Path, loggerName: str = "info"):
    logger = logging.getLogger(loggerName)
    logger.setLevel(logging.INFO)

    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])

    formatter = logging.Formatter('%(asctime)s - %(message)s', '%d-%b-%y %H:%M:%S')

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

    fileHandler = logging.FileHandler(outDir / f"{loggerName}.log", mode="a")
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)


def initProgressLogger(outDir: Path, loggerName: str = "progress"):
    auxLogger = logging.getLogger(loggerName)
    while auxLogger.hasHandlers():
        auxLogger.removeHandler(auxLogger.handlers[0])

    auxLogger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler(outDir / f"{loggerName}.csv", mode="w")
    fileHandler.setLevel(logging.INFO)
    auxLogger.addHandler(fileHandler)

from argparse import ArgumentParser
from pathlib import Path

from htr.runner import Runner
from htr.utils.config import getConfiguration


def main():
    argParser = ArgumentParser()
    argParser.add_argument("-m", "--mode",
                           help="run mode - one of: train, test, train+test, finetune, finetune+test, infer",
                           default="train", type=str)
    argParser.add_argument("-c", "--config", help="path to config-file", default="config.cfg", type=Path)
    argParser.add_argument("-s", "--section", help="section of config-file to use", default="DEFAULT")
    argParser.add_argument("-e", "--eval", help="Path to inference data, ignored for all other modes", type=Path)
    args = argParser.parse_args()

    config = getConfiguration(args)
    runner = Runner(config)

    if args.mode.startswith("train"):
        runner.train()
    if args.mode.startswith("finetune"):
        runner.finetune()
    if "test" in args.mode:
        runner.test()
    if args.mode == "infer":
        runner.infer()


if __name__ == '__main__':
    main()

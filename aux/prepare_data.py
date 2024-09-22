import csv
import random
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from math import floor
from pathlib import Path
from typing import List, Tuple

import numpy as np
from skimage.io import imread, imsave


def parseAltoXml(filePath: Path) -> List[Tuple[slice, str]]:
    results = []
    root = ET.parse(filePath).getroot()
    rootTag = root.tag
    if "alto" in rootTag:
        namespace = {"": rootTag.split("}")[0].strip("{")}
    else:
        namespace = {"": "http://www.loc.gov/standards/alto/ns-v3#"}

    pages = root.findall(".//Page", namespace)
    if len(pages) == 0:
        raise ValueError("no page found")
    if len(pages) > 1:
        raise ValueError("too many pages")

    textLines = root.findall(".//TextLine", namespace)

    if textLines:
        for tL in textLines:
            y0 = int(tL.attrib["VPOS"])
            x0 = int(tL.attrib["HPOS"])
            y1 = y0 + int(tL.attrib["HEIGHT"])
            x1 = x0 + int(tL.attrib["WIDTH"])

            content = []
            words = tL.findall("String", namespace)
            for entry in words:
                content.append(entry.attrib["CONTENT"])
            transcription = " ".join(content)
            results.append((np.s_[y0:y1, x0:x1], transcription))

    return results


def parsePageXml(filePath: Path) -> List[Tuple[slice, str]]:
    results = []
    root = ET.parse(filePath).getroot()
    rootTag = root.tag
    if "PcGts" in rootTag:
        namespace = {"": rootTag.replace("PcGts", "").strip("{}")}
    else:
        namespace = {"": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
    pages = root.findall("Page", namespace)
    if len(pages) == 0:
        raise ValueError("no page found")
    if len(pages) > 1:
        raise ValueError("too many pages")

    for line in root.findall('.//TextLine', namespace):
        x = []
        y = []
        for p in line.find("Coords", namespace).attrib["points"].split():
            s = p.split(",")
            y.append(int(s[1]))
            x.append(int(s[0]))
        x0 = min(x)
        x1 = max(x)
        y0 = min(y)
        y1 = max(y)

        unicode = line.find(".//Unicode", namespace)
        text = unicode.text
        if text:
            results.append((np.s_[y0:y1, x0:x1, :], text))
    return results


def main():
    argParser = ArgumentParser()
    argParser.add_argument("-i", "--images", help="image directory", type=Path, required=True)
    group = argParser.add_mutually_exclusive_group(required=True)
    group.add_argument("-a", "--alto", help="alto xml directory", type=Path)
    group.add_argument("-p", "--page", help="page xml directory", type=Path)
    argParser.add_argument("-o", "--out", help="output directory", type=Path, required=True)
    argParser.add_argument("-s", "--splits",
                           help="[train validation test] portions in integer percentages. Must to sum to 100 - default: 75 10 15",
                           nargs=3, default=["75", "10", "15"])
    args = argParser.parse_args()

    train = int(args.splits[0])
    val = int(args.splits[1])
    test = int(args.splits[2])

    if (train + val + test) != 100:
        raise ValueError(f"Data splits have to sum to 100 but are {train + val + test}")

    outputDir = args.out

    imageBase = args.images
    if not imageBase.exists():
        raise ValueError(f"Image directory does not exist ({imageBase})")

    if args.alto:
        parseXml = parseAltoXml
        xmlBase = args.alto
    else:
        parseXml = parsePageXml
        xmlBase = args.page

    if not xmlBase.exists():
        raise ValueError(f"XML directory does not exist ({xmlBase})")

    xmlNames = [filename.stem for filename in xmlBase.glob("*.xml")]
    imageNames = [filename.name for filename in imageBase.glob("*.[jp][pn]g") if filename.stem in xmlNames]

    random.shuffle(imageNames)

    count = len(imageNames)

    valPortion = val / 100
    valCount = floor(count * valPortion)

    testPortion = test / 100
    testCount = floor(count * testPortion)
    trainCount = count - valCount - testCount

    offset = 0
    for split, count in [("train", trainCount), ("validation", valCount), ("test", testCount)]:
        print(split)
        print(count, "pages")
        splitOut = outputDir / split
        splitOut.mkdir(exist_ok=True, parents=True)

        splitIndex = []
        lineCounter = 0
        for pageName in imageNames[offset:offset + count]:
            xmlName = (xmlBase / pageName).with_suffix(".xml")
            image = imread(imageBase / pageName)
            lines = parseXml(xmlName)
            for idx, (slice, transcription) in enumerate(lines):
                outName = f"{pageName[:-4]}_{idx:03}.png"
                lineImage = image[slice]
                imsave(splitOut / outName, lineImage)
                splitIndex.append([outName, transcription])
                lineCounter += 1

        with (splitOut / "index.tsv").open("w") as outFile:
            csvWriter = csv.writer(outFile, delimiter='\t')
            csvWriter.writerows(splitIndex)
        offset += count
        print(lineCounter, "line images")
        print("---------------")


if __name__ == '__main__':
    main()

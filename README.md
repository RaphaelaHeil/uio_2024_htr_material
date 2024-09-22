# Code for 'Getting Started with Handwritten Text Recognition in PyTorch'

Workshop held at the University of Oslo, September 2024. More details available on the [workshop website](https://raphaelaheil.github.io/2024-09-24-uio-htr/). 


## Installation

1. Install the PyTorch version that matches your local set-up (GPU/no GPU, cuda version) - see the [PyTorch Website](https://pytorch.org/get-started/locally/) for available options (torch and torchvision are enough, torchaudio is not needed)
2. Install `scikit-image` and `torchmetrics`


## Data Preparation

The repository contains a script to prepare a dataset with a matching structure for you, assuming you have images and corresponding **PAGE** or **ALTO** XML files. Each image and its transcription file have to share the same name, differing only in the extension. 

### Running the script

Page XML:
```
python -m aux.prepare_data --images <path_to_images> -p <path_to_page_xml_files>  -o <path_to_output_dir> 
```

ALTO XML: 
```
python -m aux.prepare_data --images <path_to_images> -p <path_to_page_xml_files>  -o <path_to_output_dir> 
```

The script will automatically split the data at a page level into 75% training, 10% validation and 15% test images. Other splitting proportions can be defined via the option `-s <train> <validation> <test>`, e.g. `-s 50 25 25`. Note that the splitting has to sum to 100%! 


## Running HTR

Modify the file `config.cfg` to match your desired configuration. The start the training via: 

```
python main.py -m train -c <path_to_config>
```

- if your configuration file contains several sections and you **do not** want to use the "DEFAULT" one, specify the named section via the option `-s <section_name>`
	- in order to automatically run the testing after the training completes, use the mode `-m train+test`
- to test an already trained model, use the mode `-m test` and give the path to the configuration file in the trained model's output directory, i.e. `-c <model_output_dir>/config.cfg`
- to finetune a previously trained model, use the mode `-m finetune` and give the path to the configuration file in the trained model's output directory, i.e. `-c <model_output_dir>/config.cfg`
	- in order to automatically run the testing after the finetuning completes, use the mode `-m finetune+test`
- to use a previously trained model for inference, use the mode `-m infer`, give the path to the configuration file in the trained model's output directory, i.e. `-c <model_output_dir>/config.cfg`, and specify the path to the inference dataset via `-e <path_to_data>`
	- note that this data has to be segmented into individual lines but does not require transcriptions


## Exploring Image Augmentations

This repository also contains a script to explore the impact of image augmentations on handwritten images. It can be run via: 

```
python -m demo.aug_demo
```






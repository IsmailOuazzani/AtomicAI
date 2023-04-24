# Instructions
## Dataset
Install ztsd to unzip the dataset: 

https://github.com/facebook/zstd/releases
Download dataset into the dataset folder from: https://database.lichess.org/#variant_games

Unzip the dataset:
```bash
ztsd -d (filaname.pgn.zst) -o (filename.pgn)
```

Install python-chess:
```bash
pip install python-chess
```

The data can be read and processed using utils/data_processing.py

## Installation

Install CUDA 11.7: https://developer.nvidia.com/cuda-11-7-0-download-archive

Download CUDNN for CUDA 11.7, extract it somewhere and add the bin folder to environment variables

Install PyTorch from: https://pytorch.org/get-started/locally/ (don't forget to select the right options)

# Features

## Heuristic function
Model can be trained using the atomicdeep.py file. It can also resume the training of an existing model by loading it (uncomment relevant sections of the code). To train optimally, install PyTorch with CUDA.

## Engine
Run ui.py inside the engine folder to play against the engine. The engine was adapted from https://github.com/healeycodes/andoma by replacing the evalution function with our neural network heuristic, and by adapting the endgame to take atomic chess wins into account.

By default, the engine will play against the baseline. To play against the model, set human=False in ui.py

To change the model used, modify the chessnet2.py file. The game can be visualized through the board.svg file that the engine generates in the engine folder after every turn.


# Report
Our project report is included in the AtomicAI_report.pdf document (root directory).
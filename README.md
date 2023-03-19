# Notes
This project is still in the works.

# Dataset
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

The data can be read using utils/data_processing.py

# Neural net
See atomicdeep.py. Note that you need to install PyTorch with CUDA to run it optimally.

We are using the evaluation function on atomicdeep.py to evaluate the board from the current player's perspective.

The model is currently able to overfit on a small part of the dataset dataset (1 000 000 boards).

Latest test: ran on 1 000 000 boards from games where both players had elo greater than 1500, with 5000 games per file, batch size 1000, learning rate 0.01 and on 20 epochs. It took about 1h40mins to run the entire atomicdeep.py code.

| ![err](model_error.png) | ![loss](model_loss.png) | ![acc](model_acc.png) |
|-----------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
<div id="b" align="center">
<h5>Metrics of latest test</h5>
</div>

## Installation

Install CUDA 11.7: https://developer.nvidia.com/cuda-11-7-0-download-archive

Download CUDNN for CUDA 11.7, extract it somewhere and add the bin folder to environment variables

Install PyTorch from: https://pytorch.org/get-started/locally/ (don't forget to select the right options)

## Ideas:
- Use 3D convolutions to capture the relationships between pieces
- Experiment with both 3x3 and 7x7 convolutions in the first layer to capture piece surroundings and long range relationships
- Use PyTorch's profiling option to see how the resources are used

# To-do
Export neural net to ONNX and integrate it to fairy stockfish (which supports Atomic Chess).


https://github.com/fairy-stockfish/Fairy-Stockfish/wiki/Understanding-the-code. Here, need to modify this file:
https://github.com/fairy-stockfish/Fairy-Stockfish/blob/master/src/evaluate.cpp at line 1558.

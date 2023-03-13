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

# Engine
Engine allows user to play against the AI. It uses Monte Carlo Tree Search. For now, the engine  has a dummy evaluation function based on the material of the pieces, and uses random moves for the simulation.

To play, run engine.py and enter the moves using the algebraic notation. You can visualize the game in the board.svg file using your browser (keep refreshing).

Note that the engine is really bad.

# Neural net
We are using the evaluation function on atomicdeep.py to evaluate the board from the current player's perspective.

The model is currently able to overfit on a small part of the dataset(3000 boards).
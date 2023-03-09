# Dataset
Install ztsd to unzip the dataset: 

https://github.com/facebook/zstd/releases

Download dataset into the dataset folder from: https://database.lichess.org/#variant_games

Unzip the dataset:
```bash
ztsd -d (filaname.pgn.zst) -o (filename.pgn)
```

The data can be read using utils/data_processing.py


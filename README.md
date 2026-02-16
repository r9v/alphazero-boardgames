# AlphaZero for Board Games

An implementation of the [AlphaZero](https://arxiv.org/abs/1712.01815) algorithm applied to classic board games — Connect 4, Tic-Tac-Toe, and Hive.

Based on the paper: **"Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"** by David Silver, Thomas Hubert, Julian Schrittwieser et al. (DeepMind), [arXiv:1712.01815](https://arxiv.org/abs/1712.01815) [[PDF]](https://arxiv.org/pdf/1712.01815)

## Overview

AlphaZero learns to play board games entirely from self-play, with no human knowledge beyond the rules of the game. This project reproduces the core components of that system:

- **Monte Carlo Tree Search (MCTS)** guided by a neural network
- **Dual-head neural network** outputting both a move policy and a position value
- **Self-play training loop** that generates training data from games the agent plays against itself

## Supported Games

| Game            | Board        | Status                                  |
| --------------- | ------------ | --------------------------------------- |
| **Tic-Tac-Toe** | 3x3          | Fully integrated with training pipeline |
| **Connect 4**   | 6x7          | Game logic implemented                  |
| **Hive**        | 25x25x5 (3D) | Game logic + GUI implemented            |

## Architecture

The neural network follows the AlphaZero design:

```
Input (board state + history)
        |
  Conv2D + BatchNorm + ReLU
        |
  Residual Block x2
   /              \
Policy Head     Value Head
 (softmax)       (tanh)
```

- **Input**: Current board + 2 previous board states + player indicator (8 channels for TTT)
- **Backbone**: 1 convolutional layer (256 filters) followed by 2 residual blocks
- **Policy head**: Outputs a probability distribution over legal moves
- **Value head**: Outputs a scalar in [-1, 1] estimating the winning probability

## MCTS

The search uses PUCT (Predictor + Upper Confidence bound for Trees):

```
a* = argmax [ Q(s,a) + P(s,a) * sqrt(N(s)) / (1 + N(s,a)) ]
```

- Neural network priors guide exploration
- Dirichlet noise added at the root for exploration during training (alpha=0.03, epsilon=0.25)
- 50 simulations per move

## Project Structure

```
├── MCTS.py                 # Monte Carlo Tree Search
├── TrainingData.py         # Ring buffer for training samples
├── trainTTT.py             # Self-play training loop
├── Connect4Game/
│   └── Connect4Game.py     # Connect 4 game logic
├── TTT/
│   ├── TTT.py              # Tic-Tac-Toe game logic
│   └── Net.py              # Neural network (TensorFlow)
├── TTTNet/
│   ├── TicTacToeNNet.py    # Alternative network (Keras)
│   └── NNetWrapper.py      # Network inference wrapper
└── Hive/
    ├── Hive.py             # Hive game logic
    ├── Net.py              # Hive neural network
    ├── GUI.py              # Graphical interface
    └── const.py            # Game constants
```

## References

- Silver, D., Hubert, T., Schrittwieser, J., et al. (2017). _Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm_. [arXiv:1712.01815](https://arxiv.org/abs/1712.01815)
- Silver, D., Schrittwieser, J., Simonyan, K., et al. (2017). _Mastering the game of Go without human knowledge_. Nature, 550, 354-359.

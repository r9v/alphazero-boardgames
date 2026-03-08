# AlphaZero for Board Games

An implementation of the [AlphaZero](https://arxiv.org/abs/1712.01815) algorithm applied to classic board games — Tic-Tac-Toe, Connect 4, and Hive.

Originally implemented in 2020. Restructured and modernized in 2026 (PyTorch migration, unified game interface).

Based on: **"Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"** by Silver et al. (DeepMind), [arXiv:1712.01815](https://arxiv.org/abs/1712.01815)

## Overview

AlphaZero learns to play board games entirely from self-play, with no human knowledge beyond the rules of the game. This project reproduces the core components of that system:

- **Monte Carlo Tree Search (MCTS)** guided by a neural network
- **Dual-head neural network** outputting both a move policy and a position value
- **Self-play training loop** that generates training data from games the agent plays against itself

## Quick Start

```bash
pip install -r requirements.txt
```

**Train:**

```bash
python train.py --game tictactoe --simulations 25 --games 20 --iterations 10 --filters 64
```

**Play against the AI:**

```bash
python play.py --game tictactoe --human-first
```

## Supported Games

| Game            | Board        | Status                                 |
| --------------- | ------------ | -------------------------------------- |
| **Tic-Tac-Toe** | 3x3          | Fully trained, playable                |
| **Connect 4**   | 6x7          | Fully trained, playable                |
| **Hive**        | 25x25x5 (3D) | Game logic + GUI (action encoding TBD) |

## Architecture

The neural network follows the AlphaZero design:

- **Input**: Current board + 2 previous board states + player indicator (8 channels for TTT)
- **Backbone**: 1 convolutional layer (256 filters) followed by 2 residual blocks
- **Policy head**: Outputs a probability distribution over legal moves
- **Value head**: Outputs a scalar in [-1, 1] estimating the winning probability

```
Input (board state + 2 previous states + player indicator)
        │
  Conv2D + BatchNorm + ReLU
        │
  Residual Block × N
     ┌──┴──┐
  Policy  Value
  Head    Head
(softmax) (tanh)
```

All games share the same configurable PyTorch `AlphaZeroNet` — only the input channels, board shape, and action size change per game.

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
games/
  base.py              # Abstract Game / GameState interface
  tictactoe.py         # Tic-Tac-Toe
  connect4.py          # Connect 4
  hive/                # Hive (game + GUI + pieces)
network/
  alphazero_net.py     # Configurable dual-head ResNet (PyTorch)
mcts/
  mcts.py              # Monte Carlo Tree Search
training/
  trainer.py           # Self-play + network training loop
  replay_buffer.py     # Ring buffer for training samples
train.py               # CLI: train any game
play.py                # CLI: play against trained model
```

## References

- Silver, D., Hubert, T., Schrittwieser, J., et al. (2017). _Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm_. [arXiv:1712.01815](https://arxiv.org/abs/1712.01815)
- Silver, D., Schrittwieser, J., Simonyan, K., et al. (2017). _Mastering the game of Go without human knowledge_. Nature, 550, 354-359.

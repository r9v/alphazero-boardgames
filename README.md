# AlphaZero for Board Games

An implementation of the [AlphaZero](https://arxiv.org/abs/1712.01815) algorithm applied to classic board games — Tic-Tac-Toe, Connect 4, and Santorini.

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
python setup.py build_ext --inplace
```

**Play against the AI** (pretrained models included in `checkpoints/`):

```bash
python play.py --game tictactoe --human-first
python play.py --game connect4 --human-first
python play.py --game santorini --human-first
```

**Train from scratch:**

```bash
python train.py --game tictactoe
python train.py --game connect4
python train.py --game santorini
```

All training parameters (iterations, games per iteration, simulations, network size) have per-game defaults in `game_configs.py` and can be overridden via CLI:

```bash
python -u train.py --game santorini --iterations 64 --games 64 --simulations 256
python -u train.py --game santorini --iterations 128 2>&1 | tee training_log_santorini.txt
python -u train.py --game tictactoe 2>&1 | tee training_log_ttt.txt
python -u train.py --game connect4 2>&1 | tee training_log_c4.txt

```

**Monitor training:**

```bash
tensorboard --logdir runs/
```

## Supported Games

| Game            | Board | Action Space | Status                  |
| --------------- | ----- | ------------ | ----------------------- |
| **Tic-Tac-Toe** | 3×3   | 9            | Fully trained, playable |
| **Connect 4**   | 6×7   | 7            | Fully trained, playable |
| **Santorini**   | 5×5   | 128          | Fully trained, playable |

## Architecture

The neural network follows the AlphaZero design:

- **Input**: 2 channels — current player's pieces and opponent's pieces (relative encoding)
- **Backbone**: 1 convolutional layer + N residual blocks (size varies per game)
- **Policy head**: Outputs a probability distribution over legal moves
- **Value head**: Outputs a scalar in [-1, 1] estimating the winning probability

```
Input (2 channels: my pieces, opponent pieces)
        │
  Conv2D + BatchNorm + ReLU
        │
  Residual Block x N
   ┌────┴────┐
 Policy    Value
  Head      Head
   |         |
 Conv 1x1  Conv 1x1
 BN+ReLU   BN+LeakyReLU
 Linear    Linear+LeakyReLU
 Softmax   Dropout+Linear+Tanh
```

All games share the same configurable PyTorch `AlphaZeroNet` — only the input channels, board shape, and action size change per game.

## MCTS

The search uses PUCT (Predictor + Upper Confidence bound for Trees):

```
a* = argmax [ Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a)) ]
```

- Neural network priors guide exploration
- Dirichlet noise at root for training exploration (alpha=0.03, epsilon=0.25)
- Virtual loss for parallel tree traversal (Santorini uses vl=3.0 with 8 selects/round)
- Temperature annealing: proportional play early, greedy after `temp_threshold` moves

MCTS and Santorini game logic are implemented in Cython for performance. The Cython extensions must be compiled before use (`python setup.py build_ext --inplace`).

## Tests

```bash
python -m tests.test_connect4
python -m tests.test_mcts
python -m tests.test_santorini_placement
python -m tests.test_santorini_symmetry
```

## References

- Silver, D., Hubert, T., Schrittwieser, J., et al. (2017). _Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm_. [arXiv:1712.01815](https://arxiv.org/abs/1712.01815)
- Silver, D., Schrittwieser, J., Simonyan, K., et al. (2017). _Mastering the game of Go without human knowledge_. Nature, 550, 354-359.

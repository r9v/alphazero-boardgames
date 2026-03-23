# AlphaZero for Board Games

An implementation of the [AlphaZero](https://arxiv.org/abs/1712.01815) algorithm applied to board games — Tic-Tac-Toe, Connect 4, and Santorini.

Originally implemented in 2020. Restructured and modernized in 2026 (PyTorch migration, unified game interface, KataGo-inspired training improvements).

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
python -u train.py --game connect4 2>&1 | tee training_log_c4.txt
python -u train.py --game santorini --iterations 128 2>&1 | tee training_log_santorini.txt
```

All training parameters have per-game defaults in `game_configs.py` and can be overridden via CLI:

```bash
python -u train.py --game santorini --iterations 64 --games 64 --simulations 256
```

**Monitor training:**

```bash
tensorboard --logdir runs/
```

## Supported Games

| Game            | Board | Action Space |
| --------------- | ----- | ------------ |
| **Tic-Tac-Toe** | 3x3   | 9            |
| **Connect 4**   | 6x7   | 7            |
| **Santorini**   | 5x5   | 128          |

## Architecture

```
Input (2ch: my pieces, opponent pieces)
        |
  WS-Conv2D + GroupNorm + ReLU
        |
  Pre-activation ResBlock x N (with Weight Standardization)
   +--------+
 Policy    Value
  Head      Head
   |         |
 Conv 1x1  Conv 1x1 + GAP
 GN+ReLU   GN+LeakyReLU
 Linear    FC + Dropout
 Softmax   WDL logits (Win/Draw/Loss)
```

- **Weight Standardization** on all conv layers — normalizes weights per filter, prevents weight explosion in non-stationary RL training
- **Batched parallel self-play** — all games evaluate simultaneously on GPU in a single forward pass

## Training Improvements

Beyond standard AlphaZero, this implementation includes several [KataGo](https://arxiv.org/abs/1902.10565)-inspired improvements to address the self-play data distribution bias:

- **Policy Surprise Weighting** — positions where MCTS disagrees with the network's prior (high KL divergence) are oversampled during training. Ensures the network trains more on positions it gets wrong.

- **Search-Contempt** — at opponent nodes in the MCTS tree, after sufficient visits, switches from PUCT to Thompson sampling. Forces the opponent to occasionally play unexpected moves, diversifying the positions reached during self-play.

- **Policy Target Pruning** — after MCTS, subtracts Dirichlet noise and exploration-induced visits from non-best children before constructing the policy training target. Prevents the network from learning to predict exploration artifacts.

## MCTS

PUCT action selection with Cython-accelerated tree operations:

```
a* = argmax [ Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a)) ]
```

- Dirichlet noise at root for exploration
- Virtual loss for parallel tree traversal
- Temperature annealing: proportional play early, greedy after `temp_threshold` moves
- Tree reuse between moves

MCTS, Connect 4, and Santorini game logic are all implemented in Cython for performance.

## Tournament

Battle all saved checkpoints in a single-elimination tournament:

```bash
python battle/tournament.py --game connect4 --sims 50 --games 50 --parallel 50
```

Loads all `.pt` checkpoints from `checkpoints/<game>/`, pairs them chronologically, and plays elimination matches. Each match alternates who goes first.

## Tests

```bash
python -m tests.test_connect4
python -m tests.test_mcts
python -m tests.test_santorini_placement
python -m tests.test_santorini_symmetry
```

## References

- Silver, D., et al. (2017). _Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm_. [arXiv:1712.01815](https://arxiv.org/abs/1712.01815)
- Silver, D., et al. (2017). _Mastering the game of Go without human knowledge_. Nature, 550, 354-359.
- Wu, D. J. (2019). _Accelerating Self-Play Learning in Go_. [arXiv:1902.10565](https://arxiv.org/abs/1902.10565)
- KataGo methods and training improvements: [github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)
- Löwisch, M. & Wiering, M. (2020). _Reducing the Variance of AlphaZero_. Adaptive and Learning Agents Workshop (ALA), AAMAS 2020.

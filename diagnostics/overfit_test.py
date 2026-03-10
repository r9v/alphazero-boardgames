"""Diagnostic B: Overfit test.

Tests whether the value head CAN fit a small set of training examples.
If it can't even overfit 50 examples, capacity is the bottleneck.
If it can, the problem is training dynamics or data distribution.

Run: python -m diagnostics.overfit_test [--game connect4]
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from network import AlphaZeroNet
from game_configs import GAME_CONFIGS
from train import load_game
from games.connect4 import GameState as C4State


def make_c4_position(board_array, player):
    """Create a Connect4 state from a 2D array."""
    return C4State(None, np.array(board_array, dtype="int"), player)


def generate_test_examples(game):
    """Generate small set of positions with KNOWN correct values."""
    examples = []

    # --- Positions where X is clearly winning (target = +1 from X perspective) ---
    # X about to win horizontally
    board = np.zeros((6, 7), dtype="int")
    board[0][0:3] = -1
    board[0][4] = 1
    s = make_c4_position(board, player=-1)
    examples.append((game.state_to_input(s), 1.0, "X_wins_horiz"))

    # X about to win vertically
    board = np.zeros((6, 7), dtype="int")
    board[0][3] = -1; board[1][3] = -1; board[2][3] = -1
    board[0][2] = 1; board[0][4] = 1
    s = make_c4_position(board, player=-1)
    examples.append((game.state_to_input(s), 1.0, "X_wins_vert"))

    # --- Positions where O is clearly winning (target = +1 from O perspective) ---
    # O about to win horizontally
    board = np.zeros((6, 7), dtype="int")
    board[0][0:3] = 1
    board[0][4] = -1; board[1][4] = -1
    s = make_c4_position(board, player=1)
    examples.append((game.state_to_input(s), 1.0, "O_wins_horiz"))

    # O about to win vertically
    board = np.zeros((6, 7), dtype="int")
    board[0][3] = 1; board[1][3] = 1; board[2][3] = 1
    board[0][2] = -1; board[0][4] = -1; board[1][2] = -1
    s = make_c4_position(board, player=1)
    examples.append((game.state_to_input(s), 1.0, "O_wins_vert"))

    # --- Positions where current player is losing (target = -1) ---
    # X to move but O threatens (target = -1 from X perspective)
    board = np.zeros((6, 7), dtype="int")
    board[0][0:3] = 1  # O has 3 in a row
    board[0][4] = -1; board[1][4] = -1
    s = make_c4_position(board, player=-1)
    examples.append((game.state_to_input(s), -1.0, "X_losing"))

    # O to move but X threatens (target = -1 from O perspective)
    board = np.zeros((6, 7), dtype="int")
    board[0][0:3] = -1
    board[0][4] = 1
    s = make_c4_position(board, player=1)
    examples.append((game.state_to_input(s), -1.0, "O_losing"))

    # --- Neutral positions (target ~0) ---
    board = np.zeros((6, 7), dtype="int")
    s = make_c4_position(board, player=-1)
    examples.append((game.state_to_input(s), 0.0, "empty_X"))

    board = np.zeros((6, 7), dtype="int")
    board[0][3] = -1
    s = make_c4_position(board, player=1)
    examples.append((game.state_to_input(s), 0.0, "one_move_O"))

    # --- Also generate random game positions for variety ---
    rng = np.random.RandomState(42)
    for i in range(42):  # total ~50 examples
        s = game.new_game()
        moves = rng.randint(1, 20)
        for _ in range(moves):
            if s.terminal:
                break
            avail = np.nonzero(s.available_actions)[0]
            s = game.step(s, rng.choice(avail))
        if s.terminal:
            target = s.terminal_value * s.player  # would be wrong player but doesn't matter
            # For terminal, use actual value
            # go back one move
            continue
        # Assign random target for non-terminal
        target = rng.choice([-1.0, 0.0, 1.0])
        examples.append((game.state_to_input(s), target, f"random_{i}"))

    return examples


def run_overfit_test(game, num_steps=2000, lr=0.001):
    """Train on a small fixed set and check if value head can overfit."""
    examples = generate_test_examples(game)
    print(f"Generated {len(examples)} test examples")
    print(f"  Targets: {[f'{e[2]}={e[1]:+.1f}' for e in examples[:8]]}")

    game_cfg = GAME_CONFIGS.get("connect4", {})
    input_channels = 2 * (game.num_history_states + 1)
    net = AlphaZeroNet(
        input_channels=input_channels,
        board_shape=game.board_shape,
        action_size=game.action_size,
        num_res_blocks=game_cfg.get("num_res_blocks", 3),
        num_filters=game_cfg.get("num_filters", 128),
        value_head_channels=game_cfg.get("value_head_channels", 2),
        value_head_fc_size=game_cfg.get("value_head_fc_size", 64),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

    # Prepare tensors
    states = torch.FloatTensor(np.array([e[0] for e in examples])).to(device)
    targets = torch.FloatTensor([e[1] for e in examples]).unsqueeze(1).to(device)
    names = [e[2] for e in examples]

    print(f"\nTraining on {len(examples)} examples for {num_steps} steps (lr={lr})...")
    print(f"{'Step':>6}  {'VLoss':>8}  {'X_win_h':>8}  {'O_win_h':>8}  {'X_losing':>8}  {'O_losing':>8}  {'empty':>8}")
    print("-" * 70)

    for step in range(num_steps):
        net.train()
        pred_vs, pred_pis = net(states)
        value_loss = F.mse_loss(pred_vs, targets)
        # Don't need policy loss for this test
        optimizer.zero_grad()
        value_loss.backward()
        optimizer.step()

        if step % 200 == 0 or step == num_steps - 1:
            net.eval()
            with torch.no_grad():
                pv, _ = net(states)
                preds = pv.squeeze(1).cpu().numpy()

            # Show key examples
            named_preds = {}
            for nm, pred in zip(names, preds):
                named_preds[nm] = pred

            print(f"{step:>6}  {value_loss.item():>8.4f}  "
                  f"{named_preds.get('X_wins_horiz', 0):>+8.4f}  "
                  f"{named_preds.get('O_wins_horiz', 0):>+8.4f}  "
                  f"{named_preds.get('X_losing', 0):>+8.4f}  "
                  f"{named_preds.get('O_losing', 0):>+8.4f}  "
                  f"{named_preds.get('empty_X', 0):>+8.4f}")

    # Final detailed report
    print(f"\n{'='*70}")
    print("FINAL PREDICTIONS vs TARGETS:")
    print(f"{'Name':<20} {'Target':>8} {'Pred':>8} {'Error':>8} {'Correct?':>8}")
    print("-" * 60)

    net.eval()
    with torch.no_grad():
        pv, _ = net(states)
        preds = pv.squeeze(1).cpu().numpy()

    correct = 0
    total = len(examples)
    for nm, tgt, pred in zip(names, [e[1] for e in examples], preds):
        err = abs(pred - tgt)
        sign_ok = (tgt == 0) or (np.sign(pred) == np.sign(tgt))
        correct += int(sign_ok)
        marker = "OK" if sign_ok else "WRONG"
        if nm.startswith("random_") and int(nm.split("_")[1]) > 5:
            continue  # skip most random ones
        print(f"{nm:<20} {tgt:>+8.2f} {pred:>+8.4f} {err:>8.4f} {marker:>8}")

    final_vloss = F.mse_loss(torch.FloatTensor(preds).unsqueeze(1),
                              torch.FloatTensor([e[1] for e in examples]).unsqueeze(1)).item()
    print(f"\nFinal value loss: {final_vloss:.6f}")
    print(f"Sign accuracy: {correct}/{total} ({correct/total:.1%})")

    if final_vloss < 0.01:
        print("\nVERDICT: Value head CAN fit small data. Capacity is NOT the bottleneck.")
        print("Problem is likely training dynamics, data distribution, or gradient interference.")
    elif final_vloss < 0.1:
        print("\nVERDICT: Value head partially fits. May be mild capacity issue.")
    else:
        print("\nVERDICT: Value head CANNOT fit even 50 examples. CAPACITY IS THE BOTTLENECK.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="connect4")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    game = load_game(args.game)
    run_overfit_test(game, num_steps=args.steps, lr=args.lr)

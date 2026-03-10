"""Diagnostic C: Sign sanity check.

Verifies the entire training pipeline has correct sign conventions:
1. Creates positions with KNOWN values
2. Does one forward pass to get predictions
3. Computes loss and gradient
4. Takes one optimizer step
5. Checks predictions moved TOWARD targets (not away)

If predictions move AWAY from targets, there's a sign error somewhere.

Run: python -m diagnostics.sign_check [--game connect4]
"""
import numpy as np
import torch
import torch.nn.functional as F

from network import AlphaZeroNet
from game_configs import GAME_CONFIGS
from train import load_game
from games.connect4 import GameState as C4State


def main():
    game = load_game("connect4")
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
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)  # large LR to see effect

    # === Test cases with KNOWN correct values ===
    test_cases = []

    # Case 1: X about to win (X to move) → target = +1
    board = np.zeros((6, 7), dtype="int")
    board[0][0:3] = -1; board[0][4] = 1
    s = C4State(None, board, player=-1)
    test_cases.append(("X_winning_X_move", game.state_to_input(s), +1.0))

    # Case 2: O about to win (O to move) → target = +1
    board = np.zeros((6, 7), dtype="int")
    board[0][0:3] = 1; board[0][4] = -1; board[1][4] = -1
    s = C4State(None, board, player=1)
    test_cases.append(("O_winning_O_move", game.state_to_input(s), +1.0))

    # Case 3: X about to lose (X to move, O has 3) → target = -1
    board = np.zeros((6, 7), dtype="int")
    board[0][0:3] = 1; board[0][4] = -1; board[1][4] = -1
    s = C4State(None, board, player=-1)
    test_cases.append(("X_losing_X_move", game.state_to_input(s), -1.0))

    # Case 4: O about to lose (O to move, X has 3) → target = -1
    board = np.zeros((6, 7), dtype="int")
    board[0][0:3] = -1; board[0][4] = 1
    s = C4State(None, board, player=1)
    test_cases.append(("O_losing_O_move", game.state_to_input(s), -1.0))

    # Case 5: Empty board (X to move) → target ~0
    s = game.new_game()
    test_cases.append(("empty_board", game.state_to_input(s), 0.0))

    states = torch.FloatTensor(np.array([tc[1] for tc in test_cases])).to(device)
    targets = torch.FloatTensor([tc[2] for tc in test_cases]).unsqueeze(1).to(device)

    print("=" * 70)
    print("SIGN SANITY CHECK")
    print("=" * 70)

    # === Step 1: Initial predictions ===
    net.eval()
    with torch.no_grad():
        pred_before, _ = net(states)
    pred_before_np = pred_before.squeeze(1).cpu().numpy()

    print("\nBEFORE training step:")
    print(f"  {'Name':<25} {'Target':>8} {'Pred':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8}")
    for tc, pred in zip(test_cases, pred_before_np):
        print(f"  {tc[0]:<25} {tc[2]:>+8.3f} {pred:>+8.4f}")

    # === Step 2: Compute loss ===
    net.train()
    pred_vs, pred_pis = net(states)
    value_loss = F.mse_loss(pred_vs, targets)
    print(f"\nValue loss: {value_loss.item():.6f}")

    # === Step 3: Check gradient of loss w.r.t. predictions ===
    # d(MSE)/d(pred) = 2*(pred - target)/N
    # If pred > target, gradient is positive → optimizer should decrease pred
    # If pred < target, gradient is negative → optimizer should increase pred
    with torch.no_grad():
        grad_direction = 2 * (pred_vs - targets)
    print(f"\nGradient direction (2*(pred-target)):")
    for tc, gd in zip(test_cases, grad_direction.squeeze(1).cpu().numpy()):
        print(f"  {tc[0]:<25} grad={gd:+.4f}  -> should {'decrease' if gd > 0 else 'increase'} pred")

    # === Step 4: Take optimizer step ===
    optimizer.zero_grad()
    value_loss.backward()

    # Check actual gradients on value head parameters
    print(f"\nValue head gradients:")
    for name, param in net.named_parameters():
        if "value" in name and param.grad is not None:
            g = param.grad
            print(f"  {name:<30} shape={str(list(g.shape)):<15} "
                  f"mean={g.mean().item():+.6f} std={g.std().item():.6f} "
                  f"norm={g.norm().item():.6f}")

    optimizer.step()

    # === Step 5: Check predictions AFTER step ===
    net.eval()
    with torch.no_grad():
        pred_after, _ = net(states)
    pred_after_np = pred_after.squeeze(1).cpu().numpy()

    print(f"\nAFTER training step:")
    print(f"  {'Name':<25} {'Target':>8} {'Before':>8} {'After':>8} {'Delta':>8} {'Correct?':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    all_correct = True
    for tc, before, after in zip(test_cases, pred_before_np, pred_after_np):
        delta = after - before
        target = tc[2]
        error_before = abs(before - target)
        error_after = abs(after - target)
        moved_toward = error_after < error_before
        marker = "OK" if moved_toward else "WRONG!"
        if not moved_toward:
            all_correct = False
        print(f"  {tc[0]:<25} {target:>+8.3f} {before:>+8.4f} {after:>+8.4f} "
              f"{delta:>+8.4f} {marker:>8}")

    # === Step 6: Multi-step convergence test ===
    print(f"\n{'='*70}")
    print("MULTI-STEP CONVERGENCE TEST (100 steps, lr=0.01)")
    print("=" * 70)

    # Reset network
    net2 = AlphaZeroNet(
        input_channels=input_channels,
        board_shape=game.board_shape,
        action_size=game.action_size,
        num_res_blocks=game_cfg.get("num_res_blocks", 3),
        num_filters=game_cfg.get("num_filters", 128),
        value_head_channels=game_cfg.get("value_head_channels", 2),
        value_head_fc_size=game_cfg.get("value_head_fc_size", 64),
    ).to(device)
    opt2 = torch.optim.Adam(net2.parameters(), lr=0.01)

    for step in range(100):
        net2.train()
        pv, pp = net2(states)
        vl = F.mse_loss(pv, targets)
        opt2.zero_grad()
        vl.backward()
        opt2.step()

        if step % 20 == 0 or step == 99:
            net2.eval()
            with torch.no_grad():
                pv2, _ = net2(states)
            preds = pv2.squeeze(1).cpu().numpy()
            sign_correct = sum(1 for tc, p in zip(test_cases, preds)
                             if tc[2] == 0 or np.sign(p) == np.sign(tc[2]))
            print(f"  Step {step:>3}: vloss={vl.item():.4f} "
                  f"signs={sign_correct}/{len(test_cases)} "
                  f"preds=[{', '.join(f'{p:+.3f}' for p in preds)}]")

    print(f"\n{'='*70}")
    if all_correct:
        print("VERDICT: Sign convention is CORRECT. Gradient pushes predictions toward targets.")
    else:
        print("VERDICT: SIGN ERROR DETECTED! Some predictions moved AWAY from targets.")
    print("=" * 70)


if __name__ == "__main__":
    main()

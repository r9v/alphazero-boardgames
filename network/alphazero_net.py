import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Post-activation ResBlock (standard AlphaZero).

    Conv→BN→ReLU→Conv→BN, then add skip and ReLU.
    BN after Conv2 stabilizes residual magnitude (~gamma) regardless of
    weight growth, preventing the eff_gain collapse seen with pre-activation.
    """
    def __init__(self, num_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class AlphaZeroNet(nn.Module):
    def __init__(self, input_channels, board_shape, action_size,
                 num_res_blocks=2, num_filters=256,
                 value_head_channels=2, value_head_fc_size=64,
                 policy_head_channels=2):
        super().__init__()
        self.board_shape = board_shape
        self.action_size = action_size
        board_area = board_shape[0] * board_shape[1]

        # Initial conv block
        self.conv = nn.Conv2d(input_channels, num_filters, 3, padding=1)
        self.bn = nn.BatchNorm2d(num_filters)

        # Residual blocks (post-activation)
        self.res_blocks = nn.ModuleList(
            [ResBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # Value head
        self.value_conv = nn.Conv2d(num_filters, value_head_channels, 1)
        self.value_bn = nn.BatchNorm2d(value_head_channels)
        self.value_fc1 = nn.Linear(value_head_channels * board_area, value_head_fc_size)
        self.value_dropout = nn.Dropout(p=0.2)
        self.value_fc2 = nn.Linear(value_head_fc_size, 3)  # WDL: Win/Draw/Loss logits

        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, policy_head_channels, 1)
        self.policy_bn = nn.BatchNorm2d(policy_head_channels)
        self.policy_fc = nn.Linear(policy_head_channels * board_area, action_size)

    def forward(self, x):
        # Backbone
        x = F.relu(self.bn(self.conv(x)))
        for block in self.res_blocks:
            x = block(x)

        # Value head
        v = F.leaky_relu(self.value_bn(self.value_conv(x)), negative_slope=0.01)
        v = v.view(v.size(0), -1)
        v = F.leaky_relu(self.value_fc1(v), negative_slope=0.01)
        v = self.value_dropout(v)
        v = self.value_fc2(v)  # [B, 3] raw WDL logits (no tanh)

        # Policy head (returns raw logits; callers apply softmax/log_softmax)
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        return v, p

    def compile_for_inference(self):
        """Compile the forward pass with torch.compile for faster inference."""
        try:
            self.eval()
            # Store compiled forward function (not module) to avoid
            # circular reference that causes recursion in self.eval()
            self._compiled_forward = torch.compile(self.forward, mode="reduce-overhead")
        except Exception:
            self._compiled_forward = None

    @torch.no_grad()
    def predict(self, state_input):
        """Run inference on a single state input.

        Args:
            state_input: numpy array of shape (C, H, W)

        Returns:
            (value, policy) where value is a float and policy is a numpy array
        """
        self.eval()
        device = next(self.parameters()).device
        x = torch.FloatTensor(state_input).unsqueeze(0).to(device)
        v, p = self(x)
        # WDL logits → scalar: v = P(win) - P(loss)
        probs = F.softmax(v, dim=1)
        value = (probs[0, 0] - probs[0, 2]).item()
        policy = F.softmax(p, dim=1).squeeze(0).cpu().numpy()
        return value, policy

    @torch.no_grad()
    def batch_predict(self, state_inputs, detailed_timing=False):
        """Run inference on a batch of state inputs.

        Args:
            state_inputs: list of numpy arrays, each of shape (C, H, W)
            detailed_timing: if True, return timing breakdown as third element

        Returns:
            (values, policies) or (values, policies, timing_dict)
        """
        self.eval()
        device = next(self.parameters()).device
        use_fp16 = device.type == 'cuda'
        fwd = getattr(self, '_compiled_forward', None) or self.forward

        t0 = time.time()
        n = len(state_inputs)
        inp_array = np.array(state_inputs)
        # Pad to next multiple of 8 to reduce CUDAGraph recompilation
        PAD_MULTIPLE = 8
        padded_n = ((n + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
        if padded_n > n:
            pad_shape = (padded_n - n,) + inp_array.shape[1:]
            inp_array = np.concatenate([inp_array,
                                        np.zeros(pad_shape, dtype=inp_array.dtype)])
        x = torch.FloatTensor(inp_array).to(device)
        if use_fp16:
            x = x.half()
            torch.cuda.synchronize()
        transfer_time = time.time() - t0

        t0 = time.time()
        with torch.autocast('cuda', enabled=use_fp16):
            v, p = fwd(x)
        if use_fp16:
            torch.cuda.synchronize()
        forward_time = time.time() - t0

        t0 = time.time()
        # WDL logits → scalar: v = P(win) - P(loss)
        probs = F.softmax(v.float(), dim=1)[:n]
        values = (probs[:, 0] - probs[:, 2]).cpu().numpy().tolist()
        policies = F.softmax(p.float(), dim=1)[:n].cpu().numpy()
        result_time = time.time() - t0

        if detailed_timing:
            return values, list(policies), {
                "transfer_time": transfer_time,
                "forward_time": forward_time,
                "result_time": result_time,
            }
        return values, list(policies)

    def save(self, directory, iteration=None, num_iterations=None):
        os.makedirs(directory, exist_ok=True)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(directory, f"{timestr}.pt")
        torch.save(self.state_dict(), path)

        latest_path = os.path.join(directory, "latest.txt")
        with open(latest_path, "w") as f:
            f.write(f"{timestr}.pt")

        iter_str = f"iter {iteration+1}/{num_iterations}" if iteration is not None else ""
        print(f"  Checkpoint saved: {path} {iter_str}")

        return path

    def load_latest(self, directory):
        latest_path = os.path.join(directory, "latest.txt")
        if os.path.exists(latest_path):
            with open(latest_path) as f:
                name = f.read().strip()
            path = os.path.join(directory, name)
            if self.load(path):
                return path
            return None
        # Fall back to best.pt (e.g. fresh clone without latest.txt)
        best_path = os.path.join(directory, "best.pt")
        if self.load(best_path):
            return best_path
        return None

    def load(self, path):
        if not os.path.exists(path):
            return False
        device = next(self.parameters()).device
        state_dict = torch.load(path, weights_only=True, map_location=device)
        try:
            self.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            missing = [k for k in self.state_dict() if k not in state_dict]
            unexpected = [k for k in state_dict if k not in self.state_dict()]
            print(f"  [WARNING] Checkpoint mismatch: "
                  f"missing={missing}, unexpected={unexpected}")
            print(f"  Loading with strict=False — mismatched layers keep random weights")
            self.load_state_dict(state_dict, strict=False)
        return True

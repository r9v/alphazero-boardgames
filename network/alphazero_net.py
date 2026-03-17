import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import wdl_to_scalar


def ws_conv2d(x, conv):
    """Conv2d with Weight Standardization + Kaiming scaling.

    Normalizes conv weights to zero-mean, then scales to Kaiming magnitude
    (sqrt(2/fan_in)) per output filter. Weights learn direction only;
    magnitude is fixed at the variance-preserving scale.

    Without the Kaiming factor, WS normalizes to std=1.0 per filter, but
    with fan_in=C*kH*kW=1152 elements, this makes activations ~24x too large.
    The Kaiming factor corrects this to preserve activation variance.
    """
    w = conv.weight
    mean = w.mean(dim=[1, 2, 3], keepdim=True)
    std = w.std(dim=[1, 2, 3], keepdim=True) + 1e-5
    fan_in = w.shape[1] * w.shape[2] * w.shape[3]
    w = (w - mean) / std * (2.0 / fan_in) ** 0.5
    return F.conv2d(x, w, conv.bias, conv.stride, conv.padding)


class ResBlock(nn.Module):
    """Pre-activation ResBlock with Weight Standardization on both convs.

    GN→ReLU→WS-Conv1→GN→ReLU→WS-Conv2 + skip.
    Clean residual path. Weight Standardization on all convs prevents weight
    explosion by normalizing weights per filter before each forward pass.
    Residual branch scaled by 1/√L (Fixup-style) to prevent variance
    explosion through depth: Var grows as (1+1/L)^L ≈ e instead of 2^L.
    GroupNorm used instead of BatchNorm: immune to non-stationary RL data
    distribution (no running stats to drift).
    """
    def __init__(self, num_filters, res_scale=0.5, num_groups=8):
        super().__init__()
        self.bn1 = nn.GroupNorm(num_groups, num_filters)
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn2 = nn.GroupNorm(num_groups, num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.res_scale = res_scale

    def forward(self, x):
        residual = x
        x = ws_conv2d(F.relu(self.bn1(x)), self.conv1)
        x = ws_conv2d(F.relu(self.bn2(x)), self.conv2)
        return x * self.res_scale + residual


class AlphaZeroNet(nn.Module):
    def __init__(self, input_channels, board_shape, action_size,
                 num_res_blocks=2, num_filters=256,
                 value_head_channels=2, value_head_fc_size=64,
                 policy_head_channels=2,
                 backbone_dropout=0.15, num_groups=8):
        super().__init__()
        self.board_shape = board_shape
        self.action_size = action_size
        board_area = board_shape[0] * board_shape[1]

        # Initial conv block (GroupNorm: immune to non-stationary RL data)
        self.conv = nn.Conv2d(input_channels, num_filters, 3, padding=1)
        self.bn = nn.GroupNorm(num_groups, num_filters)

        # Residual blocks (pre-activation / pre-norm)
        # Scale residual branch by 1/√L to control variance growth through depth
        res_scale = num_res_blocks ** -0.5
        self.res_blocks = nn.ModuleList(
            [ResBlock(num_filters, res_scale=res_scale, num_groups=num_groups)
             for _ in range(num_res_blocks)]
        )

        # Final GN→ReLU after all ResBlocks (standard pre-norm pattern)
        # Ensures backbone output is normalized before heads
        self.final_bn = nn.GroupNorm(num_groups, num_filters)

        # Channel dropout on backbone output before heads split.
        # Prevents channel segregation between value and policy heads.
        # Drops entire channels (Dropout2d) so neither head can exclusively own channels.
        self.backbone_dropout = nn.Dropout2d(p=backbone_dropout)

        # Value head (1 group = LayerNorm-like for small channel count)
        self.value_conv = nn.Conv2d(num_filters, value_head_channels, 1)
        self.value_bn = nn.GroupNorm(1, value_head_channels)
        self.value_fc1 = nn.Linear(value_head_channels * board_area, value_head_fc_size)
        self.value_dropout = nn.Dropout(p=0.2)
        self.value_fc2 = nn.Linear(value_head_fc_size, 3)  # WDL: Win/Draw/Loss logits

        # Policy head (1 group = LayerNorm-like for small channel count)
        self.policy_conv = nn.Conv2d(num_filters, policy_head_channels, 1)
        self.policy_bn = nn.GroupNorm(1, policy_head_channels)
        self.policy_fc = nn.Linear(policy_head_channels * board_area, action_size)

        # GN config validation: warn if any group normalizes over too few elements
        H, W = board_shape
        gn_configs = [
            ("backbone", num_groups, num_filters, H * W),
            ("value_head", 1, value_head_channels, H * W),
            ("policy_head", 1, policy_head_channels, H * W),
        ]
        for label, G, C, spatial in gn_configs:
            elems = (C // G) * spatial
            status = "OK" if elems >= 64 else "WARN" if elems >= 16 else "LOW"
            if status != "OK":
                print(f"  [GN-{status}] {label}: {G} groups, {C//G} ch/group, "
                      f"{elems} elems/group (recommend >=64)")

    def backbone_forward(self, x):
        """Run backbone only (conv + res_blocks + final_bn), no heads or dropout."""
        x = F.relu(self.bn(ws_conv2d(x, self.conv)))
        for block in self.res_blocks:
            x = block(x)
        return F.relu(self.final_bn(x))

    def forward(self, x):
        # Backbone
        x = self.backbone_forward(x)

        # Channel dropout: prevent head channel segregation
        x = self.backbone_dropout(x)

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
        value = wdl_to_scalar(v)[0].item()
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
        values = wdl_to_scalar(v.float()[:n]).cpu().numpy().tolist()
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

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import wdl_to_scalar, find_latest_checkpoint


def ws_conv2d(x, conv):
    """Conv2d with Weight Standardization + Kaiming scaling."""
    w = conv.weight
    mean = w.mean(dim=[1, 2, 3], keepdim=True)
    std = w.std(dim=[1, 2, 3], keepdim=True) + 1e-5
    fan_in = w.shape[1] * w.shape[2] * w.shape[3]
    w = (w - mean) / std * (2.0 / fan_in) ** 0.5
    return F.conv2d(x, w, conv.bias, conv.stride, conv.padding)


class ResBlock(nn.Module):
    """Pre-activation ResBlock: GN→ReLU→WS-Conv, residual scaled by 1/√L."""
    def __init__(self, num_filters, res_scale=0.5, num_groups=8, dropout=0.0):
        super().__init__()
        self.bn1 = nn.GroupNorm(num_groups, num_filters)
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn2 = nn.GroupNorm(num_groups, num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.res_scale = res_scale
        self.drop = nn.Dropout2d(p=dropout) if dropout > 0 else None

    def forward(self, x):
        residual = x
        x = ws_conv2d(F.relu(self.bn1(x)), self.conv1)
        if self.drop is not None:
            x = self.drop(x)
        x = ws_conv2d(F.relu(self.bn2(x)), self.conv2)
        return x * self.res_scale + residual


class AlphaZeroNet(nn.Module):
    def __init__(self, input_channels, board_shape, action_size,
                 num_res_blocks=2, num_filters=256,
                 value_head_channels=2, value_head_fc_size=64,
                 policy_head_channels=2,
                 backbone_dropout=0.15, num_groups=8,
                 resblock_dropout=0.0):
        super().__init__()
        self.board_shape = board_shape
        self.action_size = action_size
        board_area = board_shape[0] * board_shape[1]

        self.conv = nn.Conv2d(input_channels, num_filters, 3, padding=1)
        self.bn = nn.GroupNorm(num_groups, num_filters)

        res_scale = num_res_blocks ** -0.5
        self.res_blocks = nn.ModuleList(
            [ResBlock(num_filters, res_scale=res_scale, num_groups=num_groups,
                      dropout=resblock_dropout)
             for _ in range(num_res_blocks)]
        )

        self.final_bn = nn.GroupNorm(num_groups, num_filters)

        # Channel dropout prevents head channel segregation
        self.backbone_dropout = nn.Dropout2d(p=backbone_dropout)

        # Value head: GAP → FC → WDL logits
        self.value_conv = nn.Conv2d(num_filters, value_head_channels, 1)
        self.value_bn = nn.GroupNorm(1, value_head_channels)
        self.value_fc1 = nn.Linear(value_head_channels, value_head_fc_size)
        self.value_dropout = nn.Dropout(p=0.2)
        self.value_fc2 = nn.Linear(value_head_fc_size, 3)

        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, policy_head_channels, 1)
        self.policy_bn = nn.GroupNorm(1, policy_head_channels)
        self.policy_fc = nn.Linear(policy_head_channels * board_area, action_size)

        # Warn if any GN group has too few elements
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
        x = self.backbone_forward(x)
        x = self.backbone_dropout(x)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.mean(dim=(2, 3))
        v = F.relu(self.value_fc1(v))
        v = self.value_dropout(v)
        v = self.value_fc2(v)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        return v, p

    def compile_for_inference(self):
        """Compile the forward pass with torch.compile for faster inference."""
        try:
            self.eval()
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
        out = self(x)
        v, p = out[0], out[1]
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
        # Pad to fixed bucket sizes so CUDAGraph can reuse recorded graphs
        _BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        padded_n = next((b for b in _BUCKETS if b >= n), ((n + 63) // 64) * 64)
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
            v, p = fwd(x)[:2]
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
        path = find_latest_checkpoint(directory)
        if path and self.load(path):
            return path
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

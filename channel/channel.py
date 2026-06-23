from __future__ import annotations

import numpy as np
import torch
from torch import nn


class BSC_Channel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.bit_num = config.bit_num
        self.symbols = 2 ** self.bit_num
        self.ber = None
        self.transition_matrix = None
        self.register_buffer(
            "weights",
            torch.tensor([[2**i] for i in range(self.bit_num)], dtype=torch.float32, device=config.device),
        )

    def calculate_transition_matrix(self, ber):
        if self.transition_matrix is not None and self.ber == ber:
            return self.transition_matrix, self.ber
        symbols = torch.arange(self.symbols, device=self.weights.device)
        bits = symbols.unsqueeze(1).bitwise_and(1 << torch.arange(self.bit_num, device=self.weights.device)).ne(0)
        hamming = torch.logical_xor(bits.unsqueeze(1), bits.unsqueeze(0)).sum(dim=2)
        probs = torch.tensor([(ber**k) * ((1 - ber) ** (self.bit_num - k)) for k in range(self.bit_num + 1)],
                             dtype=torch.float32, device=self.weights.device)
        self.transition_matrix = probs[hamming]
        self.ber = ber
        return self.transition_matrix, self.ber

    def BSC_forward(self, input_symbols, ber):
        self.calculate_transition_matrix(ber)
        bsz, length = input_symbols.shape
        probs = self.transition_matrix[input_symbols.reshape(-1)]
        return torch.multinomial(probs, num_samples=1).reshape(bsz, length)

    def BSC_bit_flip(self, input_symbols, ber):
        input_device = input_symbols.device
        bits = input_symbols.unsqueeze(2).bitwise_and(1 << torch.arange(self.bit_num, device=input_device)).ne(0).long()
        flips = torch.rand(bits.shape, device=input_device) < ber
        flipped = bits ^ flips.long()
        weights = (1 << torch.arange(self.bit_num, device=input_device)).reshape(1, 1, self.bit_num)
        return (flipped * weights).sum(dim=-1).reshape_as(input_symbols).long()


class convention_channel(nn.Module):
    """Lightweight uncoded 4QAM/QPSK AWGN channel.

    This keeps the original `integer_trans_llr` interface without depending on
    TensorFlow/Sionna. It returns hard decoded symbols plus p(y|x0) likelihoods.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bit_num = config.bit_num
        self.register_buffer(
            "weights",
            torch.tensor([[2**i] for i in range(config.bit_num)], dtype=torch.float32, device=config.device),
        )
        constellation = torch.tensor(
            [[1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]],
            dtype=torch.float32,
            device=config.device,
        ) / np.sqrt(2.0)
        self.register_buffer("qam4_constellation", constellation)

    def integer_trans_llr(self, input_integer, snr=100, chan_type="AWGN", ldpc=False, snr_offset=0):
        if chan_type.upper() != "AWGN":
            raise ValueError("The lightweight release channel currently supports AWGN only.")
        if self.bit_num % 2 != 0:
            raise ValueError("4QAM maps two bits per symbol, so bit_num must be even.")

        input_shape = input_integer.shape
        device = input_integer.device
        flat = input_integer.reshape(-1).long()
        bits = flat.unsqueeze(1).bitwise_and(1 << torch.arange(self.bit_num, device=device)).ne(0).long()
        bit_pairs = bits.reshape(-1, self.bit_num // 2, 2)
        pair_indices = bit_pairs[:, :, 0] + 2 * bit_pairs[:, :, 1]

        snr_linear = 10 ** (snr / 10)
        sigma = np.sqrt(1.0 / (2.0 * snr_linear))
        tx = self.qam4_constellation[pair_indices]
        rx = tx + sigma * torch.randn_like(tx)

        hard_pairs = torch.stack((rx[..., 0] < 0, rx[..., 1] < 0), dim=-1).long()
        decoded_bits = hard_pairs.reshape(-1, self.bit_num)
        out = torch.matmul(decoded_bits.float(), self.weights).reshape(*input_shape).long()

        sigma_est = np.sqrt(1.0 / (2.0 * (10 ** ((snr + snr_offset) / 10))))
        diff = rx.unsqueeze(2) - self.qam4_constellation.reshape(1, 1, 4, 2)
        dist_sq = diff.pow(2).sum(dim=-1)
        pair_prob = torch.softmax(-dist_sq / (2.0 * sigma_est**2 + 1e-12), dim=-1)
        prob = self.cal_symbol_prob_from_4qam_pairs(pair_prob, self.bit_num).to(device)

        ber = (decoded_bits.long() != bits.long()).float().mean().item()
        ser = (out.reshape(-1) != flat).float().mean().item()
        return {
            "out": out,
            "hard_out": out,
            "hard_bits": decoded_bits.reshape(*input_shape, self.bit_num),
            "ber": ber,
            "ser": ser,
            "prob": prob,
            "qam_pair_prob": pair_prob,
        }

    @staticmethod
    def cal_trans_prob(p0, num_bits_per_symbol):
        symbols = torch.arange(2**num_bits_per_symbol)
        symbol_bits = symbols.unsqueeze(1).bitwise_and(1 << torch.arange(num_bits_per_symbol)).ne(0).float()
        p0 = p0.reshape(-1, num_bits_per_symbol).unsqueeze(1)
        symbol_bits = symbol_bits.unsqueeze(0)
        probs = (1 - p0) * symbol_bits + p0 * (1 - symbol_bits)
        return probs.prod(dim=2).float()

    @staticmethod
    def cal_symbol_prob_from_4qam_pairs(pair_prob, num_bits_per_symbol):
        device = pair_prob.device
        symbols = torch.arange(2**num_bits_per_symbol, device=device)
        bits = symbols.unsqueeze(1).bitwise_and(1 << torch.arange(num_bits_per_symbol, device=device)).ne(0).long()
        candidate_pair_indices = bits.reshape(-1, num_bits_per_symbol // 2, 2)
        candidate_pair_indices = candidate_pair_indices[:, :, 0] + 2 * candidate_pair_indices[:, :, 1]
        gather_index = candidate_pair_indices.transpose(0, 1).unsqueeze(0).expand(pair_prob.shape[0], -1, -1)
        selected = torch.gather(pair_prob, dim=2, index=gather_index)
        return selected.prod(dim=1).float()

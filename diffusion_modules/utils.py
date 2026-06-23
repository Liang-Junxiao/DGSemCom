import torch
import torch.nn.functional as F

def index_to_log_onehot(x: torch.LongTensor, num_classes: int) -> torch.FloatTensor:
    """
    Convert batch of vector of class indices into batch of log onehot vectors

    Args:
        x (`torch.LongTensor` of shape `(batch size, vector length)`):
            Batch of class indices

        num_classes (`int`):
            number of classes to be used for the onehot vectors

    Returns:
        `torch.FloatTensor` of shape `(batch size, num classes, vector length)`:
            Log onehot vectors
    """
    x_onehot = F.one_hot(x, num_classes)
    x_onehot = x_onehot.permute(0, 2, 1)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

def gumbel_noised(logits: torch.FloatTensor, generator) -> torch.FloatTensor:
    """
    Apply gumbel noise to `logits`
    """
    uniform = torch.rand(logits.shape, device=logits.device, generator=generator)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    noised = gumbel_noise + logits
    return noised

def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)

def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))
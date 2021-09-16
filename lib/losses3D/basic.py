import torch
# Code was adapted and mofified from https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 5D output image (NxCxDxHxW)
    """
    # print('start expand_as_one_hot----------------------------------------------------------------------------------')
    if input.dim() == 5:
        return input
    assert input.dim() == 4

    print(f'bf input.dim() = {input.dim()}, input.shape = {input.size()}')
    # expand the input tensor to Nx1xDxHxW before scattering
    input = input.unsqueeze(1)
    print(f'af input.dim() = {input.dim()}, input.shape = {input.size()}')
    # create result tensor shape (NxCxDxHxW)
    shape = list(input.size())
    print(f'basic.py: bf shape len = {len(shape)}, shape = {shape}')
    shape[1] = C
    print(f'basic.py: af shape len = {len(shape)}, shape = {shape}')
    # print(f'basic.py C = {C}')
    print(f'afaf input.dim() = {input.dim()}, input.shape = {input.size()}')

    if ignore_index is not None:
        print(f'basic.py: ignore_index is not None')
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the lib tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # print(f'basic.py: ignore_index is None')
        # print(f'torch.zeros(shape).to(input.device).scatter_(1, input, 1) = {torch.zeros(shape).to(input.device).scatter_(1, input, 1)}')
        # print(f'shape = {shape}')
        # print(f'bf basic.py: input size = {input.size()}')
        # input = input.expand(shape)
        # print(f'af basic.py: input size = {input.size()}')
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)


def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """
    # print('start compute_per_channel_dice----------------------------------------------------')
    # input and target shapes must match
    # print(f'input.size = {input.size()}, target.size = {target.size()}')
    # target = target.unsqueeze(1)
    # print(f'input.size = {input.size()}, target.size = {target.size()}')
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # print(f'basic.py: input = {input}')
    # print(f'target = {target}')
    input = flatten(input)
    # print(f'basic.py: bf flatten target.size() = {target.size()}')
    target = flatten(target)
    # print(f'basic.py: af flatten target.size() = {target.size()}')
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    # clamp: make all elements in input into range [min, max]
    return 2 * (intersect / denominator.clamp(min=epsilon))


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # print('start flatten----------------------------------------------------------')
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

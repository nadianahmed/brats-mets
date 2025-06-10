
import torch.nn.functional as F

def crop_or_pad(tensor, target_shape):
    '''
    Crop or pad a 5D tensor (N, C, D, H, W) to match target shape.

    Parameters:
    - tensor(torch.Tensor): the input tensor.
    - target_shape(Int, Int, Int, Int, Int): the shape for the target output.

    Returns:
    - torch.Tensor: the updated sensor.
    '''
    _, _, d, h, w = tensor.shape
    td, th, tw = target_shape

    # --- Depth (D) ---
    if d > td:
        dd = (d - td) // 2
        tensor = tensor[:, :, dd:dd+td, :, :]
    elif d < td:
        pad_d = (td - d)
        tensor = F.pad(tensor, (0, 0, 0, 0, pad_d // 2, pad_d - pad_d // 2))

    # --- Height (H) ---
    if h > th:
        dh = (h - th) // 2
        tensor = tensor[:, :, :, dh:dh+th, :]
    elif h < th:
        pad_h = (th - h)
        tensor = F.pad(tensor, (0, 0, pad_h // 2, pad_h - pad_h // 2, 0, 0))

    # --- Width (W) ---
    if w > tw:
        dw = (w - tw) // 2
        tensor = tensor[:, :, :, :, dw:dw+tw]
    elif w < tw:
        pad_w = (tw - w)
        tensor = F.pad(tensor, (pad_w // 2, pad_w - pad_w // 2, 0, 0, 0, 0))

    return tensor
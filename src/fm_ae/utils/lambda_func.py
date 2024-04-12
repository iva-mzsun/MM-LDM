import torch

def obtain_lambda(t, func_type, use_fp16):
    assert torch.all(t >= 0)
    assert torch.all(t <= 1)
    # t: [b, n], normalized values
    if func_type.lower() == 'exp':
        lamb = torch.exp(t * -3)
    else:
        raise NotImplementedError

    if use_fp16:
        lamb = lamb.to(memory_format=torch.contiguous_format).half()
    else:
        lamb = lamb.to(memory_format=torch.contiguous_format).float()
    return lamb
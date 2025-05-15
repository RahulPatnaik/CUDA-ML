import torch, time
from conv2d_cuda import conv2d_forward
import torch.nn.functional as F

for size in [256, 512, 1024]:
    inp = torch.randn(size, size, device="cuda")
    flt = torch.ones(3,3, device="cuda")/9

    # warmup
    conv2d_forward(inp, flt)
    F.conv2d(inp.unsqueeze(0).unsqueeze(0), flt.unsqueeze(0).unsqueeze(0), padding=1)

    # custom
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(20): conv2d_forward(inp, flt)
    torch.cuda.synchronize(); t1 = time.time()

    # pytorch
    torch.cuda.synchronize(); t2 = time.time()
    for _ in range(20): F.conv2d(inp.unsqueeze(0).unsqueeze(0), flt.unsqueeze(0).unsqueeze(0), padding=1)
    torch.cuda.synchronize(); t3 = time.time()

    print(size,
          f"custom {(t1-t0)/20*1e3:.3f}ms",
          f"torch  {(t3-t2)/20*1e3:.3f}ms")

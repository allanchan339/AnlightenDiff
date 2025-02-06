import torch
import os


def gpuoption():
    g1 = torch.cuda.get_device_name(0)
    if g1 == 'NVIDIA GeForce RTX 4090':
        os.environ["NCCL_P2P_DISABLE"] = "1"
        return True
    else:
        return False


if __name__ == '__main__':
    gpuoption()

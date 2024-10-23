import numpy as np
import torch.optim as optim

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup=50, max_iters=100):
        self.warmup_epoch = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup_epoch:
            lr_factor *= epoch * 1.0 / self.warmup_epoch
        return lr_factor

if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    warmup = 100
    max_iters = 100
    # Needed for initializing the lr scheduler
    p = nn.Parameter(torch.empty(4, 4))
    optimizer = optim.Adam([p], lr=1e-3)
    lr_scheduler = CosineWarmupScheduler(
        optimizer=optimizer, warmup=warmup, max_iters=max_iters)

    # Plotting
    epochs = list(range(2000))
    # sns.set()
    
    x = plt.figure(figsize=(8, 3))
    plt.plot(epochs, [lr_scheduler.get_lr_factor(e) for e in epochs])
    plt.ylabel("Learning rate factor")
    plt.xlabel("Iterations (in batches)")
    plt.title(f"Cosine Warm-up Learning Rate Scheduler warmup={warmup}, max_iter={max_iters}")
    print()
    # plt.savefig()

import torch
from torch import nn
from torch.nn import functional as F
import os
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self, seed=1234):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )
        # init weights to the same thing
        gen = torch.Generator()
        gen.manual_seed(seed)
        nn.init.kaiming_normal_(self.fc1.weight, generator=gen)
        nn.init.kaiming_normal_(self.fc2[0].weight, generator=gen)
        nn.init.kaiming_normal_(self.fc2[2].weight, generator=gen)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2[0].bias)
        nn.init.zeros_(self.fc2[2].bias)
    
    def forward(self, x, y=None):
        if y is None:
            y = 0
        print(list(self.fc1.parameters()), x.shape, x.device, y.shape, y.device)
        x = self.fc1(x) + y 
        print(x.shape, x.device, list(self.fc2.parameters()))
        x = self.fc2(x)
        return x

def init_dist(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    # torchrun --nproc-per-node=2 test_fsdp.py

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    print(f"LOCAL_RANK: {local_rank}, WORLD_SIZE: {world_size}")
    init_dist(local_rank, world_size)
    torch.cuda.set_device(local_rank)
    dev = torch.cuda.current_device()

    model = Net().to(dev)
    ignored_states = []
    ## when model fc1 grad is set to false, then it can be added to `ignored_states` and FSDP doesnt mind
    # model.fc1.requires_grad_(False)
    # ignored_states = [model.fc1] # + [x for x in model.fc1.modules()]
    model = FSDP(model, ignored_states=ignored_states)
    # model.fc1.requires_grad_(False)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(10000, 32).to(dev)
    y = torch.randn(10000, 64).to(dev)
    pbar = tqdm(list(range(1000)))

    print(model) 

    for i in pbar:
        optim.zero_grad()
        out = model(x, y)

        ### This does not work, gives some error about Outputview (only on the layers that are sharded)
        # x1 = model.fc1(x) + y
        # print(x1.shape)
        # out = model.fc2(x1)
        # print(out.shape)

        loss = (out**2).mean()
        loss.backward()
        optim.step()
        pbar.set_description(f"Iteration: {i}, loss: {loss.item()}.")

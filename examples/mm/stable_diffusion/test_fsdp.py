import torch
from torch import nn
from torch.nn import functional as F
import os
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP 
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
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
        x = self.fc1(x) + y 
        x = self.fc2(x)
        return x

def init_dist(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def test_recursive_model(dev):
    print("Testing recursiveness. ")
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(32, 32)
            self.ref = self

        def forward(self, x):
            return self.fc(x)
    
    model = Net()
    print(model.fc)
    print(model.ref.__class__.__name__)


def test_ignored_states(dev):
    print("Testing ignored states.")
    model = Net().to(dev)
    ignored_states = []
    ## when model fc1 grad is set to false, then it can be added to `ignored_states` and FSDP doesnt mind
    model.fc1.requires_grad_(False)
    ignored_states = [model.fc1] 
    model = FSDP(model, ignored_states=ignored_states)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)   # optimizer must be called AFTER fsdp

    x = torch.randn(10000, 32).to(dev)
    y = torch.randn(10000, 64).to(dev)
    pbar = tqdm(list(range(100)))

    print(model) 
    for i in pbar:
        optim.zero_grad()
        out = model(x, y)
        ### This does not work, we're calling submodules of Net independently. Note that these submodules are just empty views now
        ### and the actual parameters are sharded and are managed internally
        ### to invoke them the actual model has to be called
        # x1 = model.fc1(x) + y
        # print(x1.shape)
        # out = model.fc2(x1)
        # print(out.shape)
        loss = (out**2).mean()
        loss.backward()
        optim.step()
        pbar.set_description(f"Iteration: {i}, loss: {loss.item()}.")

def test_nested_policy(dev):
    print("Testing nested policy")
    # we will have two modules, nested within the same module (the idea is that second module acts like an adapter)
    # we will attempt to learn only the unfrozen part inside fsdp
    class AdapterModule(nn.Module):
        def __init__(self, *chans):
            super().__init__()
            n = len(chans)
            layers = []
            for i in range(n-1):
                layers.append(nn.Linear(chans[i], chans[i+1]))
                layers.append(nn.ReLU())
            self.layers = nn.Sequential(*layers[:-1])
        
        def forward(self, x):
            return self.layers(x)

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.mods = nn.ModuleList(
                [nn.Sequential(   # we will freeze this one
                    nn.Linear(32, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64)
                ),
                AdapterModule(64, 128, 1)  # we give it a different name to wrap appropriately
                ]
            )
        
        def forward(self, x):
            x = self.mods[0](x)
            x = self.mods[1](x)
            return x
    
    # define a model
    model = Network().to(dev)
    model.mods[0].requires_grad_(False)
    # wrap in fsdp
    model = FSDP(model, auto_wrap_policy=ModuleWrapPolicy([AdapterModule]))       # works
    # model = FSDP(model)           # does not work since it has submodules in different `requires_grad` states
    print(model)
    

if __name__ == '__main__':
    # torchrun --nproc-per-node=2 test_fsdp.py

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    print(f"LOCAL_RANK: {local_rank}, WORLD_SIZE: {world_size}")
    init_dist(local_rank, world_size)
    torch.cuda.set_device(local_rank)
    dev = torch.cuda.current_device()

    # test_ignored_states(dev)
    # test_nested_policy(dev)
    test_recursive_model(dev)

    cleanup()
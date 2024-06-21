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
    
def test_partial_batchsize_fsdp(dev, rank, world_size, forward_max_rank=None):
    '''
    The idea of this test is to reduce the memory usage with only running forward passes on 1 machine and seeing the mem usage stats
    
    Update: This does not work with subset of workers, all workers have to contribute.
    '''
    class Module(nn.Module):
        def __init__(self,):
            super().__init__()
            modules = [nn.Linear(32, 32) for _ in range(20)]
            self.mods = nn.Sequential(*modules)

        def forward(self, x):
            return self.mods(x)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            modules = [Module() for _ in range(world_size)]
            self.mods = nn.Sequential(*modules)

        def forward(self, x):
            return self.mods(x)
    
    # we have defined the model class, now instantiate
    net = FSDP(Net(), auto_wrap_policy=ModuleWrapPolicy([Module]), device_id=dev)
    if local_rank == 0:
        print(net)
    # now run an optimization on this
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)
    if local_rank == 0:
        input("Before training...")
    torch.distributed.barrier()
    pbar = range(1000) if local_rank > 0 else tqdm(range(1000))
    for i in pbar:
        optim.zero_grad()
        # if local_rank < forward_max_rank:
        if True:
            x = torch.randn(10000, 32).to(dev)
            out = net(x)
            loss = F.mse_loss(out, -x)
            loss.backward()
            print(f"iter: {i}, {local_rank}, {loss.item()}")
        optim.step()

    if local_rank == 0:
        input("After training...")
    torch.distributed.barrier()
    

    
def print_cuda_stats(rank):
    print(f"Stats for rank {rank}")
    total_memory = torch.cuda.get_device_properties(rank).total_memory
    reserved_memory = torch.cuda.memory_reserved(rank)
    allocated_memory = torch.cuda.memory_allocated(rank)
    free_memory = reserved_memory - allocated_memory
    print(f"Total GPU Memory: {total_memory / 1e9} GB")
    print(f"Reserved Memory: {reserved_memory / 1e9} GB")
    print(f"Allocated Memory: {allocated_memory / 1e9} GB")
    print(f"Free Memory (within reserved): {free_memory / 1e9} GB")

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
    # test_recursive_model(dev)
    test_partial_batchsize_fsdp(dev, local_rank, world_size, 2)
    if local_rank == 0:
        print_cuda_stats(local_rank)

    cleanup()
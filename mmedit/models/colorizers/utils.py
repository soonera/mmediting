from typing import List, Tuple, Callable, Any, Collection
import torch
from torch import Tensor, nn

from fastai.callbacks.hooks import Hooks

Sizes = List[List[int]]
LayerFunc = Callable[[nn.Module], None]
ModuleList = Collection[nn.Module]


def is_listy(x:Any)->bool: return isinstance(x, (tuple,list))

def _hook_inner(m,i,o): return o if isinstance(o,Tensor) else o if is_listy(o) else list(o)

def hook_outputs(modules:Collection[nn.Module], detach:bool=True, grad:bool=False)->Hooks:
    "Return `Hooks` that store activations of all `modules` in `self.stored`"
    return Hooks(modules, _hook_inner, detach=detach, is_forward=not grad)


def children(m:nn.Module)->ModuleList:
    "Get children of `m`."
    return list(m.children())


def num_children(m:nn.Module)->int:
    "Get number of children modules in `m`."
    return len(children(m))


class ParameterModule(nn.Module):
    "Register a lone parameter `p` in a module."

    def __init__(self, p: nn.Parameter):
        super().__init__()
        self.val = p

    def forward(self, x): return x


def children_and_parameters(m:nn.Module):
    "Return the children of `m` and its direct parameters not registered in modules."
    children = list(m.children())
    children_p = sum([[id(p) for p in c.parameters()] for c in m.children()],[])
    for p in m.parameters():
        if id(p) not in children_p: children.append(ParameterModule(p))
    return children


flatten_model = lambda m: sum(map(flatten_model,children_and_parameters(m)),[]) if num_children(m) else [m]


def in_channels(m:nn.Module) -> List[int]:
    "Return the shape of the first weight layer in `m`."
    for l in flatten_model(m):
        if hasattr(l, 'weight'): return l.weight.shape[1]
    raise Exception('No weight layer')


def one_param(m: nn.Module)->Tensor:
    "Return the first parameter of `m`."
    return next(m.parameters())


def dummy_batch(m: nn.Module, size:tuple=(64,64))->Tensor:
    "Create a dummy batch to go through `m` with `size`."
    ch_in = in_channels(m)
    return one_param(m).new(1, ch_in, *size).requires_grad_(False).uniform_(-1.,1.)


def dummy_eval(m:nn.Module, size:tuple=(64,64)):
    "Pass a `dummy_batch` in evaluation mode in `m` with `size`."
    return m.eval()(dummy_batch(m, size))


def model_sizes(m:nn.Module, size:tuple=(64,64))->Tuple[Sizes,Tensor,Hooks]:
    "Pass a dummy input through the model `m` to get the various sizes of activations."
    with hook_outputs(m) as hooks:
        x = dummy_eval(m, size)
        return [o.stored.shape for o in hooks]


def sigmoid_range(x, low, high):
    "Sigmoid function with range `(low, high)`"
    return torch.sigmoid(x) * (high - low) + low


def ifnone(a:Any,b:Any)->Any:
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a



























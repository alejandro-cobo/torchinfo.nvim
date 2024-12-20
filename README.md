# torchinfo.nvim

Show information about Pytorch modules inside Neovim.

## Installation

Using lazy.nvim:

```lua
{ "alejandro-cobo/torchinfo.nvim", name = "torchinfo" }
```

In order for the plugin to work, the ```python``` command must point to a python
installation with all your required packages installed, including
[Pytorch](https://pytorch.org/).

If you are using a virtual environment (which is recommended), this means that
you must run ```source <venv/bin/activate>``` or similar before launching Neovim.

## Usage

This plugin exports a function called ```get_info``` which accepts a python file
path as an argument. You can run this function inside Neovim. For example, to
pass the current file:

```
:lua require("torchinfo").get_info(vim.fn.expand("%"))
```

You can also define a keymap to make this process easier:

```lua
{
    "alejandro-cobo/torchinfo.nvim",
    name = "torchinfo",
    config = function()
        local torchinfo = require("torchinfo")
        -- Get info from current file
        vim.keymap.set("n", "<leader>ti", function()
            torchinfo.get_info(vim.fn.expand("%"))
        end)
    end
}
```

This plugin imports all the ```torch.nn.Module``` classes from the input file
and tries to instantiate them. This means that only classes with defined
default parameters are selected.

For example, given the following python file:
```python
from torch import nn


class ModuleA(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.fc = nn.Linear(in_dims, out_dims)

    def forward(self, x):
        return self.fc(x)


class ModuleB(nn.Module):
    def __init__(self, in_dims=100, out_dims=100):
        super().__init__()
        self.fc = nn.Linear(in_dims, out_dims)

    def forward(self, x):
        return self.fc(x)
```

ModuleA will be ignored while ModuleB will be successfully analyzed. You can
temporally add default values to any module to include it in the analysis.

## Potential future features

- [x] Get a minimal version of the plugin working.
- [x] Explore other ways of showing information other than cmdline messages.
- [ ] Use concurrency to run the script in background.
- [ ] Output more info other than number of parameters, such as FLOPs.
- [ ] Output a graph showing module connections.


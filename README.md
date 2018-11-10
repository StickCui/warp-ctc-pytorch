# Warp-CTC-PyTorch

An extension of Baidu [warp-ctc](https://github.com/baidu-research/warp-ctc) for [PyTorch](https://github.com/pytorch/pytorch).

## Introduction

This is a modified version of [SeanNaren/warp-ctc](https://github.com/SeanNaren/warp-ctc). I just modify the code to the new CPP Extensions API style of PyTorch.

When I use the source [SeanNaren/warp-ctc](https://github.com/SeanNaren/warp-ctc), some problems bother me. So I modify the source codes.

## Requirements

- pytorch>=0.4.0
- cmake>=2.8

Note: test environment: pytorch 0.4.1 & 1.0.0.dev20181106

## Installation

```bash
git clone --recursive https://github.com/StickCui/warp-ctc-pytorch.git
```

1. Install to your PYTHONPATH

    Run the script as follow (Using Python3 as default):
    ```bash
    cd warp-ctc-pytorch
    sh make.sh install
    ```
    It will install the extension module in your user pythonpath (e.g. ~/.local/lib/python3.5/site_package)

    You can also install it by yourself like:
    ```bash
    cd warp-ctc-pytorch
    sh make.sh core
    sudo python3 setup.py install
    ```
    or
    ```bash
    cd warp-ctc-pytorch/warpctc/core
    mkdir build
    cd build
    cmake ..
    make
    cd ../../../
    sudo python3 setup.py install
    ```
2. Build inplace to embed to your project

    Run the script as follow (Using Python3 as default):
    ```bash
    cd warp-ctc-pytorch
    sh make.sh build
    ```
    or
    ```bash
    cd warp-ctc-pytorch
    sh make.sh core
    python3 setup.py build_ext --inplace
    ```

## How to Use

For initialization, there two parameters:
```Python
CTCLoss(self, size_average=True, reduces=True):
"""
Args:
    size_average (bool, optional): By default,
            the losses are averaged by minibatch.
            If the field :attr:`size_average`
            is set to ``False``, the losses are instead
            summed for each minibatch. Ignored
            when reduces is ``False``. Default: ``True``
    reduce (bool, optional): By default, the losses are averaged
            or summed over observations for each minibatch
            depending on :attr:`size_average`. When :attr:`reduce`
            is ``False``, returns a loss per batch element instead
            and ignores :attr:`size_average`. Default: ``True``
"""
```

As for forward:
```Python
forward(self, preds, labels, preds_lens, label_lens)
"""
Shape:
        preds: :math:`(seqLength, batch, outputDim)`. Tensor contains output from network
        labels: :math:`(X,)`. Tensor contains all the targets of the batch in one sequence
        preds_lens: :math:`(batch,)`. Tensor contains size of each output sequence from the network
        label_lens: :math:`(batch,)`. Tensor contains label length of each example
"""
```

Similar with the [Document](https://github.com/SeanNaren/warp-ctc#documentation).

Example:
```Python
import torch
import numpy as np
from warpctc import CTCLoss

costfunc = CTCLoss()

preds = torch.Tensor(100, 128, 37)
label_length = torch.from_numpy(np.random.randint(20, 80, (128,), dtype=np.int32))
s = label_length.sum()
s = int(s)
labels = torch.from_numpy(np.random.randint(1, 36, (s,), dtype=np.int32))

loss = costfunc(preds, labels, preds_size, label_length)
loss.backward()
```

## References

[1] SeanNaren. https://github.com/SeanNaren/warp-ctc. 2018/11/10

[2] baidu-research. https://github.com/baidu-research/warp-ctc. 2018/11/10

#pragma once

#ifdef PYTORCH_VER_0_4
#include <torch/torch.h>
#else
#include <torch/extension.h>
#endif

#include "ctc.h"

int gpu_ctc(const at::Tensor &probs,
            at::Tensor &grads,
            const at::Tensor &labels,
            const at::Tensor &label_sizes,
            const at::Tensor &sizes,
            const int minibatch_size,
            at::Tensor &costs);
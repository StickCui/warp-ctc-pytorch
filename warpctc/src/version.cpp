#include <iostream>

#ifdef WITH_CUDA
    #include "cpu/version.h"
    #include "cuda/version.h"
#else
    #include "cpu/version.h"
#endif

int ctc(const at::Tensor &probs,
            at::Tensor &grads,
            const at::Tensor &labels,
            const at::Tensor &label_sizes,
            const at::Tensor &sizes,
            const int minibatch_size,
            at::Tensor &costs)
{
    if (probs.type().is_cuda())
    {
#ifdef WITH_CUDA
        return gpu_ctc(probs, grads, labels, label_sizes, sizes, minibatch_size, costs);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    return cpu_ctc(probs, grads, labels, label_sizes, sizes, minibatch_size, costs);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("ctc", &ctc, "Warp CTC Loss");
}

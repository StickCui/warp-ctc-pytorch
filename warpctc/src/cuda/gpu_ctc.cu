//#include "cuda/version.h"
#include "ctc.h"

#include <ATen/ATen.h>

#ifdef PYTORCH_VER_0_4
extern THCState *state;
#else
#include <ATen/cuda/CUDAContext.h>
#endif

#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>

int gpu_ctc(const at::Tensor &probs,
            at::Tensor &grads,
            const at::Tensor &labels,
            const at::Tensor &label_sizes,
            const at::Tensor &sizes,
            const int minibatch_size,
            at::Tensor &costs)
{
    AT_ASSERTM(probs.type().is_cuda(), "probs must be a CUDA tensor");
    AT_ASSERTM(grads.type().is_cuda(), "grads must be a CUDA tensor");

    float *probs_ptr = probs.data<float>();
    float *grads_ptr = grads.data<float>();
    //AT_DISPATCH_FLOATING_TYPES(probs.to(at::ScalarType::Float).type(), "gpu_ctc", [&] {
    //    probs_ptr = probs.contiguous().data<float>(); // point
    //    grads_ptr = grads.contiguous().data<float>(); //s-calar_t
    //});

    int *sizes_ptr = sizes.data<int>();
    int *labels_ptr = labels.data<int>();
    int *label_sizes_ptr = label_sizes.data<int>();
    /*
    AT_DISPATCH_INTEGRAL_TYPES(sizes.to(at::ScalarType::Int).type(), "gpu_ctc", [&] {
        sizes_ptr = sizes.contiguous().data<int>();
        labels_ptr = labels.contiguous().data<int>();
        label_sizes_ptr = label_sizes.contiguous().data<int>();
    });
    */

    float *costs_ptr = costs.data<float>();
    /*
    AT_DISPATCH_FLOATING_TYPES(costs.to(at::ScalarType::Float).type(), "gpu_ctc", [&] {
        costs_ptr = costs.contiguous().data<float>();
    });
    */

    int nclass = probs.size(2);

#ifdef PYTORCH_VER_0_4
    cudaStream_t stream = THCState_getCurrentStream(state);
#else
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#endif

    ctcOptions options;
    memset(&options, 0, sizeof(options));
    options.loc = CTC_GPU;
    options.stream = stream;

    size_t gpu_size_bytes;
    get_workspace_size(label_sizes_ptr, sizes_ptr,
                       nclass, minibatch_size,
                       options, &gpu_size_bytes);

#ifndef PYTORCH_VER_0_4
    THCState *state = at::globalContext().lazyInitCUDA(); // TODO replace with getTHCState
#endif
    void *gpu_workspace = NULL;
    gpu_workspace = THCudaMalloc(state, gpu_size_bytes);
    //at::Allocator *allocator = at::cuda::getCUDADeviceAllocator();
    //void *gpu_workspace = allocator->raw_allocate(gpu_size_bytes);
    compute_ctc_loss(probs_ptr, grads_ptr,
                     labels_ptr, label_sizes_ptr,
                     sizes_ptr, nclass,
                     minibatch_size, costs_ptr,
                     gpu_workspace, options);

    THCudaFree(state, gpu_workspace);
    //allocator->raw_deallocate(gpu_workspace);
    //THCudaCheck(cudaGetLastError());
    return 1;
}
#include "cpu/version.h"

int cpu_ctc(const at::Tensor &probs,
            at::Tensor &grads,
            const at::Tensor &labels,
            const at::Tensor &label_sizes,
            const at::Tensor &sizes,
            const int minibatch_size,
            at::Tensor &costs)
{
    AT_ASSERTM(!probs.type().is_cuda(), "probs must be a CPU tensor");
    AT_ASSERTM(!grads.type().is_cuda(), "grads must be a CPU tensor");

    float *probs_ptr = probs.contiguous().data<float>();
    float *grads_ptr = grads.contiguous().data<float>();
    //AT_DISPATCH_FLOATING_TYPES(probs.to(at::ScalarType::Float).type(), "gpu_ctc", [&] {
    //    probs_ptr = probs.contiguous().data<float>(); // point
    //    grads_ptr = grads.contiguous().data<float>(); //s-calar_t
    //});

    int *sizes_ptr = sizes.contiguous().data<int>();
    int *labels_ptr = labels.contiguous().data<int>();
    int *label_sizes_ptr = label_sizes.contiguous().data<int>();
    /*
    AT_DISPATCH_INTEGRAL_TYPES(sizes.to(at::ScalarType::Int).type(), "gpu_ctc", [&] {
        sizes_ptr = sizes.contiguous().data<int>();
        labels_ptr = labels.contiguous().data<int>();
        label_sizes_ptr = label_sizes.contiguous().data<int>();
    });
    */

    float *costs_ptr = costs.contiguous().data<float>();
    /*
    AT_DISPATCH_FLOATING_TYPES(costs.to(at::ScalarType::Float).type(), "gpu_ctc", [&] {
        costs_ptr = costs.contiguous().data<float>();
    });
    */

    int nclass = probs.size(2);

    ctcOptions options;
    memset(&options, 0, sizeof(options));
    options.loc = CTC_CPU;
    options.num_threads = 0; // will use default number of threads

#if defined(CTC_DISABLE_OMP) || defined(APPLE)
    // have to use at least one
    options.num_threads = std::max(options.num_threads, (unsigned int)1);
#endif

    size_t cpu_size_bytes;
    get_workspace_size(label_sizes_ptr, sizes_ptr,
                       nclass, minibatch_size,
                       options, &cpu_size_bytes);

    float *cpu_workspace = new float[cpu_size_bytes / sizeof(float)];

    compute_ctc_loss(probs_ptr, grads_ptr,
                     labels_ptr, label_sizes_ptr,
                     sizes_ptr, nclass,
                     minibatch_size, costs_ptr,
                     cpu_workspace, options);

    delete[] cpu_workspace;
    return 1;
}
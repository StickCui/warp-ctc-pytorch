import os
import sys
import platform
from setuptools import setup, find_packages

import torch
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    warp_ctc_path = 'warpctc/core/build'

    main_file = ['warpctc/src/version.cpp']
    cpu_file = ['warpctc/src/cpu/cpu_ctc.cpp']
    cuda_file = ['warpctc/src/cuda/gpu_ctc.cu']

    include_dirs = ['warpctc/core/include', 'warpctc/src']
    extra_compile_args = {"cxx": ['-std=c++11', '-fPIC']}
    define_macros = []
    source_files = main_file + cpu_file
    extension = CppExtension

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        source_files += cuda_file
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-Wno-deprecated-gpu-targets"
        ]
    
    # torch version
    if '0.4' in torch.__version__:
        define_macros += [("PYTORCH_VER_0_4", None)]
    elif ('0.2.0' in torch.__version__) or ('0.3.0' in torch.__version__) or ('0.3.1' in torch.__version__):
        raise RuntimeError("Pytorch version less than 0.4.0.\n"
                           "Can not use this version.")
    
    if platform.system() == 'Darwin':
        lib_ext = ".dylib"
    else:
        lib_ext = ".so"
    
    enquire_lib = os.path.join(warp_ctc_path, "libwarpctc" + lib_ext)


    if not os.path.exists(enquire_lib):
        print(("Could not find libwarpctc.so in {}.\n"
            "Build warp-ctc and set WARP_CTC_PATH to the location of"
            " libwarpctc.so (default is 'core/build')").format(warp_ctc_path))
        sys.exit(1)
    
    ext_modules = [
        extension(
            "warpctc._warp_ctc",
            source_files,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            library_dirs=[os.path.join(this_dir, warp_ctc_path)],
            runtime_library_dirs=[os.path.join(this_dir, warp_ctc_path)],
            libraries=['warpctc']
        )
    ]

    return ext_modules


setup(
    name="warpctc",
    version="0.1",
    author="StickCui",
    url="https://github.com/StickCui/warp-ctc-pytorch",
    description="Baidu WarpCTC in pytorch",
    packages=find_packages(),
    install_requires=['torch'],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)

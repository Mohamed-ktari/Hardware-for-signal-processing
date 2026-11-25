from setuptools import setup
from torch.utils.cpp_extension import (
 BuildExtension,
 CUDAExtension
)
setup(
  name="custom_lcuda",
  ext_modules=[
  CUDAExtension(
  name="custom_cuda",
  sources=["linear.cpp", "linear_cuda.cu"]
  )
  ],
  cmdclass={"build_ext": BuildExtension}
)
# Introduction to our project
It is the repo for final project of course EECS E4750 Heterogenous Computing 2021 Fall. Our project name is Fast and High Precision Signal MultiresolutionEngine using Parallel Processors. For detailed introduction, please refer to our wiki page.

# File Organization
```
├── CUDA-HSPME.ipynb
├── README.md
├── cuda_hspme.py
├── figures
│   ├── conv_result.png
│   ├── mask_generation_test.png
│   ├── multi_convolution_over_input.png
│   └── multi_convolution_over_scale.png
├── kernel_code.txt
├── requirements.txt
└── test_cuda.py

2 directories, 12 files
```

## Detailed description

`kernel_code.txt` contains all kernel code used in our project.

All kernels are divided into three parts:

**Improved kernel implementation code** this section contains three kernels we implemented for our core algorithm, mask generation, conv for small masks and conv for large masks

**Naive kernel implementation code** this section contains two kernels for naive gpu implementation, one for simple mask generation and one for simple conv

**Experimental Kernel and Utils** this section contains some utilization kernel, like scan... and some experimental kernels that we did not do a complete test integrated with other parts, like mask generation with jitter and dither.

`cuda_hspme.py` contains all pycuda code about processing on host and kernel calls

`CUDA-HSPME.ipynb` is a jupyter notebook that contains all test code and all results we have got for our final report. It is the best way to run code if you want to reproduce our result.

The test part can be divided into 3 parts, corresponding to section V-B ~ V-D in our final report. First part is test of mask generation. Second part is test of runtime analysis of multiple convolutions. Third part is visualization and numerical analysis of multiple convolutions.

`test_cuda.py` is another choice in case you do not want to run jupyter notebook. It contains all the same code with in CUDA-HSPME.ipynb. You can run it using python3 in command line to get the results. However, the output may be hard to visualize in this way.

`figures` this directory contains all figures we generated for final project

## Code Running Instructions
We did not use any outside data so there is no need to download or include extra data. You should first install all dependent libraries provided in requirements.txt. Then the best way we recommend to reproduce our results is running CUDA-HSPME.ipynb line by line. If you prefer to run with command line, we also provided a py file, you can run python3 test_cuda.py in command line to get prints and figures.

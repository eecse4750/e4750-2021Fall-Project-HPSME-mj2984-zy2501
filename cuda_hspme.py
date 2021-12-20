import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda import gpuarray

import pywt
import numpy as np
import scipy as sp
from scipy.signal import convolve
import time
import math

class Wavelet_Transform():
    def __init__(self, name, file_path):
        
        self.mod = self.getSourceModule(file_path)
        
        # define datatype used for each type of input
        self.mask_length_matrix_datatype = 'int32'
        self.mask_length_matrix_scan_datatype = 'int32'
        self.multiresolution_mask_datatype = 'float64' # This will also be considered as datatype for scales as well.
        
        # define constant memory and L2 cache as limitation
        self.constant_mem_size_bytes = 64 * 1024 # 64Kb in bytes.
        self.thread_cmem_cache = 8 * 1024 # 8Kb in bytes.
        
        # the number of mask elements that can be stored into constant memory in total. Generally 1024.
        self.constant_mem_max_num_maskelements = (self.constant_mem_size_bytes/8) 
        # the maximum number of mask elements that can be stored into constant memory AND moved into thread cache.
        self.thread_cmem_max_num_maskelements = (self.thread_cmem_cache/8) 
        
        # A little over 1 million samples at most analysed per kernel launch.
        # The next 1 million will be done in next kernel launch
        self.input_signal_size_per_kernel_launch = 1024*1024
        
        # initialize mother function
        if name == 'morl':
            # directly generate wavelet function
            wavelet = pywt.ContinuousWavelet('morl')
            self.int_psi, self.x = wavelet.wavefun(10)
            self.int_psi = self.int_psi.astype('float32')
            self.x = self.x.astype('float32')

    def getSourceModule(self, file_path):
        # read kernel code and compile it using SourceModule
        
        with open(file_path, "r") as f:  
            kernel_wrapper = f.read() 
            
        return SourceModule(kernel_wrapper)
    
    ######################################## Improved GPU Implementation ####################################################
    def gpu_generate_mask_1b1s(self, scales):
        """
        Mask generation part.
        ----------
        Argument
        - scales: A list of scales that used for generation of different masks. If there are n scales, n masks will
                  be generated. Scales should be defined in an increasing order.
        ----------
        Return
        - masks: A 1-D array that contains all masks that are used to do multi-convolution
        - mask_lengths: A n elements array to save the exclusive scan result of mask lengths
        - time: time used in this kernel
        """
        
        # initialize variable to save masks
        num_scales = len(scales)
        mask_lengths = np.array(scales) * 16 + 1
        
        mask_lengths = np.cumsum(mask_lengths)
        max_length = mask_lengths[-1].copy()
        mask_lengths = np.insert(mask_lengths,0,0)
        step = self.x[1] - self.x[0]
        masks = np.empty(max_length, dtype=self.int_psi.dtype)
        
        
        # start record
        e_start = cuda.Event()
        e_end = cuda.Event()
        e_start.record()
        
        scales = scales.astype('int32')
        mask_lengths = mask_lengths.astype('int32')
        
        int_psi_d, _ = self.mod.get_global('int_psi')
        masks_d = cuda.mem_alloc(masks.nbytes)
        scales_d = cuda.mem_alloc(scales.nbytes)
        mask_lengths_d = cuda.mem_alloc(mask_lengths.nbytes)
        
        cuda.memcpy_htod(int_psi_d, self.int_psi)
        cuda.memcpy_htod(scales_d, scales)
        cuda.memcpy_htod(mask_lengths_d, mask_lengths)
        
        func = self.mod.get_function('generate_mask_1b1s')
        
        block_dim = (1024,1,1)
        grid_dim = (num_scales,1,1)
        
        func(scales_d, mask_lengths_d, masks_d, 
             np.intc(num_scales), np.float32(step), np.intc(max_length), block=block_dim, grid=grid_dim)
            
        cuda.memcpy_dtoh(masks, masks_d)
        
        # stop record
        e_end.record()
        e_end.synchronize()
        
        return masks, mask_lengths, e_start.time_till(e_end)*1e-3
    
    def gpu_convolve_small_mask(self, x, masks, mask_lengths):
        """
        Do multi-convolutions to many samll masks whose length is smaller than full_cmem(1024).
        Most of the time there will be more than 1 mask, so it will generate a 2-D output matrix.
        ---------
        Argument
        - x: Input signal with shape (T,)
        - masks: A 1-D array that contains all small masks
        - mask_lengths: A array of (N,) the exclusive scan result
        ---------
        Return
        - conv_results: the output matrix of multi-convolution, should be shape (N,T)
        - time: time used in this kernel
        """
        
        x = x.astype('double')
        masks = masks.astype('double')
        mask_lengths = mask_lengths.astype('int32')
        
        T = len(x)
        N = len(mask_lengths) - 1
        conv_results = np.zeros([N,T], dtype = x.dtype)
        
        masks_d = cuda.mem_alloc(masks.nbytes)
        x_d = cuda.mem_alloc(x.nbytes)
        mask_lengths_d = cuda.mem_alloc(mask_lengths.nbytes)
        conv_results_d = cuda.mem_alloc(conv_results.nbytes)
        
        cuda.memcpy_htod(masks_d, masks)
        cuda.memcpy_htod(x_d, x)
        cuda.memcpy_htod(mask_lengths_d, mask_lengths)
        
        func = self.mod.get_function('small_mask_multi_convolution')
        
        block_dim = (512,1,1)
        grid_dim = (math.ceil(T/block_dim[0]),1,1)
        
        mask_begin = 1
        mask_end = len(mask_lengths)-1
        
        input_size = block_dim[0]
        padding_size = mask_lengths[mask_end] - mask_lengths[mask_end-1] - 1
        num_masks = mask_end-mask_begin+1
        shared_size = (input_size + padding_size + 2*num_masks) * 8
        
        # start record
        e_start = cuda.Event()
        e_end = cuda.Event()
        e_start.record()
        
        func(x_d, conv_results_d, masks_d, mask_lengths_d, np.intc(mask_begin), 
             np.intc(mask_end), np.intc(T), block=block_dim, grid=grid_dim, shared=int(shared_size))

        cuda.memcpy_dtoh(conv_results, conv_results_d)
        
        # stop record
        e_end.record()
        e_end.synchronize()
        
        return conv_results, e_start.time_till(e_end)*1e-3
    
    def gpu_convolve_large_mask(self, x, masks, n_sidelobe, center_length):
        """
        Do convolutions to sidelobes of large masks.
        ----------
        Argument
        - x: Input signal with shape (T,)
        - masks: A 1-D array that contains the mask elements of this large mask
        - n_sidelobe: Define how many sidelobes this large mask contains
        - center_length: The length of center part of this large mask, should be in [0, 1024)
        ----------
        Return
        - conv_results: (1,T), partial result of this large mask, should be added to center result
        - time: time used in this kernel
        """
        # prepare datatype to conform to kernel argument
        x = x.astype('double')
        masks = np.ascontiguousarray(masks[::-1])
        masks = masks.astype('double')
        
        # initialize data that is out of the loop
        full_mask_length = n_sidelobe * 1024 + center_length
        T = len(x)
        conv_results = np.zeros(T, dtype = x.dtype)
        conv_results_d = cuda.mem_alloc(conv_results.nbytes)
        x_d = cuda.mem_alloc(x.nbytes)
        cuda.memcpy_htod(x_d, x)
        cuda.memcpy_htod(conv_results_d, conv_results)
        func = self.mod.get_function('two_full_sidelobe_convolution')
        
        block_dim = (512,1,1)
        grid_dim = (math.ceil(T/block_dim[0]),1,1)
        
        # use a for loop to process full sidelobes
        
        # start record
        e_start = cuda.Event()
        e_end = cuda.Event()
        e_start.record()
        
        for sidelobe_idx in range(n_sidelobe):
            left_start_point = int(sidelobe_idx*512)
            right_start_point = int(full_mask_length - (sidelobe_idx+1)*512)
            left_mask = masks[left_start_point:left_start_point+512]
            right_mask = masks[right_start_point:right_start_point+512]

            left_mask_d = cuda.mem_alloc(left_mask.nbytes)
            right_mask_d = cuda.mem_alloc(right_mask.nbytes)
            cuda.memcpy_htod(left_mask_d, left_mask)
            cuda.memcpy_htod(right_mask_d, right_mask)

            func(x_d, conv_results_d, left_mask_d, right_mask_d, np.intc(left_start_point),
                 np.intc(right_start_point), np.intc(T), np.intc(full_mask_length), block=block_dim, grid=grid_dim)
        
        # copy conv results back
        cuda.memcpy_dtoh(conv_results, conv_results_d)
        
        # stop record
        e_end.record()
        e_end.synchronize()
        
        return conv_results, e_start.time_till(e_end)*1e-3
    
    def multiresolution_conv(self, x, allmasks, mask_lengths_scan, scales):
        """
        A big function that input all masks, then do mask preparation and split. Then conduct multiple convolutions to
        small masks and shifted/multiple convolutions to large masks respectively. Finally collect and concatenate all
        results to return final results for wavelet transform.
        -----------
        Argument
        - x: Input signal with shape (T,), T should be much longer than number of masks to fully use GPU resources
        - all_masks: A 1-D array that contans all masks values, should be generated by gpu_generate_mask
        - mask_lengths_scam: A array of (N,) the exclusive scan result, should be generated by gpu_generate_mask
        - scales: A list of scales that used for generation of different masks.
        ----------
        Return
        - conv_results: Final conv results for wavelet transform on n scales, should be (N,T)
        - time: Time consumed by the whole kernel process
        """
        # define small masks
        mask_lengths = scales * 16 + 1
        if_small_mask = mask_lengths < 1024
        small_masks_lengths = mask_lengths[if_small_mask]
        
        # split out small masks and its elements
        idx_last_small_mask = len(small_masks_lengths)
        bound = mask_lengths_scan[idx_last_small_mask]
        
        # do small mask convolutions
        small_masks = allmasks[:bound]
        small_mask_res, t1 = self.gpu_convolve_small_mask(x, small_masks, mask_lengths_scan[:idx_last_small_mask+1])

        
        # split out large masks and its elements
        large_masks = allmasks[bound:]
        if len(large_masks):
            # fetch helper arrays of large masks
            large_masks_lengths = mask_lengths[if_small_mask == False]
            large_masks_scan = np.insert(np.cumsum(large_masks_lengths),0,0)
            
            # initialize results
            large_mask_res = np.zeros([len(large_masks_lengths), len(x)], dtype='float64')
            
            
            ## mask preparation
            large_mask_list = []
            center_mask_len_list = []
            center_mask_list = []

            for mask_idx in range(len(large_masks_lengths)):
                
                mask_length = large_masks_lengths[mask_idx]
                start_point = large_masks_scan[mask_idx]
                
                # compute out how many sidelobe pairs inside this mask
                num_full_sidelobes =  np.floor(mask_length/self.thread_cmem_max_num_maskelements)
                # compute the rest center length, this center length can be used in multiconv
                center_length = mask_length - self.thread_cmem_max_num_maskelements*num_full_sidelobes
                
                # fetch mask values for this large mask
                single_mask = large_masks[start_point:start_point+mask_length]
                # compute middle length
                middle_idx = (mask_length-1)/2
                # fetch center mask
                center_mask = single_mask[int(middle_idx-(center_length-1)/2):int(middle_idx+(center_length-1)/2+1)]

                large_mask_list.append([single_mask, num_full_sidelobes, center_length])
                center_mask_len_list.append(center_length)
                center_mask_list.append(center_mask)
            
            # sort all center masks in increasing order
            sort_idx = np.argsort(center_mask_len_list)
            sorted_mask = []
            sorted_center_len = np.array(center_mask_len_list)[sort_idx]
            for idx in sort_idx:
                sorted_mask.extend(center_mask_list[idx])
            
            masks_scan = np.insert(np.cumsum(sorted_center_len),0,0)
            
            # do multiple convolutions to center parts of all large masks
            center_mask_res, t2 = self.gpu_convolve_small_mask(x, np.array(sorted_mask), masks_scan)
            t3 = 0
            
            # do shifted convolutions to all sidelobes of each large mask
            for i, idx in enumerate(sort_idx):
                mask_info = large_mask_list[idx]
                sidelobe_res, t = self.gpu_convolve_large_mask(x, mask_info[0], int(mask_info[1]), mask_info[2])
                large_mask_res[idx,:] = sidelobe_res + center_mask_res[i,:]
                t3 = t
            
            # concatenate the results of small masks and large masks to get final wavelet transform results
            res = np.concatenate([small_mask_res, large_mask_res])
            return res, t1+t2+t3
        
        # if no large masks, just return small masks result
        else:
            return small_mask_res, t1
    
    ######################################## Naive GPU Implementation ####################################################
    def gpu_generate_mask(self, scales):
        
        # initialize variable to save masks
        num_scales = len(scales)
        step = self.x[1] - self.x[0]
        length = self.x[-1] - self.x[0]
        longest_mask = ((self.x[-1] - self.x[0]) * scales[-1] + 1).astype('int')
        masks = np.empty([num_scales, longest_mask], dtype=self.int_psi.dtype)
        mask_lengths = scales * 16 + 1
        
        # start record
        e_start = cuda.Event()
        e_end = cuda.Event()
        e_start.record()
        
        masks_d = cuda.mem_alloc(masks.nbytes)
        int_psi_d = cuda.mem_alloc(self.int_psi.nbytes)
        scales_d = cuda.mem_alloc(scales.nbytes)
        
        cuda.memcpy_htod(int_psi_d, self.int_psi)
        cuda.memcpy_htod(scales_d, scales.astype('float32'))
        
        func = self.mod.get_function('generate_mask')
        
        block_dim = (1024,1,1)
        grid_dim = (math.ceil(masks.shape[1]/block_dim[0]), math.ceil(masks.shape[0]/block_dim[1]), 1)
        
        func(scales_d, int_psi_d, masks_d, np.intc(masks.shape[0]), np.intc(masks.shape[1]),
             np.intc(num_scales), np.float32(step), np.intc(length), block=block_dim, grid=grid_dim)
        
        cuda.memcpy_dtoh(masks, masks_d)
        
        # stop record
        e_end.record()
        e_end.synchronize()
        
        return masks, mask_lengths, e_start.time_till(e_end)*1e-3
    
    def gpu_convolve(self, x, masks, mask_lengths):
        
        x = x.astype('float32')
        masks = masks.astype('float32')
        mask_lengths = mask_lengths.astype('int32')
        
        T = x.shape[0]
        N = masks.shape[0]
        conv_results = np.empty([N,T], dtype = x.dtype)
        
        masks_d = cuda.mem_alloc(masks.nbytes)
        x_d = cuda.mem_alloc(x.nbytes)
        mask_lengths_d = cuda.mem_alloc(mask_lengths.nbytes)
        conv_results_d = cuda.mem_alloc(conv_results.nbytes)
        
        # start record
        e_start = cuda.Event()
        e_end = cuda.Event()
        e_start.record()
        
        cuda.memcpy_htod(masks_d, masks)
        cuda.memcpy_htod(x_d, x)
        cuda.memcpy_htod(mask_lengths_d, mask_lengths)
        
        func = self.mod.get_function('simple_conv')
        
        block_dim = (1024,1,1)
        grid_dim = (math.ceil(T/block_dim[0]), math.ceil(N/block_dim[1]), 1)
        
        func(x_d, masks_d, mask_lengths_d, conv_results_d, 
             np.intc(N), np.intc(T), np.intc(masks.shape[1]), block=block_dim, grid=grid_dim)
        
        cuda.memcpy_dtoh(conv_results, conv_results_d)
        
        # stop record
        e_end.record()
        e_end.synchronize()
        
        return conv_results, e_start.time_till(e_end)*1e-3
    
        
    ########################################### Serial Code #######################################################
    def py_generate_mask(self, scales):
        """
        Python serial code to generate masks.
        For comparison with GPU implementation.
        """
        num_scales = len(scales)
        longest_mask = ((self.x[-1] - self.x[0]) * scales[-1] + 1).astype('int')
        masks = []
        mask_lengths = np.zeros(num_scales, dtype=np.int32)
        start = time.time()
        
        for i in range(num_scales):
            scale = scales[i]
            step = self.x[1] - self.x[0]
            length = (self.x[-1] - self.x[0]) * scale + 1
            j = np.arange(length) / (scale * step)
            j = j.astype(int)
            int_psi_scale = self.int_psi[j]
            masks.append(int_psi_scale)
            
        end = time.time()
        
        return masks,end - start
    
    def py_convolve(self, x, masks):
        """
        Python serial code to conduct multiple convolutions one by one.
        For comparison with GPU implementation.
        """
        T = x.shape[0]
        N = len(masks)
        conv_results = np.zeros([N,T], dtype = x.dtype)
        
        start = time.time()
        
        for i in range(N):
            mask = masks[i]
            if len(mask)>len(x):
                conv = convolve(x, mask, 'same')
            else:
                conv = np.convolve(x, mask, 'same')
            conv_results[i] = conv
        
        end = time.time()
        
        return conv_results, end-start
    
    ########################################### Experimental Function#######################################################
    def generate_masks_and_scans(self, multiresolution_function_psi, scale_factors_nr, scale_factors_dr, dither_psi, jitter_psi, realtime, inputfile_masks):
        """
        Used to generate masks with jitter and ditther.
        Done with small tests
        Did not do tests integrated with other parts.
        """
        multiresolution_function_enum = 1 # To be generated from wavelet_psi (which is a string), using a look-up table
        jitter_enum = 0 # To be generated from jitter_psi (which is a string), using a look-up table
        dither_enum = 0 # To be generated from dither_psi (which is a string), using a look-up table
        multiresolution_function_temporal_extents = [8,10,14] # Stores the starting and shifting temporal extents of each wavelet_enum.
        # Here for 1st wavelet_enum which may be say a morlet wavelet, it has 8 elements before and after for the highest frequency (total of 17)
        # For the second wavelet enum which may be say a mexican hat wavelet, it has 10 elements before and after (total of 21)
        # Scale factor says how much the scales expand (linearly), if we want it non linearly, we input the scale factors aptly (or change the function, not covered here.)

        # realtime to say if masks are computed and then directly run on convolution (no copying back to CPU), the CPU adaptively intitates different masks to convolve
        # at different times. This is for adaptive algorithm, we are not using it for now. We are also not using datatype and inputfilemasks for now.

#         if (multiresolution_function_enum == 0):
#             f1 = inputfile_masks
#             # Have to import from an existing file and do some checks which parameters are available which are not.
#             # We will do this later.
#             # some may have only lengths but no scan, some may have only scan but no lengths, but all should have the entire mask list into one contiguous array.

#         else:
        mask_length_scan_available = 0
        number_of_masks = len(scale_factors_nr)
        mask_lengths = np.empty(number_of_masks, dtype=self.mask_length_matrix_datatype)

        temporal_extent_ini = multiresolution_function_temporal_extents[multiresolution_function_enum - 1]

        for mask_count in range(number_of_masks):
            mask_lengths[mask_count] = (2 * np.floor(temporal_extent_ini * scale_factors_nr[mask_count]/scale_factors_dr[mask_count])) + 1

#             if (mask_length_scan_available == 0):
#                 if (realtime == 0):
        mask_lengths_scan = np.cumsum(mask_lengths)

        total_allmasks_size = np.int(mask_lengths_scan[number_of_masks - 1]) # since it is a prefix sum, it has the last element.

#             if (multiresolution_function_enum != 0): # We need to compute the masks
        allmasks_dithers_jitters = np.empty((total_allmasks_size,),dtype='float64')
        allmasks_dithers_jitters_gpu = cuda.mem_alloc(allmasks_dithers_jitters.size * allmasks_dithers_jitters.dtype.itemsize)
        
        cuda.memcpy_htod(allmasks_dithers_jitters_gpu, allmasks_dithers_jitters)

        self.grid_dim_x = int(np.ceil(total_allmasks_size/self.threads_per_block_x))
        self.grid_dim_y = 1
        self.grid_dim_z = 1
        grid_dimensions = (self.grid_dim_x,self.grid_dim_y,self.grid_dim_z)

        mask_generator_parameters_cpu = np.empty(number_of_masks*4, dtype=self.multiresolution_mask_datatype)

        for mask_count in range(number_of_masks):
            mask_generator_parameters_cpu[mask_count*4] = mask_lengths_scan[mask_count]
            mask_generator_parameters_cpu[mask_count*4+1] = mask_lengths[mask_count]
            mask_generator_parameters_cpu[mask_count*4+2] = scale_factors_nr[mask_count]
            mask_generator_parameters_cpu[mask_count*4+3] = scale_factors_dr[mask_count]

        func_mask_generate = self.mod.get_function("mask_generate")
        mask_generator_parameters_gpu, _ = self.mod.get_global('mask_generator_parameters')

        cuda.memcpy_htod(mask_generator_parameters_gpu, mask_generator_parameters_cpu)
        func_mask_generate(allmasks_dithers_jitters_gpu, np.intc(number_of_masks), np.intc(multiresolution_function_enum), np.intc(dither_enum), np.intc(jitter_enum), block=self.blockdimensions, grid=grid_dimensions)
        cuda.memcpy_dtoh (allmasks_dithers_jitters, allmasks_dithers_jitters_gpu)
        
        return allmasks_dithers_jitters, None, None
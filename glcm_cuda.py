from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
import numpy as np
from skimage.util import view_as_windows
from tqdm import tqdm

BLOCKS = 256
THREADS = 256
MAX_VALUE_SUPPORTED = 255
NO_VALUES_SUPPORTED = 256 ** 2

@dataclass
class GLCM:
    """
    
    Args:
        max_value: Maximum value of the image, default 256
        step_size: Step size of the window
        radius: Radius of the windows
        bins: Bin reduction. If None, then no reduction is done

    """
    
    max_value: int = 255
    step_size: int = 1
    radius: int = 2
    bins: int | None = None

    HOMOGENEITY = 0
    CONTRAST = 1
    ASM = 2
    MEAN_I = 3
    MEAN_J = 4
    VAR_I = 5
    VAR_J = 6
    CORRELATION = 7

    @staticmethod
    def binarize(im: np.ndarray, from_bins: int, to_bins: int):
        """ Binarize an image from a certain bin to another

        Args:
            im: Image as np.ndarray
            from_bins: From the Bin of input image
            to_bins: To the Bin of output image

        Returns:
            Binarized Image

        """
        return (im / from_bins * to_bins).astype(int)

    def from_nd_image(self,
                      im: np.ndarray):
        """ Generates the GLCM from a multi band image

        Args:
            im: A 3 dim image as an ndarray

        Returns:
            The GLCM Array 4dim with shape
                rows, cols, channel, feature
        """

        glcm_chs = []
        for ch in range(im.shape[-1]):
            glcm_chs.append(self.from_image(im[...,ch]))

        return np.stack(glcm_chs, axis=2)

    def from_image(self,
                   im: np.ndarray) -> np.ndarray:
        """ Generates the GLCM from a single band image

        Args:
            im: Image in np.ndarray. Cannot be in cp.ndarray

        Returns:
            The GLCM Array 3dim with shape
                rows, cols, feature
        """
        diameter = self.radius * 2 + 1

        if self.bins is not None:
            im = self.binarize(im, self.max_value, self.bins)

        # This will yield a shape (window_i, window_j, row, col)
        # E.g. 100x100 with 5x5 window -> 96, 96, 5, 5
        windows_ij = cp.asarray(view_as_windows(im, (diameter, diameter)))

        # We flatten the cells as cell order is not important
        windows_ij = windows_ij.reshape((*windows_ij.shape[:-2], -1))

        # Yield Windows and flatten the windows
        windows_i = windows_ij[:-self.step_size, :-self.step_size] \
            .reshape((-1, windows_ij.shape[-1]))
        windows_j = windows_ij[self.step_size:, self.step_size:] \
            .reshape((-1, windows_ij.shape[-1]))

        glcm_features = cp.zeros((windows_i.shape[0], 8), dtype=cp.float32)

        for e, (i, j) in tqdm(enumerate(zip(windows_i, windows_j)),
                              total=len(windows_i)):
            glcm_features[e] = self._from_windows(i, j)

        return glcm_features.reshape(windows_ij.shape[0] - self.step_size,
                                     windows_ij.shape[1] - self.step_size, 8)

    def _from_windows(self,
                      i: cp.ndarray,
                      j: cp.ndarray, ) -> np.ndarray:
        """ Generate the GLCM from the I J Window

        Notes:
            i must be the same shape as j

        Args:
            i: I Window
            j: J Window

        Returns:
            The GLCM array, of size (8,)

        """
        assert i.shape == j.shape, f"Shape of i {i.shape} != j {j.shape}"
        i_flat = i.flatten()
        j_flat = j.flatten()
        no_of_values = i_flat.size
        glcm = cp.zeros((self.max_value + 1) ** 2, dtype=cp.uint8).flatten()
        features = cp.zeros(8, dtype=cp.float32)

        assert self.max_value <= MAX_VALUE_SUPPORTED, \
            f"Max value supported is {MAX_VALUE_SUPPORTED}"
        assert no_of_values < NO_VALUES_SUPPORTED, \
            f"Max number of values supported is {NO_VALUES_SUPPORTED}"

        self.glcm_kernel(
            grid=(256,),
            block=(256,),
            args=(
                i_flat, j_flat, self.max_value, no_of_values, glcm, features
            ),
        )

        return features

    glcm_kernel = cp.RawKernel(
            r"""
            #define HOMOGENEITY 0 
            #define CONTRAST 1    
            #define ASM 2         
            #define MEAN_I 3      
            #define MEAN_J 4      
            #define VAR_I 5      
            #define VAR_J 6       
            #define CORRELATION 7 
            extern "C" {
                __device__ static inline char atomicAdd(
                    unsigned char* address,
                    unsigned char val
                    ) 
                {
                    // https://stackoverflow.com/questions/5447570/cuda-atomic-operations-on-unsigned-chars
                    size_t long_address_modulo = (size_t) address & 3;
                    unsigned int* base_address = (unsigned int*) ((char*) address - long_address_modulo);
                    unsigned int long_val = (unsigned int) val << (8 * long_address_modulo);
                    unsigned int long_old = atomicAdd(base_address, long_val);
            
                    if (long_address_modulo == 3) {
                        // the first 8 bits of long_val represent the char value,
                        // hence the first 8 bits of long_old represent its previous value.
                        return (char) (long_old >> 24);
                    } else {
                        // bits that represent the char value within long_val
                        unsigned int mask = 0x000000ff << (8 * long_address_modulo);
                        unsigned int masked_old = long_old & mask;
                        // isolate the bits that represent the char value within long_old, add the long_val to that,
                        // then re-isolate by excluding bits that represent the char value
                        unsigned int overflow = (masked_old + long_val) & ~mask;
                        if (overflow) {
                            atomicSub(base_address, overflow);
                        }
                        return (char) (masked_old >> 8 * long_address_modulo);
                    }
                }
                __global__ void glcm(const int* window_i,
                    const int* window_j,
                    const int maxValue,
                    const int noOfValues,
                    unsigned char* g,
                    float* features) 
                {
                    int tid = threadIdx.x + blockIdx.x * blockDim.x;
                    unsigned char x = 1;
                    // Prevent OOB
                    if (tid < noOfValues){
                        int row = window_i[tid];
                        int col = window_j[tid];
                        atomicAdd(&(g[col + row * (maxValue + 1)]), x);
                    }
                    __syncthreads();

                    // Calculate the GLCM metrics from here         
                    const float i = (float)(tid / (maxValue + 1));
                    const float j = (float)(tid % (maxValue + 1));
            
                    __syncthreads();
                                       
                    // Prevent OOB
                    // As i, j are integers, we avoid float rounding errors
                    // by -0.5
                    // E.g. If i should be 16, it may round down incorrectly
                    // i = 15.9999999
                    // maxValue + 1 = 16          <- Incorrectly passed
                    // maxValue + 1 - 0.5 = 15.5  <- Correctly stopped
                    
                    if (i >= (maxValue + 1 - 0.5)) return;
                    if (j >= (maxValue + 1 - 0.5)) return;
                    
                    float p = (float)(g[tid]) / noOfValues;
                    assert(i < maxValue + 1);
                    assert(j < maxValue + 1);
                    
                    __syncthreads();
                    
                    atomicAdd(
                        &features[HOMOGENEITY], 
                        p / (1 + powf((i - j), 2.0f))
                    );
                    
                    atomicAdd(
                        &features[ASM], 
                        powf(p, 2.0f)
                    );
                    
                    atomicAdd(
                        &features[CONTRAST], 
                        p * powf(i - j, 2.0f)
                    );
                    
                    atomicAdd(
                        &features[MEAN_I], 
                        p * i
                    );
                    
                    atomicAdd(
                        &features[MEAN_J], 
                        p * j
                    );
            
                    __syncthreads();
                    
                    atomicAdd(
                        &features[VAR_I], 
                        p * powf((i - features[MEAN_I]), 2.0f) 
                    );
                    
                    atomicAdd(
                        &features[VAR_J], 
                        p * powf((j - features[MEAN_J]), 2.0f)
                    );
            
                    __syncthreads();
            
                    if (features[VAR_I] == 0 || features[VAR_J] == 0) return;
            
                    atomicAdd(
                        &features[CORRELATION], 
                        p 
                         * (i - features[MEAN_I])
                         * (j - features[MEAN_J]) 
                         * rsqrtf(features[VAR_I] * features[VAR_J])
                    );
                }
            }""",
            "glcm"
        )
#%%

np.random.seed(0)
# im = np.random.randint(0, 128, (50,50), dtype=int)
# im = np.random.randint(1, 256, (200,50), dtype=int)
im = np.linspace(0, 17, 100*25, dtype=int).reshape(100,25)

a = GLCM().from_image(im, radius=1)
import matplotlib.pyplot as plt
#
# plt.imshow(im)
# plt.show()
plt.imshow(a[...,2].get(), cmap='rainbow')
plt.show()


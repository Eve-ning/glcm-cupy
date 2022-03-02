import cupy as cp
from cupy.cuda.texture import TextureObject, CUDAarray
from cupyx.profiler import benchmark

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
        const float i = tid / (maxValue + 1);
        const float j = tid % (maxValue + 1);
        
        // Prevent OOB
        // We should have enough threads up to 256 ^ 2
        if (tid >= (maxValue + 1) * (maxValue + 1)) return;
        
        float g_value = (float)(g[tid]);
        assert(i < maxValue + 1);
        assert(j < maxValue + 1);
        atomicAdd(
            &features[ASM], 
            powf(g_value, 2.0f) 
        );
        atomicAdd(
            &features[CONTRAST], 
            g_value * powf(i - j, 2.0f) 
        );
        atomicAdd(
            &features[HOMOGENEITY], 
            g_value / (1 + powf(i - j, 2.0f)) 
        );
        atomicAdd(
            &features[MEAN_I], 
            g_value * i 
        );
        atomicAdd(
            &features[MEAN_J], 
            g_value * j 
        );
        
        __syncthreads();
        
        atomicAdd(
            &features[VAR_I], 
            g_value * powf((i - features[MEAN_I]), 2.0f) 
        );
        atomicAdd(
            &features[VAR_J], 
            g_value * powf((i - features[MEAN_J]), 2.0f)
        );
        
        if (features[VAR_I] == 0 || features[VAR_J] == 0) return 
        
        __syncthreads();
        
        atomicAdd(
            &features[CORRELATION], 
            g_value * (i - features[MEAN_I]) * (j - features[MEAN_J]) * 
             rsqrtf(features[VAR_I] * features[VAR_J])
        );
    }
}
    """, "glcm",
)
diameter = 3
no_of_values = diameter ** 2
max_value = 5
cp.random.seed(0)
i = cp.random.randint(0, max_value, no_of_values, dtype=cp.int_)
j = cp.random.randint(0, max_value, no_of_values, dtype=cp.int_)

# We may have an intermediate max_value here
# This may reduce the required glcm size slightly?
# May or may not be disruptive
max_value = int(max(cp.max(i), cp.max(j)))
glcm = cp.zeros((max_value + 1) ** 2, dtype=cp.uint8).flatten()

assert max_value < 256, "Max value supported is 255"
assert no_of_values < 256**2, f"Max number of values supported is {256 ** 2 - 1}"

features = cp.zeros(8, dtype=cp.float32)
glcm_kernel(
    grid=(256,),
    block=(256,),
    args=(
        i, j, max_value, no_of_values, glcm, features
    ),
)
glcm_sq = glcm.reshape((max_value + 1), (max_value + 1))

#%%
glcm.sum()

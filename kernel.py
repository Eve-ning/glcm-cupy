import cupy as cp
glcm_module = cp.RawModule(
    code=r"""
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
        __global__ void glcm_0(
            const unsigned char* window_i,
            const unsigned char* window_j,
            const int maxValue,
            const int noOfValues,
            unsigned char* g,
            float* features) 
        {
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned char x = 1;
            // Prevent OOB
            if (tid < noOfValues){
                unsigned char row = window_i[tid];
                unsigned char col = window_j[tid];
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
                &features[CONTRAST], 
                p * powf(i - j, 2.0f)
            );

            atomicAdd(
                &features[ASM], 
                powf(p, 2.0f)
            );

            atomicAdd(
                &features[MEAN_I], 
                p * i
            );

            atomicAdd(
                &features[MEAN_J], 
                p * j
            );
        }
        __global__ void glcm_1(
            const unsigned char* g,
            const int maxValue,
            const int noOfValues,
            float* features) 
        {
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            const float i = (float)(tid / (maxValue + 1));
            const float j = (float)(tid % (maxValue + 1));
            if (i >= (maxValue + 1 - 0.5)) return;
            if (j >= (maxValue + 1 - 0.5)) return;
            float p = (float)(g[tid]) / noOfValues;
            atomicAdd(
                &features[VAR_I], 
                p * powf((i - features[MEAN_I]), 2.0f)
            );

            atomicAdd(
                &features[VAR_J], 
                p * powf((j - features[MEAN_J]), 2.0f)
            );

        }

        __global__ void glcm_2(
            const unsigned char* g,
            const int maxValue,
            const int noOfValues,
            float* features) 
        {
            if (features[VAR_I] == 0 || features[VAR_J] == 0) return;

            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            const float i = (float)(tid / (maxValue + 1));
            const float j = (float)(tid % (maxValue + 1));
            if (i >= (maxValue + 1 - 0.5)) return;
            if (j >= (maxValue + 1 - 0.5)) return;
            float p = (float)(g[tid]) / noOfValues;

            atomicAdd(
                &features[CORRELATION], 
                p * (i - features[MEAN_I]) * (j - features[MEAN_J]) 
                 * rsqrtf(features[VAR_I] * features[VAR_J])
            );
        }
    }"""
)
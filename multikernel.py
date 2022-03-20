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
#define NO_OF_FEATURES 8

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
        const unsigned char* windows_i,
        const unsigned char* windows_j,
        const int glcmSize,
        const int noOfValues,
        const int noOfWindows,
        unsigned char* g,
        float* features)
    {
        /**
        =====================================
        Definitions
        =====================================

        windows_i, windows_j = 2D Array. Shape: (noOfValues, noOfWindows)

        It should be an array of windows to be used

        Take for example the following input:

            +---------+  +---------+  +---------+  +---------+
            |         |  |         |  |         |  |         |
            |  3 x 3  |  |         |  |         |  |         |
            |         |  |         |  |         |  |         |
            +---------+  +---------+  +---------+  +---------+

            +---------+  +---------+  +---------+  +---------+
            |         |  |         |  |         |  |         |
            |         |  |         |  |         |  |         |
            |         |  |         |  |         |  |         |
            +---------+  +---------+  +---------+  +---------+

            We don't require the y dimensions. Thus, [3 x 3] x [4 x 2] = 9 x 8

            windows_i will thus be 9 x 8 large. Though in CuPy, we simply flatten everything anyways.m
            It's a simpler way to represent how it works.

        glcmSize = The maximum value of the input + 1, the maximum GLCM size.

        noOfValues = Number of values per window. In the above, it's 3 x 3 = 9

        noOfWindows = Number of windows in total. In the above, it's 4 x 2 = 8

        g = Empty initialized GLCM array. Shape of (glcmSize, glcmSize, noOfWindows)

        features = Empty initialized feature array. Shape of (8, noOfWindows)

        **/

        /**
        =====================================
        Thread ID Calculation
        =====================================
        We are not interested in matching the block/thread dim to
        any dims of the input. We just want to ensure that we have
        enough threads in total.
        **/

        int blockId = blockIdx.y * gridDim.x + blockIdx.x;
        int tid = blockId * blockDim.x + threadIdx.x;

        /**
        =====================================
        TID to respective windows
        =====================================
        TID is 1D, we need to partition them to their windows
        Each window area is `noOfValues` == diameter ^ 2
        We have total of `noOfWindows` number of windows
        Thus, we simply take `wid = tid / noOfValues`
        WID: Window ID

        Note integer division
        Example:
            1  / 50 = 0
            49 / 50 = 0
            50 / 50 = 1
            51 / 50 = 1

        If we have `noOfWindows == 4`, we reject any results >= 4
        Example:
            1   / 50 = 0 (Accept)
            199 / 50 = 3 (Accept)
            200 / 50 = 4 (Reject)
            201 / 50 = 4 (Reject)

        **/

        const int glcmArea = glcmSize * glcmSize;

        int wid_image = tid / noOfValues;
        const static unsigned char x = 1;
        if (tid < noOfValues * noOfWindows)
        {
            unsigned char row = windows_i[tid];
            unsigned char col = windows_j[tid];
            // Remember that the shape of GLCM is (glcmSize, glcmSize, noOfWindows)
            atomicAdd(&(
                g[
                col +
                row * glcmSize +
                wid_image * glcmArea
                ]), x);
        }
        __syncthreads();

        /**
        ===================================
        TID to respective GLCM i, j, window
        ===================================

        To calculate the window for the tid, we simply divide by the max i & j

        +---------+ +---------+ +---------+
        |         | |         | |         |
        |  i x j  | |         | |         |
        |         | |         | |         |
        +---------+ +---------+ +---------+
        Window 0    Window 1    Window 2    ...

        The area for each GLCM window is glcmSize * glcmSize

        wid = tid / glcmArea
        i = (tid % glcmArea) / glcmSize
        j = (tid % glcmArea) % glcmSize

        **/

        const float wid = (float)(tid / glcmArea)
        if (wid >= noOfWindows) return;

        const float i = (float)((tid % glcmArea) / glcmSize)
        const float j = (float)((tid % glcmArea) % glcmSize)

        float p = (float)(g[tid]) / noOfValues;

        /**
        =====================================
        Feature Calculation
        =====================================
        
        For each feature, we require a wid * NO_OF_FEATURES offset.

        8 x 1 for each GLCM
        +----------------+ +----------------+ +----------------+
        | HOMOGENEITY    | | HOMOGENEITY    | | HOMOGENEITY    |
        | CONTRAST       | | CONTRAST       | | CONTRAST       |
        | ASM            | | ASM            | | ASM            |
        | MEAN_I         | | MEAN_I         | | MEAN_I         |
        | MEAN_J         | | MEAN_J         | | MEAN_J         |
        | VAR_I          | | VAR_I          | | VAR_I          |
        | VAR_J          | | VAR_J          | | VAR_J          |
        | CORRELATION    | | CORRELATION    | | CORRELATION    |
        +----------------+ +----------------+ +----------------+
        Window 0           Window 1           Window 2           ...
        **/
        atomicAdd(
            &features[HOMOGENEITY + wid * NO_OF_FEATURES],
            p / (1 + powf((i - j), 2.0f))
        );

        atomicAdd(
            &features[CONTRAST + wid * NO_OF_FEATURES],
            p * powf(i - j, 2.0f)
        );

        atomicAdd(
            &features[ASM + wid * NO_OF_FEATURES],
            powf(p, 2.0f)
        );

        atomicAdd(
            &features[MEAN_I + wid * NO_OF_FEATURES],
            p * i
        );

        atomicAdd(
            &features[MEAN_J + wid * NO_OF_FEATURES],
            p * j
        );
    }
    __global__ void glcm_1(
        const unsigned char* g,
        const int glcmSize,
        const int noOfValues,
        float* features)
    {
        /**
        =====================================
        Feature Calculation
        =====================================
        
        See above.
        **/
        int blockId = blockIdx.y * gridDim.x + blockIdx.x;
        int tid = blockId * blockDim.x + threadIdx.x;

        const float wid = (float)(tid / glcmArea)
        if (wid >= noOfWindows) return;
        const float i = (float)((tid % glcmArea) / glcmSize)
        const float j = (float)((tid % glcmArea) % glcmSize)

        const int glcmArea = glcmSize * glcmSize;

        float p = (float)(g[tid]) / noOfValues;

        atomicAdd(
            &features[VAR_I + wid * NO_OF_FEATURES],
            p * powf((i - features[MEAN_I + wid * NO_OF_FEATURES]), 2.0f)
        );

        atomicAdd(
            &features[VAR_J + wid * NO_OF_FEATURES],
            p * powf((j - features[MEAN_J + wid * NO_OF_FEATURES]), 2.0f)
        );

    }

    __global__ void glcm_2(
        const unsigned char* g,
        const int glcmSize,
        const int noOfValues,
        float* features)
    {
        /**
        =====================================
        Feature Calculation
        =====================================
        
        See above.
        **/

        int blockId = blockIdx.y * gridDim.x + blockIdx.x;
        int tid = blockId * blockDim.x + threadIdx.x;

        const float wid = (float)(tid / glcmArea)
        if (wid >= noOfWindows) return;
        
        // As we invert Variance, they should never be 0.
        if (features[VAR_I + wid * NO_OF_FEATURES] == 0 ||
            features[VAR_J + wid * NO_OF_FEATURES] == 0) return;

        const float i = (float)((tid % glcmArea) / glcmSize)
        const float j = (float)((tid % glcmArea) % glcmSize)

        const int glcmArea = glcmSize * glcmSize;

        float p = (float)(g[tid]) / noOfValues;

        atomicAdd(
            &features[CORRELATION],
            p * (i - features[MEAN_I + wid * NO_OF_FEATURES])
              * (j - features[MEAN_J + wid * NO_OF_FEATURES])
              * rsqrtf(features[VAR_I + wid * NO_OF_FEATURES]
                     * features[VAR_J + wid * NO_OF_FEATURES])
        );
    }
}
"""
)

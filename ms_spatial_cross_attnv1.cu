#include <iostream>
#include "debug_utils.h"
using namespace std;

#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
         i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 512;

inline int32_t GET_BLOCKS(int32_t const N, int32_t const numThreads)
{
    return (N + numThreads - 1) / numThreads;
}


template <typename scalar_t>
__device__ scalar_t ms_deform_attn_im2col_bilinear(scalar_t const*& bottomData, int32_t const& height, int32_t const& width,
    int32_t const& nHeads, int32_t const& channels, scalar_t const& h, scalar_t const& w, int32_t const& m, int32_t const& c)
{
    int32_t const hLow = floor(h);
    int32_t const wLow = floor(w);
    int32_t const hHigh = hLow + 1;
    int32_t const wHigh = wLow + 1;

    scalar_t const lh = h - hLow;
    scalar_t const lw = w - wLow;
    scalar_t const hh = 1 - lh, hw = 1 - lw;

    int32_t const wStride = nHeads * channels;
    int32_t const hStride = width * wStride;
    int32_t const hLowPtrOffset = hLow * hStride;
    int32_t const hHighPtrOffset = hLowPtrOffset + hStride;
    int32_t const wLowPtrOffset = wLow * wStride;
    int32_t const wHighPtrOffset = wLowPtrOffset + wStride;
    int32_t const basePtr = m * channels + c;

    scalar_t v1 = 0;
    if (hLow >= 0 && wLow >= 0)
    {
        int32_t const ptr1 = hLowPtrOffset + wLowPtrOffset + basePtr;
        v1 = bottomData[ptr1];
    }
    scalar_t v2 = 0;
    if (hLow >= 0 && wHigh <= width - 1)
    {
        int32_t const ptr2 = hLowPtrOffset + wHighPtrOffset + basePtr;
        v2 = bottomData[ptr2];
    }
    scalar_t v3 = 0;
    if (hHigh <= height - 1 && wLow >= 0)
    {
        int32_t const ptr3 = hHighPtrOffset + wLowPtrOffset + basePtr;
        v3 = bottomData[ptr3];
    }
    scalar_t v4 = 0;
    if (hHigh <= height - 1 && wHigh <= width - 1)
    {
        int32_t const ptr4 = hHighPtrOffset + wHighPtrOffset + basePtr;
        v4 = bottomData[ptr4];
    }

    scalar_t const w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

    scalar_t const val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}

template <typename scalar_t>
__global__ void ms_deformable_im2col_gpu_kernel(int32_t const n, scalar_t const* dataValue,
    int32_t const* dataSpatialShapes, int32_t const* dataLevelStartIndex,
    scalar_t const* dataSamplingLoc, scalar_t const* dataAttnWeight, scalar_t const* camMask,
    int32_t const batchSize, int32_t const numCam, int32_t const spatialSize,
    int32_t const numHeads, int32_t const channels, int32_t const numLevels,
    int32_t const numQuery, int32_t const numPoint, int32_t const numAnchor,
    scalar_t* dataCol)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        int32_t _temp = index;
        int32_t const cCol = _temp % channels;
        _temp /= channels;
        int32_t const samplingIndex = _temp;
        int32_t const mCol = _temp % numHeads;
        _temp /= numHeads;
        int32_t const qCol = _temp % numQuery;
        _temp /= numQuery;
        int32_t const camCol = _temp % numCam;
        _temp /= numCam;
        int32_t const bCol = _temp;

        scalar_t* dataColPtr = dataCol + index;
        int32_t dataWeightPtr = samplingIndex * numLevels * numAnchor * numPoint;
        int32_t dataLocWPtr = dataWeightPtr << 1;
        int32_t const qidStride = numHeads * channels;
        // int32_t const numPoints = numAnchor * numPoint;
        int32_t const dataValuePtrInitOffset = bCol * camCol * spatialSize * qidStride;
        scalar_t col = 0;

        int32_t dataMaskPtr = qCol * numAnchor;

        for (int32_t lCol = 0; lCol < numLevels; ++lCol)
        {
            int32_t const levelStartId = dataLevelStartIndex[lCol];
            int32_t const spatialHPtr = lCol << 1;
            int32_t const spatialH = dataSpatialShapes[spatialHPtr];
            int32_t const spatialW = dataSpatialShapes[spatialHPtr + 1];
            scalar_t const* dataValuePtr = dataValue + (dataValuePtrInitOffset + levelStartId * qidStride);
            for (int32_t aCol = 0; aCol < numAnchor; ++aCol)
            {
                scalar_t const mask = camMask[dataMaskPtr];
                for (int32_t pCol = 0; pCol < numPoint; ++pCol)
                {
                    scalar_t const locW = dataSamplingLoc[dataLocWPtr];
                    scalar_t const locH = dataSamplingLoc[dataLocWPtr + 1];
                    scalar_t const weight = dataAttnWeight[dataWeightPtr];

                    scalar_t const hIm = locH * spatialH - 0.5;
                    scalar_t const wIm = locW * spatialW - 0.5;
                    if (hIm > -1 && wIm > -1 && hIm < spatialH && wIm < spatialW && mask > 0)
                    {
                        col += ms_deform_attn_im2col_bilinear(
                                dataValuePtr, spatialH, spatialW, numHeads, channels, hIm, wIm, mCol, cCol)
                            * weight;
                    }

                    dataWeightPtr += 1;
                    dataLocWPtr += 2;
                }
                dataMaskPtr += 1;
            }
        }
        *dataColPtr = col;
    }
}


int main() {
    using namespace debug;
    typedef float scalar_t;
    int32_t const batchSize = 2;
    int32_t const numCam = 3;
    int32_t const spatialSize = 1025;
    int32_t const numHeads = 2;
    int32_t const channels = 5;
    int32_t const numLevels = 4;
    int32_t const numQuery = 30;
    int32_t const numAnchor = 2;
    int32_t const numPoint = 5;
    
    const int32_t DATA_ARRAY_SIZE = batchSize*numCam*spatialSize*numHeads*channels;
    const int32_t LOC_ARRAY_SIZE = batchSize*numCam*numQuery*numHeads*numLevels*numAnchor*numPoint*2;
    const int32_t MASK_ARRAY_SIZE = batchSize*numCam*numQuery*numAnchor;

    scalar_t dataValue[DATA_ARRAY_SIZE];
    int32_t dataSpatialShapes[numLevels][2]={{25, 25}, {20, 20}};
    int32_t dataLevelStartIndex[numLevels]={0, 625};
    scalar_t dataSamplingLoc[LOC_ARRAY_SIZE];
    scalar_t dataAttnWeight[LOC_ARRAY_SIZE>>1];
    scalar_t dataMask[MASK_ARRAY_SIZE];
    
    // init dataValue
    int idx = 0;
    for(int i = 0; i < batchSize; ++i)
    {
        for(int j = 0; j < numCam; ++j)
        {
            for(int k=0; k<spatialSize; ++k)
            {
                for(int k1=0; k1<numHeads; ++k1)
                {
                    for(int k2=0; k2<channels; ++k2)
                    {
                        dataValue[idx] = j * k1;
                        ++idx;
                    }
                }
            }
        }
    }

    // initial dataSamplingLoc
    int idx1 = 0;
    for(int i = 0; i < batchSize; ++i)
    {
        for(int j = 0; j < numCam; ++j)
        {
            for(int k=0; k<numQuery; ++k)
            {
                for(int k1=0; k1<numHeads; ++k1)
                {
                    for(int k2=0; k2<numLevels; ++k2)
                    {
                        for(int k22=0; k22<numAnchor*numPoint; ++k22)
                        {
                            for(int k3=0; k3<2; ++k3)
                            {
                                dataSamplingLoc[idx1] = 0.2 * (k3+1); // [0.2, 0.4]
                                ++idx1;
                            }
                        }
                    }
                }
            }
        }
    }

    // for(int i=0; i<LOC_ARRAY_SIZE; ++i)
    // {
    //     dataSamplingLoc[i] = 0.5f + (float)random() / RAND_MAX;
    //     // printf("%f ",dataSamplingLoc[i]);
    // }
    for(int i=0; i<LOC_ARRAY_SIZE/2; ++i)
    {
        dataAttnWeight[i] = 0.5f + (float)random() / RAND_MAX;
    }
    // init dataMask
    int idx2 = 0;
    for(int i = 0; i < batchSize; ++i)
    {
        for(int j = 0; j < numCam; ++j)
        {
            for(int k=0; k<numQuery; ++k)
            {
                for(int k1=0; k1<numAnchor; ++k1)
                {   
                    if (k1 == 0)
                    {
                        dataMask[idx2] = 1.f;
                    }
                    else
                    {
                        dataMask[idx2] = 0.f;
                    }
                    ++idx2;
                }
            }
        }
    }

    // for(int i=0; i<MASK_ARRAY_SIZE; ++i)
    // {
    //     dataMask[i] = 0.;
    // }
    
    scalar_t h_dataCol[batchSize * numCam * numQuery * numHeads * channels];

    scalar_t* d_dataValue;
    scalar_t* d_dataSamplingLoc;
    scalar_t* d_dataAttnWeight;
    scalar_t* d_dataCol;
    scalar_t* d_dataMask;
    int32_t* d_dataSpatialShapes;
    int32_t* d_dataLevelStartIndex;

    cudaMalloc((void**) &d_dataValue, DATA_ARRAY_SIZE*sizeof(scalar_t));
    cudaMalloc((void**) &d_dataSamplingLoc, LOC_ARRAY_SIZE*sizeof(scalar_t));
    cudaMalloc((void**) &d_dataAttnWeight, LOC_ARRAY_SIZE/2*sizeof(scalar_t));
    cudaMalloc((void**) &d_dataSpatialShapes, numLevels*2*sizeof(int32_t));
    cudaMalloc((void**) &d_dataLevelStartIndex, numLevels*sizeof(int32_t));
    cudaMalloc((void**) &d_dataMask, MASK_ARRAY_SIZE*sizeof(scalar_t));

    cudaMalloc((void**) &d_dataCol, batchSize*numCam*numQuery*numHeads*channels*sizeof(scalar_t));

    cudaMemcpy(d_dataValue, dataValue, DATA_ARRAY_SIZE*sizeof(scalar_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataSamplingLoc, dataSamplingLoc, LOC_ARRAY_SIZE*sizeof(scalar_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataAttnWeight, dataAttnWeight, LOC_ARRAY_SIZE/2*sizeof(scalar_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataSpatialShapes, dataSpatialShapes, numLevels*2*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataLevelStartIndex, dataLevelStartIndex, numLevels*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataMask, dataMask, MASK_ARRAY_SIZE*sizeof(scalar_t), cudaMemcpyHostToDevice);

    int32_t const numKernels = batchSize * numCam * numQuery * numHeads * channels;
    int32_t const numActualKernels = batchSize * numCam * numQuery * numHeads * channels;
    int32_t const numThreads = CUDA_NUM_THREADS;

    std::cout << "The numKernel size is: " << numKernels << std::endl;

    ms_deformable_im2col_gpu_kernel<scalar_t><<<GET_BLOCKS(numActualKernels, numThreads), numThreads, 0>>>(
        numKernels, d_dataValue, d_dataSpatialShapes, d_dataLevelStartIndex, d_dataSamplingLoc,
        d_dataAttnWeight, d_dataMask, batchSize, numCam, spatialSize, numHeads, channels,
        numLevels, numQuery, numPoint, numAnchor, d_dataCol);
    
    cudaMemcpy(h_dataCol, d_dataCol, numKernels*sizeof(scalar_t), cudaMemcpyDeviceToHost);
    for(int i=0;i<numKernels;i++){
        printf("%f ",h_dataCol[i]);
        // printf(((i%4) != 3) ? "\t" : "\n");
      }
    cudaFree(d_dataValue);
    cudaFree(d_dataSamplingLoc);
    cudaFree(d_dataAttnWeight);
    cudaFree(d_dataSpatialShapes);
    cudaFree(d_dataLevelStartIndex);
    cudaFree(d_dataCol);
    checkCudaErrors(cudaGetLastError());
    std::cout << "Execution success" << std::endl;
    return 0;
}

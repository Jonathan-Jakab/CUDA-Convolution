#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <wb.h>

#define MASK_WIDTH 5
#define O_TILE_WIDTH 16
#define clamp(x) (min(max((x), 0.0), 1.0))

#define MASK_RADIUS (MASK_WIDTH/2)
#define I_TILE_WIDTH (O_TILE_WIDTH + MASK_WIDTH - 1)


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//implement the tiled 2D convolution kernel with adjustments for channels
//use shared memory to reduce the number of global accesses, handle the boundary conditions when loading input list elements into the shared memory
//clamp your output values

__global__ void convolution_2D(float* I, const float* __restrict__ M, float* P,
    int channels, int width, int height) {

    __shared__ float N_ds[I_TILE_WIDTH][I_TILE_WIDTH];

    for (int k = 0; k < channels; k++) {

        int outputTile = threadIdx.y * O_TILE_WIDTH + threadIdx.x;
        int outputTileX = outputTile % I_TILE_WIDTH;
        int outputTileY = outputTile / I_TILE_WIDTH;
        int inputTileY = blockIdx.y * O_TILE_WIDTH + outputTileY - MASK_RADIUS;
        int inputTileX = blockIdx.x * O_TILE_WIDTH + outputTileX - MASK_RADIUS;
        int inputTile = (inputTileY * width + inputTileX) * channels + k;

        if (inputTileX >= 0 && inputTileX < width && inputTileY >= 0 && inputTileY < height) {

            N_ds[outputTileY][outputTileX] = I[inputTile];
        }
        else {

            N_ds[outputTileY][outputTileX] = 0;
        }

        outputTile = threadIdx.y * O_TILE_WIDTH + threadIdx.x + O_TILE_WIDTH * O_TILE_WIDTH;
        outputTileX = outputTile % I_TILE_WIDTH;
        outputTileY = outputTile / I_TILE_WIDTH;
        inputTileY = blockIdx.y * O_TILE_WIDTH + outputTileY - MASK_RADIUS;
        inputTileX = blockIdx.x * O_TILE_WIDTH + outputTileX - MASK_RADIUS;
        inputTile = (inputTileY * width + inputTileX) * channels + k;

        if (outputTileY < I_TILE_WIDTH) {

;           if (inputTileX >= 0 && inputTileX < width && inputTileY >= 0 && inputTileY < height) {

                N_ds[outputTileY][outputTileX] = I[inputTile];
            }
            else {

                N_ds[outputTileY][outputTileX] = 0;
            }
        }

        __syncthreads();

        float accum = 0;
        for (int y = 0; y < MASK_WIDTH; y++) {

            for (int x = 0; x < MASK_WIDTH; x++) {

                accum += N_ds[threadIdx.y + y][threadIdx.x + x] * M[y * MASK_WIDTH + x];
            }
        }

        int outX = blockIdx.x * O_TILE_WIDTH + threadIdx.x;
        int outY = blockIdx.y * O_TILE_WIDTH + threadIdx.y;

        if (outX < width && outY < height) {

            P[(outY * width + outX) * channels + k] = clamp(accum, 0, 1);
        }

        __syncthreads();
    }
}


////////////////////////////////////////////////PSEUDO CODE////////////////////////////////////////////////
//    for (int i = 0; i < height; i++) {
//        for (int j = 0; j < width; j++) {
//            for (int k = 0; k < channels; k++) {
//                float accum = 0;
//
//                for (int y = -1 * MASK_RADIUS; y < MASK_RADIUS; y++) {
//                    for (int x = -1 * MASK_RADIUS; x < MASK_RADIUS; x++) {
//                        int xOffset = j + x;
//                        int yOffset = i + y;
//                        if (xOffset >= 0 && xOffset < width && yOffset >= 0 && yOffset < height) {
//                            float imagePixel = I[(yOffset * width + xOffset) * channels + k];
//                            float maskValue = M[(y + MASK_RADIUS) * MASK_WIDTH + x + MASK_RADIUS];
//                            accum += imagePixel * maskValue;
//                        }
//                    }
//                }
//                P[(i * width + j) * channels + k] = clamp(accum, 0, 1)
//            }
//        }
//    }
//}


int main(int argc, char* argv[]) {
    wbArg_t arg;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char* inputImageFile;
    char* inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float* hostInputImageData;
    float* hostOutputImageData;
    float* hostMaskData;
    float* deviceInputImageData;
    float* deviceOutputImageData;
    float* deviceMaskData;

    arg = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(arg, 0);
    inputMaskFile = wbArg_getInputFile(arg, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float*)wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == MASK_WIDTH);    /* mask height is fixed to 5 */
    assert(maskColumns == MASK_WIDTH); /* mask width is fixed to 5 */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");

    //allocate device memory
    cudaMalloc((void**) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void**) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void**) &deviceMaskData, maskRows * maskColumns * sizeof(float));

    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");

    //copy host memory to device
    cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData, hostMaskData, maskRows * maskColumns * sizeof(float), cudaMemcpyHostToDevice);

    wbTime_stop(Copy, "Copying data to the GPU");

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //initialize thread block and kernel grid dimensions
    dim3 dimGrid(ceil((float)imageWidth / O_TILE_WIDTH), ceil((float)imageHeight / O_TILE_WIDTH));
    dim3 dimBlock(O_TILE_WIDTH, O_TILE_WIDTH, 1);

    //invoke CUDA kernel
    convolution_2D<<<dimGrid, dimBlock >> > (deviceInputImageData, deviceMaskData, deviceOutputImageData, imageChannels, imageWidth, imageHeight);
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    wbTime_stop(Compute, "Doing the computation on the GPU");

    wbTime_start(Copy, "Copying data from the GPU");

    //copy results from device to host
    cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(arg, outputImage);

    //deallocate device memory
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}

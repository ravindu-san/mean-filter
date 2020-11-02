#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void meanFilterCPU(unsigned char *image, unsigned char *filteredImage, int imgWidth, int imgHeight, int window)
{

    int bottomBoundaryOfWindow, topBoundaryOfWindow, leftBoundaryOfWindow, rightBoundaryOfWindow;
    int halfOfWindowSize = (window - 1) / 2;

    for (size_t i = 0; i < imgHeight; i++)
    {

        // printf("h : %d\n",i);
        int calculatedBottomBoundary = i - halfOfWindowSize;
        int calculatedTopBoundary = i + halfOfWindowSize;

        bottomBoundaryOfWindow = (calculatedBottomBoundary <= 0) ? 0 : calculatedBottomBoundary;
        topBoundaryOfWindow = (calculatedTopBoundary >= (imgHeight - 1)) ? (imgHeight - 1) : calculatedTopBoundary;

        for (size_t j = 0; j < imgWidth; j++)
        {

            int calculatedLeftBoundary = j - halfOfWindowSize;
            int calculatedRightBoundary = j + halfOfWindowSize;

            leftBoundaryOfWindow = (calculatedLeftBoundary <= 0) ? 0 : calculatedLeftBoundary;
            rightBoundaryOfWindow = (calculatedRightBoundary >= (imgWidth - 1)) ? (imgWidth - 1) : calculatedRightBoundary;

            int sum = 0;

            for (size_t y = bottomBoundaryOfWindow; y <= topBoundaryOfWindow; y++)
            {

                for (size_t x = leftBoundaryOfWindow; x <= rightBoundaryOfWindow; x++)
                {
                    sum += image[y * imgWidth + x];
                }
            }

            int pixelsInWindow = (rightBoundaryOfWindow - leftBoundaryOfWindow + 1) * (topBoundaryOfWindow - bottomBoundaryOfWindow + 1);
            filteredImage[i * imgWidth + j] = sum / pixelsInWindow;
        }
    }
}

__global__ void meanFilterGPU(unsigned char *image, unsigned char *filteredImage, int imgWidth, int imgHeight, int window){

    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int halfOfWindowSize = window/2;

    int bottomBoundaryOfWindow, topBoundaryOfWindow, leftBoundaryOfWindow, rightBoundaryOfWindow;

    int calculatedBottomBoundary = row - halfOfWindowSize;
    int calculatedTopBoundary = row + halfOfWindowSize;
    int calculatedLeftBoundary = column - halfOfWindowSize;
    int calculatedRightBoundary = column + halfOfWindowSize;

    bottomBoundaryOfWindow = (calculatedBottomBoundary < 0) ? 0 : calculatedBottomBoundary;
    topBoundaryOfWindow = (calculatedTopBoundary > (imgHeight-1)) ? (imgHeight-1) : calculatedTopBoundary;
    leftBoundaryOfWindow = (calculatedLeftBoundary < 0) ? 0 : calculatedLeftBoundary;
    rightBoundaryOfWindow = (calculatedRightBoundary > (imgWidth-1)) ? (imgWidth-1) : calculatedRightBoundary;

    int sum = 0;

    for(int y = bottomBoundaryOfWindow; y <= topBoundaryOfWindow; y++){

        for(int x = leftBoundaryOfWindow; x <= rightBoundaryOfWindow; x++){

            sum += image[ y*imgWidth + x];

        }
    } 

    int pixelsInWindow = (rightBoundaryOfWindow - leftBoundaryOfWindow + 1) * (topBoundaryOfWindow - bottomBoundaryOfWindow + 1);
    filteredImage[row * imgWidth + column] = sum/pixelsInWindow;
}

int main(int argc, char **argv)
{

    unsigned char *bitmapHeaders, *imgPixels, *cpuFilteredImg, *gpuFilteredImg_d, *gpuFilteredImg_h, *img_d;

    int imgWidth, imgHeight, offset, imgSize, window;
    short bitsPerPixel;

    bitmapHeaders = (unsigned char *)malloc(sizeof(char) * 54);

    FILE *imgFile = fopen("512.bmp", "rb");
    window = 3;
    // FILE *imgFile = fopen(argv[1], "rb");
    // window = atoi(argv[2]);

    //read bitmap image headers to get imgWidth and imgHeight of the image
    //imgWidth is 4 byte and starts @ 19th byte of header.
    //imgHeight is 4 byte and starts @ 23rd byte of header.

    fread(bitmapHeaders, sizeof(unsigned char), 54, imgFile);

    memcpy(&imgWidth, bitmapHeaders + 18, sizeof(int));
    memcpy(&imgHeight, bitmapHeaders + 22, sizeof(int));
    memcpy(&bitsPerPixel, bitmapHeaders + 28, sizeof(short));
    memcpy(&offset, bitmapHeaders + 10, sizeof(int));

    imgSize = imgWidth * imgHeight;

    printf("imgWidtht : %d\n", imgWidth);
    printf("imgHeight : %d\n", imgHeight);
    printf("bitsPerPixel : %d\n", bitsPerPixel);
    printf("image size : %d\n", imgSize);
    printf("offset : %d\n", offset);

    int diffBtwnHeadersAndPixels = offset - 54;
    char *bytsBtwnHeadersAndPixels = (char *)malloc(sizeof(char) * diffBtwnHeadersAndPixels);
    fread(bytsBtwnHeadersAndPixels, sizeof(char), diffBtwnHeadersAndPixels, imgFile);

    imgPixels = (unsigned char *)malloc(sizeof(unsigned char) * imgSize);
    fread(imgPixels, sizeof(char), imgSize, imgFile);

    // printf("size of array : %d\n", sizeof(imgPixels));

    // for (size_t i = 0; i < imgSize; i++)
    // {
    //     printf("%d  %d\n", i,imgPixels[i]);
    // }

    cpuFilteredImg = (unsigned char *)malloc(sizeof(unsigned char) * imgSize);

    meanFilterCPU(imgPixels, cpuFilteredImg, imgWidth, imgHeight, window);


    gpuFilteredImg_h = (unsigned char *)malloc(sizeof(unsigned char) * imgSize);

    cudaMalloc((void **)&img_d, imgSize);
    cudaMalloc((void **)&gpuFilteredImg_d, imgSize);
    cudaMemcpy(img_d, imgPixels, imgSize, cudaMemcpyHostToDevice);
    
    dim3 dimBlock(32, 32);
    dim3 dimGrid(imgWidth/32, imgHeight/32);

    meanFilterGPU<<<dimGrid, dimBlock>>>(img_d, gpuFilteredImg_d, imgWidth, imgHeight, window);


    cudaMemcpy(gpuFilteredImg_h, gpuFilteredImg_d, imgSize, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < imgSize; i++)
    {
        printf("%d  %d  %d  %d\n", i, imgPixels[i], cpuFilteredImg[i], gpuFilteredImg_h[i]);
    }



    fclose(imgFile);

    free(imgPixels);
    free(bytsBtwnHeadersAndPixels);
    free(bitmapHeaders);
    free(cpuFilteredImg);
    free(gpuFilteredImg_h);
    cudaFree(img_d);
    cudaFree(gpuFilteredImg_d);

    return 0;
}
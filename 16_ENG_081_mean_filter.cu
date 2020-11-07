#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void meanFilterCPU(unsigned char *image, unsigned char *filteredImage, int imgWidth, int imgHeight, short bitsPerPixel, int window)
{

    int bottomBoundaryOfWindow, topBoundaryOfWindow, leftBoundaryOfWindow, rightBoundaryOfWindow;
    int halfOfWindowSize = (window - 1) / 2;

    for (size_t i = 0; i < imgHeight; i++)
    {

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

                    if (bitsPerPixel == 8)
                    {
                        sum += image[y * imgWidth + x];
                    }
                    else if (bitsPerPixel == 24)
                    {

                        int possition = (y * imgWidth + x) * 3;

                        unsigned char firstByteOfPixel = image[possition];
                        unsigned char secondByteOfPixel = image[possition + 1];
                        unsigned char thirdByteOfPixel = image[possition + 2];

                        int grayscaleValue = ((firstByteOfPixel << 16) & 0x00ff0000) | ((secondByteOfPixel << 8) & 0x0000ff00) | (thirdByteOfPixel & 0x000000ff);
                        sum += grayscaleValue;
                    }
                }
            }

            int pixelsInWindow = (rightBoundaryOfWindow - leftBoundaryOfWindow + 1) * (topBoundaryOfWindow - bottomBoundaryOfWindow + 1);
            int meanValue = sum / pixelsInWindow;

            if (bitsPerPixel == 8)
            {
                int possitionInImg = i * imgWidth + j;
                filteredImage[possitionInImg] = meanValue;
            }
            else if (bitsPerPixel == 24)
            {

                int possitionInImg = (i * imgWidth + j) * 3;

                unsigned char firstByteOfPixel = (meanValue >> 16) & 0Xff;
                unsigned char secondByteOfPixel = (meanValue >> 8) & 0xff;
                unsigned char thirdByteOfPixel = meanValue & 0xff;

                filteredImage[possitionInImg] = firstByteOfPixel;
                filteredImage[possitionInImg + 1] = secondByteOfPixel;
                filteredImage[possitionInImg + 2] = thirdByteOfPixel;
            }
        }
    }
}

__global__ void meanFilterGPU(unsigned char *image, unsigned char *filteredImage, int imgWidth, int imgHeight, short bitsPerPixel, int window)
{

    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (column < imgWidth && row < imgHeight)
    {

        int halfOfWindowSize = window / 2;

        int bottomBoundaryOfWindow, topBoundaryOfWindow, leftBoundaryOfWindow, rightBoundaryOfWindow;

        int calculatedBottomBoundary = row - halfOfWindowSize;
        int calculatedTopBoundary = row + halfOfWindowSize;
        int calculatedLeftBoundary = column - halfOfWindowSize;
        int calculatedRightBoundary = column + halfOfWindowSize;

        bottomBoundaryOfWindow = (calculatedBottomBoundary < 0) ? 0 : calculatedBottomBoundary;
        topBoundaryOfWindow = (calculatedTopBoundary > (imgHeight - 1)) ? (imgHeight - 1) : calculatedTopBoundary;
        leftBoundaryOfWindow = (calculatedLeftBoundary < 0) ? 0 : calculatedLeftBoundary;
        rightBoundaryOfWindow = (calculatedRightBoundary > (imgWidth - 1)) ? (imgWidth - 1) : calculatedRightBoundary;

        int sum = 0;

        for (int y = bottomBoundaryOfWindow; y <= topBoundaryOfWindow; y++)
        {

            for (int x = leftBoundaryOfWindow; x <= rightBoundaryOfWindow; x++)
            {

                if (bitsPerPixel == 8)
                {
                    sum += image[y * imgWidth + x];
                }
                else if (bitsPerPixel == 24)
                {

                    int possition = (y * imgWidth + x) * 3;

                    unsigned char firstByteOfPixel = image[possition];
                    unsigned char secondByteOfPixel = image[possition + 1];
                    unsigned char thirdByteOfPixel = image[possition + 2];

                    int grayscaleValue = ((firstByteOfPixel << 16) & 0x00ff0000) | ((secondByteOfPixel << 8) & 0x0000ff00) | (thirdByteOfPixel & 0x000000ff);
                    sum += grayscaleValue;
                }
            }
        }

        int pixelsInWindow = (rightBoundaryOfWindow - leftBoundaryOfWindow + 1) * (topBoundaryOfWindow - bottomBoundaryOfWindow + 1);
        int meanValue = sum / pixelsInWindow;

        if (bitsPerPixel == 8)
        {

            int possitionInImg = row * imgWidth + column;
            filteredImage[possitionInImg] = meanValue;
        }
        else if (bitsPerPixel == 24)
        {

            int possitionInImg = (row * imgWidth + column) * 3;

            unsigned char firstByteOfPixel = (meanValue >> 16) & 0Xff;
            unsigned char secondByteOfPixel = (meanValue >> 8) & 0xff;
            unsigned char thirdByteOfPixel = meanValue & 0xff;

            filteredImage[possitionInImg] = firstByteOfPixel;
            filteredImage[possitionInImg + 1] = secondByteOfPixel;
            filteredImage[possitionInImg + 2] = thirdByteOfPixel;
        }
    }
}

int main(int argc, char **argv)
{

    unsigned char *bitmapHeaders, *imgPixels, *cpuFilteredImg, *gpuFilteredImg_d, *gpuFilteredImg_h, *img_d;

    int imgWidth, imgHeight, offset, imgSize, window;
    short bitsPerPixel;

    bitmapHeaders = (unsigned char *)malloc(sizeof(char) * 54);

    // FILE *imgFile = fopen("512.bmp", "rb");
    // FILE *imgFile = fopen("img_640.bmp", "rb");
    // window = 3;

    FILE *imgFile = fopen(argv[1], "rb");
    window = atoi(argv[2]);

    //read bitmap image headers to get imgWidth and imgHeight of the image
    //imgWidth is 4 byte and starts @ 19th byte of header.
    //imgHeight is 4 byte and starts @ 23rd byte of header.

    fread(bitmapHeaders, sizeof(unsigned char), 54, imgFile);

    memcpy(&imgWidth, bitmapHeaders + 18, sizeof(int));
    memcpy(&imgHeight, bitmapHeaders + 22, sizeof(int));
    memcpy(&bitsPerPixel, bitmapHeaders + 28, sizeof(short));
    memcpy(&offset, bitmapHeaders + 10, sizeof(int));

    if (bitsPerPixel == 8)
    {
        imgSize = imgWidth * imgHeight;
    }
    else if (bitsPerPixel == 24)
    {
        imgSize = 3 * imgWidth * imgHeight;
    }

    printf("imgWidtht : %d\n", imgWidth);
    printf("imgHeight : %d\n", imgHeight);
    printf("bitsPerPixel : %d\n", bitsPerPixel);
    printf("image size : %d\n", imgSize);
    printf("offset : %d\n", offset);

    int diffBtwnHeadersAndPixels = offset - 54;
    char *bytsBtwnHeadersAndPixels = (char *)malloc(sizeof(char) * diffBtwnHeadersAndPixels);

    fread(bytsBtwnHeadersAndPixels, sizeof(char), diffBtwnHeadersAndPixels, imgFile); //skip bytes between headers and image pixels

    imgPixels = (unsigned char *)malloc(sizeof(unsigned char) * imgSize);

    fread(imgPixels, sizeof(char), imgSize, imgFile);

    cpuFilteredImg = (unsigned char *)malloc(sizeof(unsigned char) * imgSize);
    gpuFilteredImg_h = (unsigned char *)malloc(sizeof(unsigned char) * imgSize);

    cudaMalloc((void **)&img_d, imgSize);
    cudaMalloc((void **)&gpuFilteredImg_d, imgSize);
    cudaMemcpy(img_d, imgPixels, imgSize, cudaMemcpyHostToDevice);

    meanFilterCPU(imgPixels, cpuFilteredImg, imgWidth, imgHeight, bitsPerPixel, window);

    dim3 dimBlock(32, 32);
    dim3 dimGrid(imgWidth / 32, imgHeight / 32);

    meanFilterGPU<<<dimGrid, dimBlock>>>(img_d, gpuFilteredImg_d, imgWidth, imgHeight, bitsPerPixel, window);

    cudaMemcpy(gpuFilteredImg_h, gpuFilteredImg_d, imgSize, cudaMemcpyDeviceToHost);

    if (bitsPerPixel == 8)
    {

        for (size_t i = 0; i < imgSize; i++)
        {
            printf("%d  pixelBeforeFilter:%d  cpuFilteredPixel:%d  gpuFilteredPixel:%d\n", i, imgPixels[i], cpuFilteredImg[i], gpuFilteredImg_h[i]);
        }
    }
    else if (bitsPerPixel == 24)
    {

        for (int i = 0; i < imgSize; i += 3)
        {

            int pixelBeforeFilter = (imgPixels[i] << 16) & 0X00ff0000 | (imgPixels[i + 1] << 8) & 0X0000ff00 | imgPixels[i + 2] & 0X000000ff;
            int cpuFilteredPixel = (cpuFilteredImg[i] << 16) & 0X00ff0000 | (cpuFilteredImg[i + 1] << 8) & 0X0000ff00 | cpuFilteredImg[i + 2] & 0X000000ff;
            int gpuFilteredPixel = (gpuFilteredImg_h[i] << 16) & 0X00ff0000 | (gpuFilteredImg_h[i + 1] << 8) & 0X0000ff00 | gpuFilteredImg_h[i + 2] & 0X000000ff;

            printf("%d  pixelBeforeFilter:%d  cpuFilteredPixel:%d  gpuFilteredPixel:%d\n", i, pixelBeforeFilter, cpuFilteredPixel, gpuFilteredPixel);
        }
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
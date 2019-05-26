#include <iostream>
#include <math.h>
#include <thread>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <time.h>
#include <chrono>


/*
Compiling: Requires a Nvidia CUDA capable graphics card and the Nvidia GPU Computing Toolkit.
 *            Linux: nvcc main.cu -o test -Xcompiler -fopenmp `pkg-config --cflags --libs opencv`
 */

#define GRIDVAL 16.0
typedef unsigned char byte;

void sobel_cpu(const cv::Mat* orig_gs, cv::Mat* edges_cpu, const unsigned int width, const unsigned int height);
void sobel_omp(const cv::Mat* orig_gs, cv::Mat* edges_omp, const unsigned int width, const unsigned int height);

__global__ void sobel_gpu(const byte* orig, byte* gpu, const unsigned int width, const unsigned int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    float dx, dy;
    if( x > 0 && y > 0 && x < width-1 && y < height-1) {
        dx = (-1* orig[(y-1)*width + (x-1)]) + (-2*orig[y*width+(x-1)]) + (-1*orig[(y+1)*width+(x-1)]) +
             (    orig[(y-1)*width + (x+1)]) + ( 2*orig[y*width+(x+1)]) + (   orig[(y+1)*width+(x+1)]);
        dy = (    orig[(y-1)*width + (x-1)]) + ( 2*orig[(y-1)*width+x]) + (   orig[(y-1)*width+(x+1)]) +
             (-1* orig[(y+1)*width + (x-1)]) + (-2*orig[(y+1)*width+x]) + (-1*orig[(y+1)*width+(x+1)]);
        gpu[y*width + x] = sqrt( (dx*dx) + (dy*dy) );
    }
}
int main (int argc, char* argv[])
{
    try
    {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, 0);
        int cores = devProp.multiProcessorCount;
        switch (devProp.major)
        {
        case 2: // Fermi
            if (devProp.minor == 1) cores *= 48;
            else cores *= 32; break;
        case 3: // Kepler
            cores *= 192; break;
        case 5: // Maxwell
            cores *= 128; break;
        case 6: // Pascal
            if (devProp.minor == 1) cores *= 128;
            else if (devProp.minor == 0) cores *= 64;
            break;
        }
        time_t rawTime;time(&rawTime);
        struct tm* curTime = localtime(&rawTime);
        char timeBuffer[80] = "";
        strftime(timeBuffer, 80, "edge map benchmarks (%c)\n", curTime);
        printf("%s", timeBuffer);
        printf("CPU: %d hardware threads\n", std::thread::hardware_concurrency());
        printf("GPGPU: %s, CUDA %d.%d, %zd Mbytes global memory, %d CUDA cores\n",
        devProp.name, devProp.major, devProp.minor, devProp.totalGlobalMem / 1048576, cores);

        cv::VideoCapture camera(0);
        if(!camera.isOpened())
            return -1;

        cv::namedWindow("Result");
        cv::resizeWindow("Result", 640, 480);
        camera.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        camera.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cv::Mat* orig, *orig_gs, *edges_cpu, *edges_omp, *edges_gpu;
        unsigned int width, height = 0;
        width = camera.get(cv::CAP_PROP_FRAME_WIDTH);
        height = camera.get(cv::CAP_PROP_FRAME_HEIGHT);
        
        orig = new cv::Mat(height,width,CV_8UC3);
        orig_gs = new cv::Mat(height,width,CV_8UC1);
        edges_cpu = new cv::Mat(height,width,CV_8UC1);
        edges_omp = new cv::Mat(height,width,CV_8UC1);
        edges_gpu = new cv::Mat(height,width,CV_8UC1);
        byte *gpu_orig, *gpu_sobel;
        cudaMalloc((void**)&gpu_orig,(width*height*sizeof(byte)));
        cudaMalloc((void**)&gpu_sobel,(width*height*sizeof(byte)));
        dim3 threadsPerBlock(GRIDVAL, GRIDVAL, 1);
        dim3 numBlocks(ceil(width/GRIDVAL), ceil(height/GRIDVAL), 1);
        

        for(;;){
            camera >> *orig;
            cv::cvtColor(*orig, *orig_gs, CV_BGR2GRAY, 0);
            auto c = std::chrono::system_clock::now();
            sobel_cpu(orig_gs, edges_cpu , width, height);
            std::chrono::duration<double> time_cpu = std::chrono::system_clock::now() - c;

            c = std::chrono::system_clock::now();
            sobel_omp(orig_gs, edges_omp , width, height);
            std::chrono::duration<double> time_omp = std::chrono::system_clock::now() - c;
            
            c = std::chrono::system_clock::now();
            cudaError_t error = cudaMemcpy(gpu_orig, orig_gs->data, (width*height*sizeof(byte)), cudaMemcpyHostToDevice);
            cudaMemset(gpu_sobel, 0, (width*height*sizeof(byte)));
            sobel_gpu<<<numBlocks, threadsPerBlock>>>(gpu_orig, gpu_sobel, width, height);
            error = cudaDeviceSynchronize(); // waits for completion, returns error code
            error = cudaMemcpy(edges_gpu->data, gpu_sobel, (width*height), cudaMemcpyDeviceToHost);
            std::chrono::duration<double> time_gpu = std::chrono::system_clock::now() - c;
            std::cout << "FPS CPU: " << (int)(1/time_cpu.count()) << "," << "\tFPS OMP: " << (int)(1/time_omp.count()) << "," << "\tFPS GPU: " << (int)(1/time_gpu.count()) << std::endl;

            cv::imshow("Result", *edges_gpu);
            if(cv::waitKey(10) == 27){
                break;
            }
        }
        free(orig); free(orig_gs); free(edges_cpu); free(edges_omp);
        cudaFree(gpu_orig);
        cudaFree(gpu_sobel);

    }
    catch(const cv::Exception& ex)
    {
        std::cout << "Error: " << ex.what() << std::endl;
    }
    return 0;
}

void sobel_cpu(const cv::Mat* orig_gs, cv::Mat* edges_cpu, const unsigned int width, const unsigned int height) {
    for(int y = 1; y < height-1; y++) {
        for(int x = 1; x < width-1; x++) {
            int dx = (-1* (orig_gs->at<uint8_t>(y-1,x-1))) + (-2*(orig_gs->at<uint8_t>(y,x-1))) + (-1*(orig_gs->at<uint8_t>(y+1,x-1))) +
            (orig_gs->at<uint8_t>(y-1,x+1)) + (2*(orig_gs->at<uint8_t>(y,x+1))) + ((orig_gs->at<uint8_t>(y+1,x+1)));
            int dy = (orig_gs->at<uint8_t>(y-1,x-1)) + (2*orig_gs->at<uint8_t>(y-1,x)) + (orig_gs->at<uint8_t>(y-1,x+1)) +
            (-1*orig_gs->at<uint8_t>(y+1,x-1)) + (-2*orig_gs->at<uint8_t>(y+1,x)) + (-1*orig_gs->at<uint8_t>(y+1,x+1));
            edges_cpu->at<uint8_t>(y,x) = sqrt((dx*dx)+(dy*dy));
        }
    }
}

void sobel_omp(const cv::Mat* orig_gs, cv::Mat* edges_omp, const unsigned int width, const unsigned int height) {
    #pragma omp parallel for
    for(int y = 1; y < height-1; y++) {
        for(int x = 1; x < width-1; x++) {
            int dx = (-1* (orig_gs->at<uint8_t>(y-1,x-1))) + (-2*(orig_gs->at<uint8_t>(y,x-1))) + (-1*(orig_gs->at<uint8_t>(y+1,x-1))) +
            (orig_gs->at<uint8_t>(y-1,x+1)) + (2*(orig_gs->at<uint8_t>(y,x+1))) + ((orig_gs->at<uint8_t>(y+1,x+1)));
            int dy = (orig_gs->at<uint8_t>(y-1,x-1)) + (2*orig_gs->at<uint8_t>(y-1,x)) + (orig_gs->at<uint8_t>(y-1,x+1)) +
            (-1*orig_gs->at<uint8_t>(y+1,x-1)) + (-2*orig_gs->at<uint8_t>(y+1,x)) + (-1*orig_gs->at<uint8_t>(y+1,x+1));
            edges_omp->at<uint8_t>(y,x) = sqrt((dx*dx)+(dy*dy));
        }
    }
}

#include <iostream>
#include <math.h>
#include <thread>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <time.h>
#include <chrono>
#include <omp.h>
#include <sched.h>


/*
Compiling: Requires a Nvidia CUDA capable graphics card and the Nvidia GPU Computing Toolkit.
 *            Linux: nvcc -Wno-deprecated-gpu-targets -O3 -o test main.cu -Xcompiler -fopenmp `pkg-config --cflags --libs opencv`
 */

#define GRIDVAL 16.0
#define meanLength 60
typedef unsigned char byte;

void sobel_cpu(const cv::Mat* orig_gs, cv::Mat* edges_cpu, const unsigned int width, const unsigned int height);
void sobel_omp(const cv::Mat* orig_gs, cv::Mat* edges_omp, const unsigned int width, const unsigned int height);
int avg(int* fpsMeanVec);

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

std::string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
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
        strftime(timeBuffer, 80, "Sobel Edge Detector Benchamrks (%c)\n", curTime);
        printf("%s", timeBuffer);
        printf("CPU: %d hardware threads\n", std::thread::hardware_concurrency());
        printf("GPGPU: %s, CUDA %d.%d, %zd Mbytes global memory, %d CUDA cores\n",
        devProp.name, devProp.major, devProp.minor, devProp.totalGlobalMem / 1048576, cores);

        

        cv::namedWindow("Sobel Edge Detector",cv::WINDOW_AUTOSIZE);
        unsigned int width = 640;
        unsigned int height = 480;
        // unsigned int framerate = 120;
        // unsigned int flip_method = 0;
        // std::string pipeline = gstreamer_pipeline(width,
        //     height,
        //     width,
        //     height,
        //     framerate,
        //     flip_method);
        
        //cv::VideoCapture camera(pipeline, cv::CAP_GSTREAMER);
        cv::VideoCapture camera(0);
        if(!camera.isOpened())
            return -1;
        //cv::resizeWindow("Sobel Edge Detector", frameWidth, frameHeight);
        //camera.set(cv::CAP_PROP_FRAME_WIDTH, width);
        //camera.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        cv::Mat* orig, *orig_gs, *edges;
        //unsigned int width, height = 0;
        //width = camera.get(cv::CAP_PROP_FRAME_WIDTH);
        //height = camera.get(cv::CAP_PROP_FRAME_HEIGHT);
        
        orig = new cv::Mat(height,width,CV_8UC3);
        orig_gs = new cv::Mat(height,width,CV_8UC1);
        edges = new cv::Mat(height,width,CV_8UC1);
        byte *gpu_orig, *gpu_sobel;
        cudaMalloc((void**)&gpu_orig,(width*height*sizeof(byte)));
        cudaMalloc((void**)&gpu_sobel,(width*height*sizeof(byte)));

        dim3 threadsPerBlock(GRIDVAL, GRIDVAL, 1);
        dim3 numBlocks(ceil(width/GRIDVAL), ceil(height/GRIDVAL), 1);
        //std::cout << ceil(width/GRIDVAL)*ceil(height/GRIDVAL) << std::endl;

        uint8_t key = 0;
        int8_t tmp = 0;
        auto c = std::chrono::system_clock::now();
        std::chrono::duration<double> time;

        std::ostringstream buf;
        int* fpsMeanVec = new int[meanLength];
        memset(fpsMeanVec,0,meanLength*sizeof(int));
        uint8_t counter = 0;

        for(;;){
            camera >> *orig;
            cv::cvtColor(*orig, *orig_gs, CV_BGR2GRAY, 0);
            tmp = cv::waitKey(1);
            if(tmp != -1){
                key = tmp;
            }
            switch(key){
                case 99:
                    c = std::chrono::system_clock::now();
                    sobel_cpu(orig_gs, edges, width, height);
                    time = std::chrono::system_clock::now() - c;
                    fpsMeanVec[counter] = (int)(1/time.count());
                    counter++;
                    buf << "Mode: CPU" << "," << "  FPS: " << avg(fpsMeanVec);
                    putText(*edges, buf.str(), cv::Point(10, 30), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
                    cv::imshow("Sobel Edge Detector", *edges);
                    buf.str("");
                    buf.clear();
                    break;
                case 111: 
                    c = std::chrono::system_clock::now();
                    sobel_omp(orig_gs, edges, width, height);
                    time = std::chrono::system_clock::now() - c;
                    fpsMeanVec[counter] = (int)(1/time.count());
                    counter++;
                    buf << "Mode: OMP" << "," << "  FPS: " << avg(fpsMeanVec);
                    putText(*edges, buf.str(), cv::Point(10, 30), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
                    cv::imshow("Sobel Edge Detector", *edges);
                    buf.str("");
                    buf.clear();
                    break;
                case 103:
                    c = std::chrono::system_clock::now();
                    cudaMemcpy(gpu_orig, orig_gs->data, (width*height*sizeof(byte)), cudaMemcpyHostToDevice);
                    //cudaMemset(gpu_sobel, 0, (width*height*sizeof(byte)));
                    sobel_gpu<<<numBlocks, threadsPerBlock>>>(gpu_orig, gpu_sobel, width, height);
                    cudaDeviceSynchronize(); // waits for completion, returns error code
                    cudaMemcpy(edges->data, gpu_sobel, (width*height), cudaMemcpyDeviceToHost);
                    time = std::chrono::system_clock::now() - c;
                    fpsMeanVec[counter] = (int)(1/time.count());
                    counter++;
                    buf << "Mode: GPU" << "," << "  FPS: " << avg(fpsMeanVec);
                    putText(*edges, buf.str(), cv::Point(10, 30), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
                    cv::imshow("Sobel Edge Detector", *edges);
                    buf.str("");
                    buf.clear();
                    break;
                case 27:
                    delete orig; delete orig_gs; delete edges;
                    cudaFree(gpu_orig); cudaFree(gpu_sobel);
                    camera.release();
                    cv::destroyAllWindows();
                    return 0;
                default:
                    cv::imshow("Sobel Edge Detector", *orig);
                    break;
            }
            if(counter==meanLength-1){
                counter = 0;
            }
            //std::cout << "FPS: " << (int)(1/time.count()) << std::endl;
            
        }

    }
    catch(const cv::Exception& ex)
    {
        std::cout << "Error: " << ex.what() << std::endl;
    }
    return 0;
}

void sobel_cpu(const cv::Mat* orig_gs, cv::Mat* edges_cpu, const unsigned int width, const unsigned int height) {
    omp_set_num_threads(1);
    
    #pragma omp parallel for
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
    omp_set_num_threads(4);
    
    #pragma omp parallel for
    for(int y = 1; y < height-1; y++) {
        for(int x = 1; x < width-1; x++) {
            int dx = (-1* (orig_gs->at<uint8_t>(y-1,x-1))) + (-2*(orig_gs->at<uint8_t>(y,x-1))) + (-1*(orig_gs->at<uint8_t>(y+1,x-1))) +
            (orig_gs->at<uint8_t>(y-1,x+1)) + (2*(orig_gs->at<uint8_t>(y,x+1))) + ((orig_gs->at<uint8_t>(y+1,x+1)));
            int dy = (orig_gs->at<uint8_t>(y-1,x-1)) + (2*orig_gs->at<uint8_t>(y-1,x)) + (orig_gs->at<uint8_t>(y-1,x+1)) +
            (-1*orig_gs->at<uint8_t>(y+1,x-1)) + (-2*orig_gs->at<uint8_t>(y+1,x)) + (-1*orig_gs->at<uint8_t>(y+1,x+1));
            edges_omp->at<uint8_t>(y,x) = sqrt((dx*dx)+(dy*dy));
            //printf("Thread %3d is running on cpu %3d\n", omp_get_thread_num(), sched_getcpu());
        }
    }
}

int avg(int* fpsMeanVec){
    int sum = 0;
    for(int i=0; i<meanLength; i++)
    {
        sum = sum + fpsMeanVec[i];
    }
    return (int)sum/meanLength;
}
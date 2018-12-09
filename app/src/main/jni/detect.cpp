//
// Created by LG-PC on 2018-11-22.
//

#include <jni.h>
#include <android/log.h>
#include "../include/leek_include/mssd.h"
#include "../include/leek_include/Obs_util.h"
#include <../include/opencv2/opencv.hpp>

#include <sys/time.h>
#include <CL/cl.h>


#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,__VA_ARGS__)
#define FIXED_WIDTH 300
#define FIXED_HEIGHT 300
#define CHANNEL 3
#define IMG_SIZE FIXED_WIDTH * FIXED_HEIGHT * CHANNEL
#define IDX_OF_STAIR 0
#define OBS_POINTER_BUFFER_SIZE 100


#define getMillisecond(start, end) \
	(end.tv_sec-start.tv_sec)*1000 + \
    (end.tv_usec-start.tv_usec)/1000.0

struct timeval start, end;
double timer[10];


////hyonzin: for OpenCL

#include <dlfcn.h>
#include <iostream>

#define CHECK_ERROR(err) if( err != CL_SUCCESS ) { LOGE("OCL", "Error: %d (%d line)", err, __LINE__); }

size_t roundUp( int group_size, int global_size )
{
    int r = global_size % group_size;
    if( r == 0) {
        return global_size;
    } else {
        return global_size + group_size - r;
    }
}

//std::string readKernel(const char* filename)
//{
//    std::ifstream ifs(filename, std::ios_base::in);
//    if( !ifs.is_open() ) {
//        std::cerr << "Failed to open file" << std::endl;
//        return NULL;
//    }
//
//    std::ostringstream oss;
//    oss << ifs.rdbuf();
//    return oss.str();
//}



#define DECLARE_FUNCTION_PTR(func_name) \
    decltype(func_name) (* func_name##_ptr) = nullptr;

DECLARE_FUNCTION_PTR(clCreateContext);
DECLARE_FUNCTION_PTR(clCreateContextFromType);
DECLARE_FUNCTION_PTR(clCreateCommandQueue);
DECLARE_FUNCTION_PTR(clGetContextInfo);
DECLARE_FUNCTION_PTR(clBuildProgram);
DECLARE_FUNCTION_PTR(clEnqueueNDRangeKernel);
DECLARE_FUNCTION_PTR(clSetKernelArg);
DECLARE_FUNCTION_PTR(clReleaseKernel);
DECLARE_FUNCTION_PTR(clCreateProgramWithSource);
DECLARE_FUNCTION_PTR(clCreateBuffer);
DECLARE_FUNCTION_PTR(clRetainKernel);
DECLARE_FUNCTION_PTR(clCreateKernel);
DECLARE_FUNCTION_PTR(clGetProgramInfo);
DECLARE_FUNCTION_PTR(clFlush);
DECLARE_FUNCTION_PTR(clFinish);
DECLARE_FUNCTION_PTR(clReleaseProgram);
DECLARE_FUNCTION_PTR(clRetainContext);
DECLARE_FUNCTION_PTR(clCreateProgramWithBinary);
DECLARE_FUNCTION_PTR(clReleaseCommandQueue);
DECLARE_FUNCTION_PTR(clEnqueueMapBuffer);
DECLARE_FUNCTION_PTR(clRetainProgram);
DECLARE_FUNCTION_PTR(clGetProgramBuildInfo);
DECLARE_FUNCTION_PTR(clEnqueueReadBuffer);
DECLARE_FUNCTION_PTR(clEnqueueWriteBuffer);
DECLARE_FUNCTION_PTR(clReleaseEvent);
DECLARE_FUNCTION_PTR(clReleaseContext);
DECLARE_FUNCTION_PTR(clRetainCommandQueue);
DECLARE_FUNCTION_PTR(clEnqueueUnmapMemObject);
DECLARE_FUNCTION_PTR(clRetainMemObject);
DECLARE_FUNCTION_PTR(clReleaseMemObject);
DECLARE_FUNCTION_PTR(clGetDeviceInfo);
DECLARE_FUNCTION_PTR(clGetDeviceIDs);
DECLARE_FUNCTION_PTR(clRetainEvent);
DECLARE_FUNCTION_PTR(clGetPlatformIDs);
DECLARE_FUNCTION_PTR(clGetKernelWorkGroupInfo);
DECLARE_FUNCTION_PTR(clGetCommandQueueInfo);
DECLARE_FUNCTION_PTR(clGetKernelInfo);
DECLARE_FUNCTION_PTR(clGetEventProfilingInfo);
DECLARE_FUNCTION_PTR(clEnqueueMarker);
DECLARE_FUNCTION_PTR(clWaitForEvents);
#undef DECLARE_FUNCTION_PTR


int output_height = FIXED_HEIGHT;
int output_width = FIXED_WIDTH;
int max_width = 1920;
int max_height = 1080;
int channels = 3;
float* rgb;
float* resized_rgb;


// for OpenCL
void *handle;
cl_device_id     device_id;
cl_context       context;
cl_command_queue commands;
cl_program       program;
cl_kernel        kernel_yuv2rgb;
cl_kernel        kernel_resize;
cl_uint          numPlatforms;
cl_platform_id   firstPlatformId;

int ndim = 1;
size_t local[] = {64};						// number of work-items per work-group
int tmp = 0;  // trick for init

//Memory allocation for gpu(device)
cl_mem d_nv21yuv, d_rgb, d_resized_rgb;


void init_opencl() {

// Load Dynamic Library


//    void *handle = dlopen("libOpenCL.so", RTLD_LAZY | RTLD_LOCAL);
    handle = dlopen("/vendor/lib64/libOpenCL.so", RTLD_LAZY | RTLD_LOCAL);

    if (handle == nullptr) {
        LOGI("OCL", "handle == NULL !!!!!");
        return;
    }

    LOGI("OCL","Loading OpenCL Symbols");

// Load OpenCL Functions
#define LOAD_FUNCTION_PTR(func_name, handle) \
        func_name##_ptr = reinterpret_cast<decltype(func_name##_ptr)>(dlsym(handle, #func_name));

    LOAD_FUNCTION_PTR(clCreateContext, handle);
    LOAD_FUNCTION_PTR(clCreateContextFromType, handle);
    LOAD_FUNCTION_PTR(clCreateCommandQueue, handle);
    LOAD_FUNCTION_PTR(clGetContextInfo, handle);
    LOAD_FUNCTION_PTR(clBuildProgram, handle);
    LOAD_FUNCTION_PTR(clEnqueueNDRangeKernel, handle);
    LOAD_FUNCTION_PTR(clSetKernelArg, handle);
    LOAD_FUNCTION_PTR(clReleaseKernel, handle);
    LOAD_FUNCTION_PTR(clCreateProgramWithSource, handle);
    LOAD_FUNCTION_PTR(clCreateBuffer, handle);
    LOAD_FUNCTION_PTR(clRetainKernel, handle);
    LOAD_FUNCTION_PTR(clCreateKernel, handle);
    LOAD_FUNCTION_PTR(clGetProgramInfo, handle);
    LOAD_FUNCTION_PTR(clFlush, handle);
    LOAD_FUNCTION_PTR(clFinish, handle);
    LOAD_FUNCTION_PTR(clReleaseProgram, handle);
    LOAD_FUNCTION_PTR(clRetainContext, handle);
    LOAD_FUNCTION_PTR(clCreateProgramWithBinary, handle);
    LOAD_FUNCTION_PTR(clReleaseCommandQueue, handle);
    LOAD_FUNCTION_PTR(clEnqueueMapBuffer, handle);
    LOAD_FUNCTION_PTR(clRetainProgram, handle);
    LOAD_FUNCTION_PTR(clGetProgramBuildInfo, handle);
    LOAD_FUNCTION_PTR(clEnqueueReadBuffer, handle);
    LOAD_FUNCTION_PTR(clEnqueueWriteBuffer, handle);
    LOAD_FUNCTION_PTR(clReleaseEvent, handle);
    LOAD_FUNCTION_PTR(clReleaseContext, handle);
    LOAD_FUNCTION_PTR(clRetainCommandQueue, handle);
    LOAD_FUNCTION_PTR(clEnqueueUnmapMemObject, handle);
    LOAD_FUNCTION_PTR(clRetainMemObject, handle);
    LOAD_FUNCTION_PTR(clReleaseMemObject, handle);
    LOAD_FUNCTION_PTR(clGetDeviceInfo, handle);
    LOAD_FUNCTION_PTR(clGetDeviceIDs, handle);
    LOAD_FUNCTION_PTR(clRetainEvent, handle);
    LOAD_FUNCTION_PTR(clGetPlatformIDs, handle);
    LOAD_FUNCTION_PTR(clGetKernelWorkGroupInfo, handle);
    LOAD_FUNCTION_PTR(clGetCommandQueueInfo, handle);
    LOAD_FUNCTION_PTR(clGetKernelInfo, handle);
    LOAD_FUNCTION_PTR(clGetEventProfilingInfo, handle);
    LOAD_FUNCTION_PTR(clEnqueueMarker, handle);
    LOAD_FUNCTION_PTR(clWaitForEvents, handle);

#undef LOAD_FUNCTION_PTR

    int err;

    if (clGetPlatformIDs_ptr == nullptr) {
        LOGI("OCL", "clGetPlatformIDs_ptr == NULL!!");
    }

    err = clGetPlatformIDs_ptr(1, &firstPlatformId, &numPlatforms);
    CHECK_ERROR(err);

    err = clGetDeviceIDs_ptr(firstPlatformId, CL_DEVICE_TYPE_GPU, 1,
                             &device_id, NULL);
    CHECK_ERROR(err);

    cl_context_properties properties [] =
            {
                    CL_CONTEXT_PLATFORM, (cl_context_properties)firstPlatformId, 0
            };
    context = clCreateContext_ptr(properties, 1, &device_id, NULL, NULL, &err);
    CHECK_ERROR(err);

    commands = clCreateCommandQueue_ptr(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_ERROR(err);

    // Print device info
    char cBuffer[1024];
    LOGI("OCL"," ---------------------------------\n");
    clGetDeviceInfo_ptr(device_id, CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);
    LOGI("OCL"," Device: %s\n", cBuffer);
    LOGI("OCL"," ---------------------------------\n");
    cl_uint compute_units;
    err = clGetDeviceInfo_ptr(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    LOGI("OCL", "  CL_DEVICE_MAX_COMPUTE_UNITS:\t\t%u\n", compute_units);
    CHECK_ERROR(err);


    // Create the compute program from the source buffer
    const char* source2 = "__kernel void kernel_yuv2rgb( \n"
                         "       __global float *rgb, \n"
                         "       __global unsigned char* nv21yuv, \n"
                         "       int width, int height) { \n"
                         "            int gid = get_global_id(0); \n"
                         "            int i = (int)(gid/width), j = (int)(gid%width); \n"
                         "            int frameSize = width * height;\n"
                         "            int y = (0xff & ((int) nv21Yuv[i * width + j]));\n"
                         "            int v = (0xff & ((int) nv21Yuv[frameSize + (i >> 1) * width + (j & ~1) + 0]));\n"
                         "            int u = (0xff & ((int) nv21Yuv[frameSize + (i >> 1) * width + (j & ~1) + 1]));\n"
                         "            y = y < 16 ? 16 : y;\n"
                         "\n"
                         "            int r = (int) (1.164f * (y - 16) + 1.596f * (v - 128));\n"
                         "            int g = (int) (1.164f * (y - 16) - 0.813f * (v - 128) - 0.391f * (u - 128));\n"
                         "            int b = (int) (1.164f * (y - 16) + 2.018f * (u - 128));\n"
                         "\n"
                         "            r = r < 0 ? 0 : (r > 255 ? 255 : r);\n"
                         "            g = g < 0 ? 0 : (g > 255 ? 255 : g);\n"
                         "            b = b < 0 ? 0 : (b > 255 ? 255 : b);\n"
                         "\n"
                         "            // CHW\n"
                         "            int pix = i*width + j;\n"
                         "            rgb[              pix] = r;\n"
                         "            rgb[frameSize   + pix] = g;\n"
                         "} \n"
                         "__kernel void kernel_resize( \n"
                         "       __global float *resized_rgb, \n"
                         "       __global float* rgb, \n"
                         "       int width,        int height, \n"
                         "       int output_width, int output_height) { \n"
                         "            int gid = get_global_id(0); \n"
                         "            int h2 = (int)(gid/width), w2 = (int)(gid%width); \n"
                         "            int channels = 3;\n"
                         "            const float rheight = (output_height > 1)\n"
                         "                                  ? (float)(height - 1) / (output_height - 1) : 0.f;\n"
                         "            const float rwidth = (output_width > 1) ? (float)(width - 1) / (output_width - 1) : 0.f;\n"
                         "\n"
                         "            const float h1r = rheight * h2;\n"
                         "            const int h1 = h1r;\n"
                         "            const int h1p = (h1 < output_height - 1) ? 1 : 0;\n"
                         "            const float h1lambda = h1r - h1;\n"
                         "            const float h0lambda = (float)1. - h1lambda;\n"
                         "\n"
                         "            const float w1r = rwidth * w2;\n"
                         "            const int w1 = w1r;\n"
                         "            const int w1p = (w1 < output_width - 1) ? 1 : 0;\n"
                         "            const float w1lambda = w1r - w1;\n"
                         "            const float w0lambda = (float)1. - w1lambda;\n"
                         "            float* pos1 = &rgb[h1 * output_width + w1];\n"
                         "            const float* pos2 = &resized_rgb[h2 * width + w2];\n"
                         "\n"
                         "            for (int c = 0; c < channels; ++c) {\n"
                         "                pos1[0] += h0lambda * w0lambda * pos2[0];\n"
                         "                pos1[w1p] += h0lambda * w1lambda * pos2[0];\n"
                         "                pos1[h1p * output_width] += h1lambda * w0lambda * pos2[0];\n"
                         "                pos1[h1p * output_width + w1p] += h1lambda * w1lambda * pos2[0];\n"
                         "                pos1 += output_width * output_height;\n"
                         "                pos2 += width * height;\n"
                         "            } \n"
                         "}";

    const char* source3 = "__kernel void kernel_yuv2rgb( \
                                __global float *rgb, \
                                __global unsigned char* nv21yuv, \
                                int width, int height) { \
                                     int gid = get_global_id(0); \
                                     int i = (int)(gid/width), j = (int)(gid%width); \
                                     int frameSize = width * height;\
                                     int y = (0xff & ((int) nv21Yuv[i * width + j]));\
                                     int v = (0xff & ((int) nv21Yuv[frameSize + (i >> 1) * width + (j & ~1) + 0]));\
                                     int u = (0xff & ((int) nv21Yuv[frameSize + (i >> 1) * width + (j & ~1) + 1]));\
                                     y = y < 16 ? 16 : y;\
                         \
                                     int r = (int) (1.164f * (y - 16) + 1.596f * (v - 128));\
                                     int g = (int) (1.164f * (y - 16) - 0.813f * (v - 128) - 0.391f * (u - 128));\
                                     int b = (int) (1.164f * (y - 16) + 2.018f * (u - 128));\
                         \
                                     r = r < 0 ? 0 : (r > 255 ? 255 : r);\
                                     g = g < 0 ? 0 : (g > 255 ? 255 : g);\
                                     b = b < 0 ? 0 : (b > 255 ? 255 : b);\
                         \
                                     int pix = i*width + j;\
                                     rgb[              pix] = r;\
                                     rgb[frameSize   + pix] = g;\
                         } \
                         __kernel void kernel_resize( \
                                __global float *resized_rgb, \
                                __global float* rgb, \
                                int width,        int height, \
                                int output_width, int output_height) { \
                                     int gid = get_global_id(0); \
                                     int h2 = (int)(gid/width), w2 = (int)(gid%width); \
                                     int channels = 3;\
                                     const float rheight = (output_height > 1)\
                                                           ? (float)(height - 1) / (output_height - 1) : 0.f;\
                                     const float rwidth = (output_width > 1) ? (float)(width - 1) / (output_width - 1) : 0.f;\
                        \
                                     const float h1r = rheight * h2;\
                                     const int h1 = h1r;\
                                     const int h1p = (h1 < output_height - 1) ? 1 : 0;\
                                     const float h1lambda = h1r - h1;\
                                     const float h0lambda = (float)1. - h1lambda;\
                        \
                                     const float w1r = rwidth * w2;\
                                     const int w1 = w1r;\
                                     const int w1p = (w1 < output_width - 1) ? 1 : 0;\
                                     const float w1lambda = w1r - w1;\
                                     const float w0lambda = (float)1. - w1lambda;\
                                     float* pos1 = &rgb[h1 * output_width + w1];\
                                     const float* pos2 = &resized_rgb[h2 * width + w2];\
                        \
                                     for (int c = 0; c < channels; ++c) {\
                                         pos1[0] += h0lambda * w0lambda * pos2[0];\
                                         pos1[w1p] += h0lambda * w1lambda * pos2[0];\
                                         pos1[h1p * output_width] += h1lambda * w0lambda * pos2[0];\
                                         pos1[h1p * output_width + w1p] += h1lambda * w1lambda * pos2[0];\
                                         pos1 += output_width * output_height;\
                                         pos2 += width * height;\
                                     }\
                                 }";


    const char* source = "__kernel void kernel_yuv2rgb( \
                                __global float *rgb, \
                                __global unsigned char* nv21yuv, \
                                int width, int height) { \
                                     int gid = get_global_id(0); \
                                     int i = (int)(gid/width), j = (int)(gid%width); \
                                     int frameSize = width * height;\
                         } \
                         __kernel void kernel_resize( \
                                __global float *resized_rgb, \
                                __global float* rgb, \
                                int width,        int height, \
                                int output_width, int output_height) { \
                                     int gid = get_global_id(0); \
                                     int h2 = (int)(gid/width), w2 = (int)(gid%width); \
                                     int channels = 3;\
                                     float rheight = (output_height > 1)\
                                                           ? (float)(height - 1) / (output_height - 1) : 0.f;\
                                     float rwidth = (output_width > 1) ? (float)(width - 1) / (output_width - 1) : 0.f;\
                        \
                                     float h1r = rheight * h2;\
                                     int h1 = h1r;\
                                     int h1p = (h1 < output_height - 1) ? 1 : 0;\
                                     float h1lambda = h1r - h1;\
                                     float h0lambda = (float)1. - h1lambda;\
                        \
                                     float w1r = rwidth * w2;\
                                     int w1 = w1r;\
                                     int w1p = (w1 < output_width - 1) ? 1 : 0;\
                                     float w1lambda = w1r - w1;\
                                     float w0lambda = (float)1. - w1lambda;\
                                     float* pos1 = &rgb[h1 * output_width + w1];\
                                     float* pos2 = &resized_rgb[h2 * width + w2];\
                        \
                                     for (int c = 0; c < channels; ++c) {\
                                         pos1[0] += h0lambda * w0lambda * pos2[0];\
                                         pos1[w1p] += h0lambda * w1lambda * pos2[0];\
                                         pos1[h1p * output_width] += h1lambda * w0lambda * pos2[0];\
                                         pos1[h1p * output_width + w1p] += h1lambda * w1lambda * pos2[0];\
                                         pos1 += output_width * output_height;\
                                         pos2 += width * height;\
                                     }\
                                 }";
    program = clCreateProgramWithSource_ptr(context, 1, &source, NULL, &err);
    CHECK_ERROR(err);

    // Build the program
    char build_option[1000] = {0,};
    err = clBuildProgram_ptr( program, 0, NULL, build_option, NULL, NULL);
    if( err != CL_SUCCESS ) {
        size_t len;
        char buffer[2048];

        LOGI("OCL","Error: Failed to build program executable!\n");
        clGetProgramBuildInfo_ptr( program, device_id, CL_PROGRAM_BUILD_LOG,
                                   sizeof(buffer), buffer, &len);
        LOGI("OCL","%s\n", buffer);
        return ;
    }

    // Create the compute kernel from the program
    kernel_yuv2rgb = clCreateKernel_ptr(program, "kernel_yuv2rgb", &err);  CHECK_ERROR(err);
    kernel_resize  = clCreateKernel_ptr(program, "kernel_resize",  &err);  CHECK_ERROR(err);


    LOGI("OCL","create buffer start");
    d_nv21yuv = clCreateBuffer_ptr(context, CL_MEM_READ_WRITE,
                                   (int)(max_width * max_height * 1.5) * sizeof(unsigned char),
                                   NULL, &err);
    CHECK_ERROR(err);

    d_rgb = clCreateBuffer_ptr(context, CL_MEM_READ_WRITE,
                               max_width * max_height * channels * sizeof(float),
                               NULL, &err);
    CHECK_ERROR(err);

    d_resized_rgb = clCreateBuffer_ptr(context, CL_MEM_READ_WRITE,
                                       FIXED_WIDTH * FIXED_HEIGHT * channels * sizeof(float),
                                       NULL, &err);
    CHECK_ERROR(err);
    LOGI("OCL","create buffer end.");

}

void nv21_to_rgb_gpu(unsigned char* nv21yuv, int width, int height) {

    int err;

    // Write the A and B matrices into compute device memory
    err = clEnqueueWriteBuffer_ptr(commands, d_nv21yuv, CL_TRUE, 0,
                                   (int)(width * height * 1.5) * sizeof(unsigned char), nv21yuv, 0,
                                   NULL, NULL);
    CHECK_ERROR(err);

    size_t global[] = {(size_t)roundUp(local[0], width * height)};	// number of total work-items

    // Set the arguments to our compute kernel
    if (tmp < 10) {
        err = 0;
        err |= clSetKernelArg_ptr(kernel_yuv2rgb, 0, sizeof(cl_mem), &d_rgb);
        err |= clSetKernelArg_ptr(kernel_yuv2rgb, 1, sizeof(cl_mem), &d_nv21yuv);
        err |= clSetKernelArg_ptr(kernel_yuv2rgb, 2, sizeof(int), &width);
        err |= clSetKernelArg_ptr(kernel_yuv2rgb, 3, sizeof(int), &height);
        CHECK_ERROR(err);
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);

    err = clEnqueueNDRangeKernel_ptr(commands, kernel_yuv2rgb, ndim, NULL,
                                     global, local, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Wait for the command to be completed before reading back results
    clFinish_ptr(commands);

    gettimeofday(&end, NULL);
    LOGI("kernel launch 1", "%f", getMillisecond(start, end));  // Preprocessing (1)

}

void resize_gpu(float* resized_rgb, int width, int height, int output_width, int output_height) {

    int err;

    size_t global[] = {(size_t)roundUp(local[0], width * height)};	// number of total work-items
    // Set the arguments to our compute kernel
    if(tmp < 10) {
        err = 0;
        err |= clSetKernelArg_ptr(kernel_resize, 0, sizeof(cl_mem), &d_resized_rgb);
        err |= clSetKernelArg_ptr(kernel_resize, 1, sizeof(cl_mem), &d_rgb);
        err |= clSetKernelArg_ptr(kernel_resize, 2, sizeof(int), &width);
        err |= clSetKernelArg_ptr(kernel_resize, 3, sizeof(int), &height);
        err |= clSetKernelArg_ptr(kernel_resize, 4, sizeof(int), &output_width);
        err |= clSetKernelArg_ptr(kernel_resize, 5, sizeof(int), &output_height);
        CHECK_ERROR(err);

        tmp++;
    }


    struct timeval start, end;
    gettimeofday(&start, NULL);

    err = clEnqueueNDRangeKernel_ptr(commands, kernel_resize, ndim, NULL,
                                     global, local, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Wait for the command to be completed before reading back results
    clFinish_ptr(commands);

    gettimeofday(&end, NULL);
    LOGI("kernel launch 2", "%f", getMillisecond(start, end));  // Preprocessing (1)

    // Read back the results from the compute device
    err = clEnqueueReadBuffer_ptr(commands, d_resized_rgb, CL_TRUE, 0,
                                  output_width * output_height * channels * sizeof(float),
                                  resized_rgb, 0, NULL, NULL);
    CHECK_ERROR(err);
}


Obs_gauge stair_guage;
graph_t global_graph= NULL;
tensor_t global_tensor_input = NULL;
tensor_t global_tensor_out = NULL;
int dims[] = {1,3,300,300}; // NCHW
int num_detected_obj = 0;


float* out_data = NULL;
float* global_input = NULL;
float threshold = 0.5;


void nv21_to_rgb_cpu (float* rgb, unsigned char* nv21Yuv, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int frameSize = width * height;
            int y = (0xff & ((int) nv21Yuv[i * width + j]));
            int v = (0xff & ((int) nv21Yuv[frameSize + (i >> 1) * width + (j & ~1) + 0]));
            int u = (0xff & ((int) nv21Yuv[frameSize + (i >> 1) * width + (j & ~1) + 1]));
            y = y < 16 ? 16 : y;

            int r = (int) (1.164f * (y - 16) + 1.596f * (v - 128));
            int g = (int) (1.164f * (y - 16) - 0.813f * (v - 128) - 0.391f * (u - 128));
            int b = (int) (1.164f * (y - 16) + 2.018f * (u - 128));

            r = r < 0 ? 0 : (r > 255 ? 255 : r);
            g = g < 0 ? 0 : (g > 255 ? 255 : g);
            b = b < 0 ? 0 : (b > 255 ? 255 : b);

            // CHW
            int pix = i*width + j;
            rgb[              pix] = r;
            rgb[frameSize   + pix] = g;
            rgb[frameSize*2 + pix] = b;
        }
    }
}


void resize_cpu(float* resized_rgb, float* rgb, int width, int height, int output_width, int output_height) {
    for (int h2 = 0; h2 < height; ++h2) {
        for (int w2 = 0; w2 < width; ++w2) {
            int channels = 3;
            const float rheight = (output_height > 1)
                                  ? (float)(height - 1) / (output_height - 1) : 0.f;
            const float rwidth = (output_width > 1) ? (float)(width - 1) / (output_width - 1) : 0.f;

            const float h1r = rheight * h2;
            const int h1 = h1r;
            const int h1p = (h1 < output_height - 1) ? 1 : 0;
            const float h1lambda = h1r - h1;
            const float h0lambda = (float)1. - h1lambda;

            const float w1r = rwidth * w2;
            const int w1 = w1r;
            const int w1p = (w1 < output_width - 1) ? 1 : 0;
            const float w1lambda = w1r - w1;
            const float w0lambda = (float)1. - w1lambda;
            float* pos1 = &rgb[h1 * output_width + w1];
            const float* pos2 = &resized_rgb[h2 * width + w2];

            for (int c = 0; c < channels; ++c) {
                pos1[0] += h0lambda * w0lambda * pos2[0];
                pos1[w1p] += h0lambda * w1lambda * pos2[0];
                pos1[h1p * output_width] += h1lambda * w0lambda * pos2[0];
                pos1[h1p * output_width + w1p] += h1lambda * w1lambda * pos2[0];
                pos1 += output_width * output_height;
                pos2 += width * height;
            }
        }
    }
}

void normalization_cpu(float* global_input, float* resized_rgb, int n) {
//    for (int i = 0; i < n; i++)    {
//        global_input[i] = 0.007843 * (resized_rgb[i] - 127.5);
//    }
    for (int i = 0; i < n; i+=2)    {
        global_input[i] = (float)(0.007843 * (resized_rgb[i] - 127.5));
    }
    for (int i = 1; i < n; i+=2)    {
        global_input[i] = (float)(0.007843 * (resized_rgb[i] - 127.5));
    }
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_leek_my_1usb_DetectManager_detect(JNIEnv *env, jclass type, jbyteArray nv21Yuv_,
                                                   jint width, jint height) {
    num_detected_obj = 0;
    jbyte* const nv21Yuv = env->GetByteArrayElements(nv21Yuv_, NULL);


    bool isGPU = true;

    /* Preprocessing */
    // 1) YUV to RGB
    // 2) Resize
    // 3) uchar to float
    // 4) Normaliztion and HWC to CHW

    gettimeofday(&start, NULL);

    if (isGPU) {
        // 1) YUV to RGB
        nv21_to_rgb_gpu((unsigned char *) nv21Yuv, width, height);

        // 2) Resize
        // And 3) uchar to float
        resize_gpu(resized_rgb, width, height, output_width, output_height);

    } else {
        // 1) YUV to RGB
        nv21_to_rgb_cpu(rgb, (unsigned char *) nv21Yuv, width, height);

        // 2) Resize
        // And 3) uchar to float
        resize_cpu(resized_rgb, rgb, width, height, output_width, output_height);
    }


    gettimeofday(&end, NULL);
    timer[0] = getMillisecond(start, end);  // Preprocessing (1)

    gettimeofday(&start, NULL);

    // 4) Normaliztion
    normalization_cpu(global_input, resized_rgb, FIXED_WIDTH * FIXED_HEIGHT * 3);

    gettimeofday(&end, NULL);
    timer[1] = getMillisecond(start, end);  // Preprocessing (2)

    /* Inference */

    gettimeofday(&start, NULL);

    int res = detect(global_input,&out_data,global_graph,global_tensor_input,&global_tensor_out,&num_detected_obj,IMG_SIZE);


    gettimeofday(&end, NULL);
    timer[2] = getMillisecond(start, end);  // Inference

    LOGI("night >> convert,resize", "%.3lf", timer[0]);
    LOGI("night >> normalization",  "%.3lf", timer[1]);
    LOGI("night >> inference",      "%.3lf", timer[2]);


    env->ReleaseByteArrayElements(nv21Yuv_, nv21Yuv, 0);
    if(res == 0)
        return JNI_TRUE;
    return JNI_FALSE;

}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_leek_my_1usb_DetectManager_delete_1graph_1space(JNIEnv *env, jclass type) {

    // TODO
    LOGI("detect", "graph_delete");
    const char* model_name = "mssd_300";
    graph_finish(global_input,global_graph,global_tensor_input,model_name);
    global_input = NULL;
    global_tensor_input = NULL;
    global_graph = NULL;

    dlclose(handle);

    clReleaseProgram_ptr(program);
    clReleaseKernel_ptr(kernel_yuv2rgb);
    clReleaseKernel_ptr(kernel_resize);
    clReleaseMemObject_ptr(d_nv21yuv);
    clReleaseMemObject_ptr(d_rgb);
    clReleaseMemObject_ptr(d_resized_rgb);
    clReleaseCommandQueue_ptr(commands);
    clReleaseContext_ptr(context);

    if(global_graph !=NULL | global_tensor_input != NULL | global_graph !=NULL)
        return JNI_FALSE;
    return JNI_TRUE;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_leek_my_1usb_DetectManager_delete_1out_1data(JNIEnv *env, jclass type) {
    delete_out_tensor(global_tensor_out);
}



extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_leek_my_1usb_DetectManager_get_1graph_1space(JNIEnv *env, jclass type,
                                                              jstring model_name_,
                                                              jstring model_path_,
                                                              jstring proto_path_,
                                                              jstring device_type_) {
    const char *model_name  = env->GetStringUTFChars(model_name_, 0);
    const char *model_path  = env->GetStringUTFChars(model_path_, 0);
    const char *proto_path  = env->GetStringUTFChars(proto_path_, 0);
    const char *device_type = env->GetStringUTFChars(device_type_, 0);


    // TODO
    global_input = (float*)malloc(sizeof(float) *IMG_SIZE);
    if(global_input == NULL)
        return JNI_FALSE;
//    int result = graph_ready(&global_graph, &global_tensor_input, dims, model_name, model_path,
//            proto_path, device_type);
    int result = graph_ready(&global_graph,&global_tensor_input,dims,model_name,model_path,proto_path,nullptr);


    env->ReleaseStringUTFChars(model_name_, model_name);
    env->ReleaseStringUTFChars(model_path_, model_path);
    env->ReleaseStringUTFChars(proto_path_, proto_path);
    env->ReleaseStringUTFChars(device_type_, device_type);

    init_opencl();

    rgb = new float[max_width * max_height * channels];
    resized_rgb = new float[output_width * output_height * channels];

    if(result == 0)
        return JNI_TRUE;
    return JNI_FALSE;
}



extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_leek_my_1usb_DetectManager_get_1out_1data(JNIEnv *env, jclass type,
                                                           jfloatArray data_of_java_) {
    jfloat *data_of_java = env->GetFloatArrayElements(data_of_java_, NULL);

    int top = -1;
    float* obs_pointer_buffer[OBS_POINTER_BUFFER_SIZE] = {NULL,};
    float* temp_processed_data = &data_of_java[1];
    float* data = out_data;
    data_of_java[0]=num_detected_obj;
    for (int i=0; i<num_detected_obj; i++)
    {
        if( data[1] > threshold ) {
            if( data[0] != IDX_OF_STAIR ) {
                temp_processed_data[0] = data[0];
                temp_processed_data[1] = -1;
                temp_processed_data[2] = data[2];
                temp_processed_data[3] = data[3];
                temp_processed_data[4] = data[4];
                temp_processed_data[5] = data[5];
            } else {
                obs_pointer_buffer[++top] = data;
            }
        }
        data+=6;
        temp_processed_data+=6;
    }
    for(int i = 0 ; i<= top ; i++){
        float* loaded_obs_pointer = obs_pointer_buffer[i];
        gauge_control(loaded_obs_pointer,&stair_guage);
        temp_processed_data[0] = loaded_obs_pointer[0];
        temp_processed_data[1] = get_state(&stair_guage);
        temp_processed_data[2] = loaded_obs_pointer[2];
        temp_processed_data[3] = loaded_obs_pointer[3];
        temp_processed_data[4] = loaded_obs_pointer[4];
        temp_processed_data[5] = loaded_obs_pointer[5];
        temp_processed_data+=6;
    }
    if(top == -1){
        gauge_control(nullptr,&stair_guage);
    }
    env->ReleaseFloatArrayElements(data_of_java_, data_of_java, 0);
    return JNI_TRUE;


}

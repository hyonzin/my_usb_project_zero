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

#include "add.hpp"


#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,__VA_ARGS__)
#define FIXED_WIDTH 300
#define FIXED_HEIGHT 300
#define CHANNEL 3
#define IMG_SIZE FIXED_WIDTH * FIXED_HEIGHT * CHANNEL
#define IDX_OF_STAIR 0
#define OBS_POINTER_BUFFER_SIZE 100

//global variable


////hyonzin: for OpenCL


#include <dlfcn.h>
#include <iostream>

#define CHECK_ERROR(err) if( err != CL_SUCCESS ) { LOGI("OCL", "Error: %d (%d line)", err, __LINE__); exit(-1); }
size_t roundUp( int group_size, int global_size )
{
    int r = global_size % group_size;
    if( r == 0) {
        return global_size;
    } else {
        return global_size + group_size - r;
    }
}

std::string readKernel(const char* filename)
{
    std::ifstream ifs(filename, std::ios_base::in);
    if( !ifs.is_open() ) {
        std::cerr << "Failed to open file" << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}



void *handle;

#define DECLARE_FUNCTION_PTR(func_name) \
    decltype(func_name) (* func_name##_ptr) = nullptr

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


void init_opencl() {
// Load Dynamic Library
//    void *handle = dlopen("libOpenCL.so", RTLD_LAZY | RTLD_LOCAL);
    handle = dlopen("/vendor/lib64/libOpenCL.so", RTLD_LAZY | RTLD_LOCAL);

    if (handle == nullptr) {
        LOGI("OCL", "handle == NULL !!!!!");
        return;
    }

//    cl_int (*s)(cl_command_queue) = NULL;
//    s = reinterpret_cast<decltype(s)> (dlsym(handle, "clFinish"));

//    decltype(clFinish) (*s) = nullptr;
//    s = reinterpret_cast<decltype(s)> (dlsym(handle, "clFinish"));
//    s(NULL);


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
}


void additionCPU(float *a,float *b, float *r, int n)
{
    int i=0;
    for(i=0;i<n;i++){
        r[i] = a[i] + b[i];
    }
}


void go() {

    ///////////////// vector addition



    int N = 10000000;
    LOGI("OCL","N: %d\n", N );
    //Total size
    size_t sz = sizeof(float) * N;
    //Struct for time measure
    struct timeval start, end, timer;
    //Memory allocation for cpu(host)
    //vectorC = vectorA + vectorB
    float *h_a = (float*)malloc(sz);
    float *h_b = (float*)malloc(sz);
    float *h_r = (float*)malloc(sz);
    float *h_result = (float*)malloc(sz);
    for(int i = 0 ; i < N; i++ ) {
        h_a[i] = i;
        h_b[i] = N-i;
        h_r[i] = 0.0;
    }
    //-------------------------------------------------------------------------
    // Set up the OpenCL platform using whchever platform is "first"
    //-------------------------------------------------------------------------
    int err;
    int ndim = 1;
    cl_device_id        device_id;
    cl_context          context;
    cl_command_queue    commands;
    cl_program          program;
    cl_kernel           kernel;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;

    if (clGetPlatformIDs_ptr == nullptr) {
        LOGI("OCL", "clGetPlatformIDs_ptr == NULL!!");
    }

    err = clGetPlatformIDs_ptr(0, NULL, &numPlatforms);
    CHECK_ERROR(err);

    LOGI("OCL", "%d platforms", numPlatforms);

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
    //-------------------------------------------------------------------------
    // Set up the buffers, initialize matrices, and wirte them into
    // global memory
    //-------------------------------------------------------------------------
    //Memory allocation for gpu(device)
    cl_mem d_a, d_b, d_r;

    LOGI("OCL","create buffer start");
    d_a = clCreateBuffer_ptr(context, CL_MEM_READ_ONLY,
                             sz, NULL, NULL);

    d_b = clCreateBuffer_ptr(context, CL_MEM_READ_ONLY,
                             sz, NULL, NULL);

    d_r = clCreateBuffer_ptr(context, CL_MEM_READ_WRITE,
                             sz, NULL, NULL);


    LOGI("OCL","enqueue write buffer start");

    // Write the A and B matrices into compute device memory
    err = clEnqueueWriteBuffer_ptr(commands, d_a, CL_TRUE, 0,
                                   sz, h_a, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer_ptr(commands, d_b, CL_TRUE, 0,
                                   sz, h_b, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer_ptr(commands, d_r, CL_TRUE, 0,
                                   sz, h_r, 0, NULL, NULL);
    CHECK_ERROR(err);


    LOGI("OCL","read kernel");

    // Create the compute program from the source buffer
//    std::string strSource = readKernel("/data/local/tmp/hyonzin/vecAdd.cl");
//    LOGI("OCL","strSource.c_str()");
//    const char* source = strSource.c_str();
    const char* source = "__kernel void additionGPU(__global float *a, __global float *b, __global float *r, int n){ \
                                int gid = get_global_id(0); \
                                if( gid < n) { \
                                    r[gid] = a[gid] + b[gid]; \
                                } \
                             }";
    program = clCreateProgramWithSource_ptr(context, 1,
                                            (const char **) &source, NULL, &err);
    CHECK_ERROR(err);

    LOGI("OCL","read kernel ends");

    // Build the program
    char kernel_name[1000];
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
    strcpy(kernel_name, "additionGPU");
    // Create the compute kernel from the program
    kernel = clCreateKernel_ptr(program, kernel_name, &err);
    CHECK_ERROR(err);
    /*************************************************
        GPU Vector addition
    *************************************************/
    size_t local[] = {64};						// number of work-items per work-group
    size_t global[] = {(size_t)roundUp(local[0], N)};	// number of total work-items
    // Set the arguments to our compute kernel
    err = 0;
    err |= clSetKernelArg_ptr( kernel, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg_ptr( kernel, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg_ptr( kernel, 2, sizeof(cl_mem), &d_r);
    err |= clSetKernelArg_ptr( kernel, 3, sizeof(int), &N);
    CHECK_ERROR(err);
    //Time measure start
    gettimeofday(&start, NULL);
    err = clEnqueueNDRangeKernel_ptr(commands, kernel, ndim, NULL,
                                     global, local, 0, NULL, NULL);

    CHECK_ERROR(err);
    // Wait for the command to be completed before reading back results
    clFinish_ptr(commands);
    //Time measure end
    gettimeofday(&end, NULL);
    // Read back the results from the compute device
    err = clEnqueueReadBuffer_ptr(commands, d_r, CL_TRUE, 0,
                                  sz, h_result, 0, NULL, NULL);
    CHECK_ERROR(err);
    timersub(&end,&start,&timer);
    LOGI("OCL","GPU elapsend time: %lf\n", (timer.tv_usec / 1000.0 + timer.tv_sec *1000.0) );
    /*************************************************
        CPU Vector addition
    *************************************************/
    gettimeofday(&start, NULL);
    additionCPU(h_a,h_b,h_r,N);
    gettimeofday(&end, NULL);
    timersub(&end,&start,&timer);
    LOGI("OCL","CPU elapsend time: %lf\n", (timer.tv_usec / 1000.0 + timer.tv_sec *1000.0) );
    /*************************************************
          Verification
    *************************************************/
    for(int i=0;i<N;i++){
        if( h_r[i] != h_result[i] ){
            LOGI("OCL","Failed at %d\n",i);
            break;
        }
    }
    clReleaseProgram_ptr(program);
    clReleaseKernel_ptr(kernel);
    clReleaseMemObject_ptr(d_a);
    clReleaseMemObject_ptr(d_b);
    clReleaseMemObject_ptr(d_r);
    clReleaseCommandQueue_ptr(commands);
    clReleaseContext_ptr(context);
    free(h_result);
    free(h_a);
    free(h_b);
    free(h_r);


    ////////////////////





    dlclose(handle);
}

#define getMillisecond(start, end) \
	(end.tv_sec-start.tv_sec)*1000 + \
    (end.tv_usec-start.tv_usec)/1000.0
struct timeval start, end;
double timer[10];


Obs_gauge stair_guage;
graph_t global_graph= NULL;
tensor_t global_tensor_input = NULL;
tensor_t global_tensor_out = NULL;
int dims[] = {1,3,300,300}; // NCHW
//int dims[] = {1,3,300,300}; // NCHW
int num_detected_obj = 0;


float* out_data = NULL;
float* global_input = NULL;
float threshold = 0.5;

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_leek_my_1usb_DetectManager_detect(JNIEnv *env, jclass type, jbyteArray nv21Yuv_,
                                                   jint width, jint height) {
    num_detected_obj = 0;
    jbyte* const i = env->GetByteArrayElements(nv21Yuv_, NULL);

    go();

    /* Preprocessing */

    gettimeofday(&start, NULL);

    cv::Mat yuv(height+height/2, width, CV_8UC1,(uchar *)i);
    cv::Mat converted(height, width, CV_8UC3);
    cv::cvtColor(yuv, converted, CV_YUV2BGR_NV21);
    cv::resize(converted,converted,cv::Size(FIXED_HEIGHT,FIXED_WIDTH));
    //cv::imwrite("/sdcard/saved_images/leek.jpg",converted);
    converted.convertTo(converted,CV_32FC3);
    float* rgb_data = (float*)converted.data;

    gettimeofday(&end, NULL);
    timer[0] = getMillisecond(start, end);  // Preprocessing (convert, resize) ( ms)


    /*
    cv::Mat img = cv::imread("/sdcard/saved_images/ssd_dog.jpg");
    if( img.empty())
        return -1;
    cv::resize(img,img,cv::Size(FIXED_HEIGHT,FIXED_WIDTH));
    img.convertTo(img,CV_32FC3);
    float *rgb_data = (float*)img.data;
     */

    gettimeofday(&start, NULL);

    int hw = FIXED_HEIGHT * FIXED_WIDTH;
    float mean[3] = {127.5,127.5,127.5};
    for (int h = 0; h < FIXED_HEIGHT; h++)
    {
        for (int w = 0; w < FIXED_WIDTH; w++)
        {
//            for (int c = 0; c < 3; c++)
//            {
//                global_input[c * hw + h * FIXED_WIDTH + w] = 0.007843* (*rgb_data - mean[c]);
//                rgb_data++;
//            }
            // Loop Unrolling
            global_input[       h * FIXED_WIDTH + w] = 0.007843* (*(rgb_data  ) - 127.5);
            global_input[hw   + h * FIXED_WIDTH + w] = 0.007843* (*(rgb_data+1) - 127.5);
            global_input[hw*2 + h * FIXED_WIDTH + w] = 0.007843* (*(rgb_data+2) - 127.5);
            rgb_data+=3;
        }
    }


    gettimeofday(&end, NULL);
    timer[1] = getMillisecond(start, end);  // Preprocessing (Normalization)


    /* Inference */

    gettimeofday(&start, NULL);

    int res = detect(global_input,&out_data,global_graph,global_tensor_input,&global_tensor_out,&num_detected_obj,IMG_SIZE);
    //post_process_ssd("/sdcard/saved_images/leek.jpg",threshold,out_data,num_detected_obj,"/sdcard/saved_images/leek_processed.jpg");

    gettimeofday(&end, NULL);
    timer[2] = getMillisecond(start, end);  // Inference

    LOGI("night >> convert,resize", "%.3lf", timer[0]); //  2.64 ms
    LOGI("night >> normalize",      "%.3lf", timer[1]); //  1.99 ms
    LOGI("night >> inference",      "%.3lf", timer[2]); // 41.26 ms


    env->ReleaseByteArrayElements(nv21Yuv_, i, 0);
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

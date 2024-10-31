#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

#define MATRIX_SIZE 1300
#define CONV_MATRIX_SIZE 1298
#define KERNEL_SIZE 3
#define BLOCK_SIZE 16

int read_matrix(const char* matrix_in_filename, int32_t* input_matrix);
int write_matrix(const char* matrix_out_filename, int64_t* output_matrix);
void apply_convolution(int32_t* input_matrix, int64_t* output_matrix, int32_t* kernel_matrix);
char* readKernelSource(const char* filename);

int32_t kernel_matrix[] = { INT32_MAX, INT32_MAX, 15, 8, -3, -21, 17, 49, 101};

int main(){
    clock_t start, end;
    double cpu_time_used;
    // Allocation
    int32_t* input_matrix = (int32_t*) malloc(sizeof(int32_t) * MATRIX_SIZE * MATRIX_SIZE);
    int64_t* output_matrix = (int64_t*) malloc(sizeof(int64_t) * CONV_MATRIX_SIZE * CONV_MATRIX_SIZE);

    if (input_matrix == NULL || output_matrix == NULL) {
        fprintf(stderr, "Errore allocazione memoria\n");
        return EXIT_FAILURE;
    }

    // Reading input matrix
    if (read_matrix("matrix.txt", input_matrix) != 0) {
        free(input_matrix);
        free(output_matrix);
        return EXIT_FAILURE;
    }

    apply_convolution(input_matrix, output_matrix, kernel_matrix);

    if (write_matrix("convoluted_matrix.txt", output_matrix) != 0) {
        free(input_matrix);
        free(output_matrix);
        return EXIT_FAILURE;
    }

    // free memory
    free(input_matrix);
    free(output_matrix);

    return 0;
}

int read_matrix(const char *matrix_in_filename, int32_t* input_matrix){
    FILE *matrix_file = fopen(matrix_in_filename, "r");
    if(matrix_file == NULL){
        fprintf(stderr, "Errore apertura file matrice input\n");
        return EXIT_FAILURE;
    }

    for(int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++){
        if(fscanf(matrix_file, "%u", &input_matrix[i]) != 1){
            fprintf(stderr, "Errore lettura valori dal file\n");
            fclose(matrix_file);
            return EXIT_FAILURE;
        }
    }

    fclose(matrix_file);
    return 0;
}

int write_matrix(const char *matrix_out_filename, int64_t* output_matrix){
    FILE *conv_matrix = fopen(matrix_out_filename, "w");
    if(conv_matrix == NULL){
        fprintf(stderr, "Errore apertura file matrice convoluta\n");
        return EXIT_FAILURE;
    }

    for(int i = 0; i < CONV_MATRIX_SIZE; i++){
        for(int j = 0; j < CONV_MATRIX_SIZE; j++){
            fprintf(conv_matrix, "%ld", output_matrix[i * CONV_MATRIX_SIZE + j]);
            if(j < CONV_MATRIX_SIZE - 1){
                fprintf(conv_matrix, " ");
            }
        }
        fprintf(conv_matrix, "\n");
    }

    fclose(conv_matrix);
    return 0;
}

void apply_convolution(int32_t* input_matrix, int64_t* output_matrix, int32_t* kernel_matrix) {
    cl_int err;
    cl_ulong time_start;
    cl_ulong time_end;
    double nanoSeconds;
    cl_command_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };

    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
    char* programSource = readKernelSource("kernel.cl");

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&programSource, NULL, NULL);

    cl_int build_err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (build_err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *build_log = (char *)malloc(log_size + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
        fprintf(stderr, "Build Log: %s\n", build_log);
        free(build_log);
        return ;
    }

    cl_kernel kernel = clCreateKernel(program, "convolution", NULL);

    cl_event kernel_event;

    cl_mem startingMatrixBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int32_t) * MATRIX_SIZE * MATRIX_SIZE, NULL, NULL);
    clEnqueueWriteBuffer(queue, startingMatrixBuffer, CL_TRUE, 0, sizeof(int32_t) * MATRIX_SIZE * MATRIX_SIZE, input_matrix, 0, NULL, &kernel_event);

    clWaitForEvents(1, &kernel_event);

    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    nanoSeconds = time_end-time_start;

    clReleaseEvent(kernel_event);

    cl_mem convMatrixBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int64_t) * CONV_MATRIX_SIZE * CONV_MATRIX_SIZE, NULL, NULL);

    cl_mem kernelBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int32_t) * KERNEL_SIZE * KERNEL_SIZE, NULL, NULL);
    clEnqueueWriteBuffer(queue, kernelBuffer, CL_TRUE, 0, sizeof(int32_t) * KERNEL_SIZE * KERNEL_SIZE, kernel_matrix, 0, NULL, &kernel_event);

    clWaitForEvents(1, &kernel_event);

    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    nanoSeconds += time_end-time_start;
    printf("%0.3f\n",nanoSeconds / 1000000.0);

    clReleaseEvent(kernel_event);

    int32_t matrix_size = (int32_t) CONV_MATRIX_SIZE;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &startingMatrixBuffer);   // First argument: input array
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &convMatrixBuffer);       // Second argument: output array
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &kernelBuffer);           // Third argument: kernel array
    clSetKernelArg(kernel, 3, sizeof(int32_t), &matrix_size);           // Fourth argument: size of convoluted matrix

    size_t globalWorkSize[1] = { CONV_MATRIX_SIZE*CONV_MATRIX_SIZE };
    size_t localWorkSize[1] = { 121 };

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, &kernel_event);
    if (err != CL_SUCCESS) {
        printf("Error in clEnqueueNDRangeKernel: %d\n", err);
    }

    clWaitForEvents(1, &kernel_event);

    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    nanoSeconds = time_end-time_start;
    printf("%0.3f\n",nanoSeconds / 1000000.0);

    clReleaseEvent(kernel_event);

    clEnqueueReadBuffer(queue, convMatrixBuffer, CL_TRUE, 0, sizeof(int64_t) * CONV_MATRIX_SIZE * CONV_MATRIX_SIZE, output_matrix, 0, NULL, &kernel_event);

    clWaitForEvents(1, &kernel_event);

    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    nanoSeconds = time_end-time_start;
    printf("%0.3f\n",nanoSeconds / 1000000.0);

    clReleaseEvent(kernel_event);

    free(programSource);
    clReleaseMemObject(startingMatrixBuffer);
    clReleaseMemObject(convMatrixBuffer);
    clReleaseMemObject(kernelBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

char* readKernelSource(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    fseek(file, 0, SEEK_END);
    size_t fileSize = ftell(file);
    rewind(file);
    char* source = (char*)malloc(fileSize + 1);
    source[fileSize] = '\0';
    fread(source, sizeof(char), fileSize, file);
    fclose(file);
    return source;
}

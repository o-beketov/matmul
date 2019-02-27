// nvcc matmul.cu --gpu-architecture=compute_20 -Xcompiler -fopenmp -Xlinker -lgomp -O3 -o matmul

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <assert.h>

#ifndef _WIN32
#include <unistd.h>
#include <pthread.h>
#else
#include <windows.h>
#endif

#define L 1024
#define M 2000
#define N 777

int strapNum_i = 3;
int strapNum_j = 5;

float ** A = NULL, ** B = NULL, ** Cc = NULL, ** Cg = NULL, * Al = NULL, *Bl = NULL;

#define print_to_file(str1, str2)		\
{										\
   FILE * pFile;						\
   pFile = fopen ("results.log", "a+"); \
   fprintf (pFile, (str1), (str2));		\
   fclose (pFile);						\
}

#define checkCuda(param) checkCuda_debug(param, __FILE__,  __FUNCTION__,  __LINE__)
//#define checkCuda(param) checkCuda_release(param)

cudaError_t checkCuda_debug(cudaError_t result, const char * file,  const char * func,  int line)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "%s:%s:%d: CUDA Runtime Error: %s\n", 
            file, func, line, cudaGetErrorString(result));
  }
  return result;
}

cudaError_t checkCuda_release(cudaError_t result)
{
  return result;
}

void CUDAEvent(cudaStream_t stream) {
    cudaEvent_t syncEvent;
    cudaEventCreate(&syncEvent);
    cudaEventRecord(syncEvent, stream);
    cudaEventSynchronize(syncEvent);
    cudaEventDestroy(syncEvent);
}

int is(int a, int b) {
    int res = a/b;
    if (a%b != 0) return (res+1);
    else return res;
}

void AllocInitialData() {

    A = (float**) malloc(L * sizeof(float*));
    for (int i = 0; i < L; i++)
        A[i] = (float*) malloc(M * sizeof(float));

    B = (float**) malloc(M * sizeof(float*));
    for (int i = 0; i < M; i++)
        B[i] = (float*) malloc(N * sizeof(float));

    Cc = (float**) malloc(L * sizeof(float*));
    Cg = (float**) malloc(L * sizeof(float*));
    for (int i = 0; i < L; i++) {
        Cc[i] = (float*) malloc(N * sizeof(float));
        Cg[i] = (float*) malloc(N * sizeof(float));
    }

    Al = (float*) malloc(L * M * sizeof(float));
    Bl = (float*) malloc(M * N * sizeof(float));

	printf("\n#C:                    %d elements", L*N);
	printf("\nsizeof(C):           %10.3lf Mb", (float)L*N * sizeof(float) / (1024*1024));
    printf("\nsizeof(A+B+Cc+Cg):   %10.3lf Mb", (float)(L*M + M*N + 2*L*N) * sizeof(float) / (1024*1024));
	print_to_file("L=%d ", L);
	print_to_file("M=%d ", M);
	print_to_file("N=%d ", N);
	print_to_file("sizeof(C)=%fMb ", (float)L*N * sizeof(float) / (1024*1024));
}

void FillInitialData() {
    for (int i = 0; i < L; i++)
        for (int j = 0; j < M; j++)
            A[i][j] = 1+sin((float)i*j+i);

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            B[i][j] = 2+cos((float)3*i-j);

    for (int i = 0; i < L; i++)
        for (int j = 0; j < N; j++) {
            Cc[i][j] = 0;
            Cg[i][j] = 0;
        }
}

void CPU_2d_arrays() {
    clock_t timer;

    timer = clock ();
    for(int i = 0; i < L; i++)
        for(int j = 0; j < N; j++)
            for(int k = 0; k < M; k++)
                Cc[i][j] += A[i][k]*B[k][j];
    timer = clock() - timer;
	
#ifndef _WIN32
	timer = timer/1000;
#endif
    printf("\nCalc time:           %10.3lf s = %d ms\n\n", (float)timer/1000, timer);
}

__global__ void kernel_ppcg_parameterized(float * A, float * B, float * C, int L_, int M_, int N_)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ float shared_A[32][32];
    __shared__ float shared_B[32][32];
    float private_C[1][2];

	int L0 = L_;
	int L1 = L_-1;
	int L16 = L_-1-16;
	int M0 = M_;
	int M1 = M_-1;
	int M16 = M_-1-16;
	int N0 = N_;
	int N1 = N_-1;
	int N16 = N_-1-16;

	  if (32 * b0 + t0 <= L1 && 32 * b1 + t1 <= N1) {
        private_C[0][0] = C[(32 * b0 + t0) * N0 + (32 * b1 + t1)];
        if (32 * b1 + t1 <= N16)
          private_C[0][1] = C[(32 * b0 + t0) * N0 + (32 * b1 + t1 + 16)];
      }
      for (int c2 = 0; c2 <= M1; c2 += 32) {
        if (32 * b0 + t0 <= L1) {
	      int Q;
		  if (31 < -c2 + M1)
			  Q = 31;
		  else
			  Q = -c2 + M1;
          for (int c4 = t1; c4 <= Q; c4 += 16)
            shared_A[t0][c4] = A[(32 * b0 + t0) * M0 + (c2 + c4)];
		}
        if (t0 + c2 <= M1) {
	      int Q;
		  if (31 < -32 * b1 + N1)
			  Q = 31;
		  else
			  Q = -32 * b1 + N1;
          for (int c4 = t1; c4 <= Q; c4 += 16)
            shared_B[t0][c4] = B[(t0 + c2) * N0 + (32 * b1 + c4)];
		}
        __syncthreads();
        if (32 * b0 + t0 <= L1 && 32 * b1 + t1 <= N1) {
	      int Q;
		  if (31 < -c2 + M1)
			  Q = 31;
		  else
			  Q = -c2 + M1;
          for (int c3 = 0; c3 <= Q; c3 += 1) {
            private_C[0][0] += (shared_A[t0][c3] * shared_B[c3][t1]);
            if (32 * b1 + t1 <= N16)
              private_C[0][1] += (shared_A[t0][c3] * shared_B[c3][t1 + 16]);
          }
		}
        __syncthreads();
      }
      if (32 * b0 + t0 <= L1 && 32 * b1 + t1 <= N1) {
        C[(32 * b0 + t0) * N0 + (32 * b1 + t1)] = private_C[0][0];
        if (32 * b1 + t1 <= N16)
          C[(32 * b0 + t0) * N0 + (32 * b1 + t1 + 16)] = private_C[0][1];
      }
}

void serialize_i(float * Ah, int e) {
    int strap_i = e/strapNum_j;
    int strap_j = e%strapNum_j;
    int L_ = is(L, strapNum_i);
    int M_ = M;
    int i = strap_i*L_;
	int j = 0;
    for(int i_ = i; i_ < min(i+L_, L); i_++) {
        for(int j_ = j; j_ < min(j+M_, M); j_++) {
            Ah[(i_-i)*M_+(j_-j)] = A[i_][j_];
        }
    }
}

void serialize_j(float * Bh, int e) {
    int strap_i = e/strapNum_j;
    int strap_j = e%strapNum_j;
    int M_ = M;
    int N_ = is(N, strapNum_j);
    int i = 0;
	int j = strap_j*N_;
    for(int i_ = i; i_ < min(i+M_, M); i_++) {
        for(int j_ = j; j_ < min(j+N_, N); j_++) {
            Bh[(i_-i)*N_+(j_-j)] = B[i_][j_];
        }
    }
}

void deserialize(float * Ch, int e) {
    int strap_i = e/strapNum_j;
    int strap_j = e%strapNum_j;
    int L_ = is(L, strapNum_i);
    int N_ = is(N, strapNum_j);
    int i = strap_i*L_;
    int j = strap_j*N_;
    for(int i_ = i; i_ < min(i+L_, L); i_++) {
        for(int j_ = j; j_ < min(j+N_, N); j_++) {
            Cg[i_][j_] = Ch[(i_-i)*N_+(j_-j)];
        }
    }
}

void MatMul_GPU_asynch(int dev) {
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, dev);
	printf("\n%s\n",  devProp.name); 

    int L_ = is(L, strapNum_i);
    int M_ = M;
    int N_ = is(N, strapNum_j);
    
	int launchNum = strapNum_i*strapNum_j;
	int threadsPerBlock = 256;
	int blocksPerGrid = is(L_*N_, threadsPerBlock);
    if (blocksPerGrid >= 65536) {
        threadsPerBlock = 512;
        blocksPerGrid = is(L_*N_, threadsPerBlock);
    }
    if (blocksPerGrid >= 65536) {
        threadsPerBlock = 1024;
        blocksPerGrid = is(L_*N_, threadsPerBlock);
    }
    if (blocksPerGrid >= 65536) {
        printf("\n\t Fatal error! blocksPerGrid >= 65536 \n");
    }
    printf("\nblocksPerGrid = %d\n", blocksPerGrid);
    printf("\nlaunchNum = %d\n", launchNum);

    clock_t TIMER, timer;
    double wall_timer = 0;
    TIMER = clock();
    #ifdef _OPENMP
    wall_timer = omp_get_wtime();
    #endif

    cudaError_t cudaStatus;

	cudaSetDevice(dev);
 
    float *Ah0 = NULL, *Bh0 = NULL, *Ch0 = NULL;
    float *Ad0 = NULL, *Bd0 = NULL, *Cd0 = NULL;
    float *Ah1 = NULL, *Bh1 = NULL, *Ch1 = NULL;
    float *Ad1 = NULL, *Bd1 = NULL, *Cd1 = NULL;

    if (launchNum>1) {
        checkCuda ( cudaMallocHost ((void**)&Ah0, L_ * M_ * sizeof(float)) );
        checkCuda ( cudaMallocHost ((void**)&Bh0, M_ * N_ * sizeof(float)) );
        checkCuda ( cudaMallocHost ((void**)&Ch0, L_ * N_ * sizeof(float)) );
    }
    else {
        Ah0 = (float *) malloc (L_ * M_ * sizeof(float));
        Bh0 = (float *) malloc (M_ * N_ * sizeof(float));
        Ch0 = (float *) malloc (L_ * N_ * sizeof(float));
    }
	
	checkCuda ( cudaMalloc ((void**)&Ad0, L_ * M_ * sizeof(float)) );
    checkCuda ( cudaMalloc ((void**)&Bd0, M_ * N_ * sizeof(float)) );
    checkCuda ( cudaMalloc ((void**)&Cd0, L_ * N_ * sizeof(float)) );
    
	if (launchNum>1) {
        checkCuda ( cudaMallocHost ((void**)&Ah1, L_ * M_ * sizeof(float)) );
        checkCuda ( cudaMallocHost ((void**)&Bh1, M_ * N_ * sizeof(float)) );
        checkCuda ( cudaMallocHost ((void**)&Ch1, L_ * N_ * sizeof(float)) );
        
        checkCuda ( cudaMalloc ((void**)&Ad1, L_ * M_ * sizeof(float)) );
        checkCuda ( cudaMalloc ((void**)&Bd1, M_ * N_ * sizeof(float)) );
        checkCuda ( cudaMalloc ((void**)&Cd1, L_ * N_ * sizeof(float)) );
	}

	cudaStream_t stream1 = NULL, stream2 = NULL, stream3 = NULL;
	checkCuda ( cudaStreamCreate(&stream1) );
	checkCuda ( cudaStreamCreate(&stream2) );
	checkCuda ( cudaStreamCreate(&stream3) );

    timer = clock();

    serialize_i(Ah0, 0);
    checkCuda ( cudaMemcpy(Ad0, Ah0, L_*M_ * sizeof(float), cudaMemcpyHostToDevice) );
    if (launchNum>1) serialize_i(Ah1, 1);

    serialize_j(Bh0, 0);
    checkCuda ( cudaMemcpy(Bd0, Bh0, M_*N_ * sizeof(float), cudaMemcpyHostToDevice) );
    if (launchNum>1) serialize_j(Bh1, 1);

#ifndef _WIN32
    pthread_mutex_t s1, s2, s3, f1, f2, f3;
    pthread_mutex_init(&s1, NULL);
    pthread_mutex_init(&s2, NULL);
    pthread_mutex_init(&f1, NULL);
    pthread_mutex_init(&f2, NULL);
    pthread_mutex_lock(&f1);
    pthread_mutex_lock(&f2);
#else
    HANDLE s1 = CreateSemaphore(NULL, 1, 1, NULL);
    HANDLE s2 = CreateSemaphore(NULL, 1, 1, NULL);
    HANDLE f1 = CreateSemaphore(NULL, 0, 1, NULL);
    HANDLE f2 = CreateSemaphore(NULL, 0, 1, NULL);
#endif

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            for (int e = 0; e < launchNum; e++)
            {
                //////////////////////////////////////////////
                if (e < launchNum-2)
					if (e%2 == 0) {
						serialize_i(Ah0, e+2);
						serialize_j(Bh0, e+2);
					}
					else {
						serialize_i(Ah1, e+2);
						serialize_j(Bh1, e+2);
					}
				//////////////////////////////////////////////
                if (e > 1) {
                    if (e%2 == 0)
                        deserialize(Ch0, e-2);
                    else
                        deserialize(Ch1, e-2);
                }
				//////////////////////////////////////////////
#ifndef _WIN32
                pthread_mutex_lock(&f1);
                pthread_mutex_unlock(&s1);
#else
                WaitForSingleObject(f1, INFINITE);
                ReleaseSemaphore(s1, 1, NULL);
#endif
            }
        }

        #pragma omp section
        {
            for (int e = 0; e < launchNum; e++)
            {
#ifndef _WIN32
                pthread_mutex_lock(&s1);
#else
                WaitForSingleObject(s1, INFINITE);
#endif
				//////////////////////////////////////////////
                if (e%2 == 0) {
                    cudaMemset(Cd0, 0, L_*N_ * sizeof(float));
					dim3 k0_dimBlock(16, 32);
					dim3 k0_dimGrid(is(N_,32), is(L_,32));
					kernel_ppcg_parameterized<<<k0_dimGrid, k0_dimBlock, 0, stream1>>>(Ad0, Bd0, Cd0, L_, M_, N_);

					if (launchNum>1) {
                    {
                        checkCuda ( cudaMemcpyAsync(Ad1, Ah1, L_*M_ * sizeof(float), cudaMemcpyHostToDevice, stream2) );
                    }
                    checkCuda ( cudaMemcpyAsync(Bd1, Bh1, M_*N_ * sizeof(float), cudaMemcpyHostToDevice, stream2) );
                    checkCuda ( cudaMemcpyAsync(Ch1, Cd1, L_*N_ * sizeof(float), cudaMemcpyDeviceToHost, stream3) );
					}
                }
                else {
                    cudaMemset(Cd1, 0, L_*N_ * sizeof(float));
                    dim3 k0_dimBlock(16, 32);
					dim3 k0_dimGrid(is(N_,32), is(L_,32));
					kernel_ppcg_parameterized<<<k0_dimGrid, k0_dimBlock, 0, stream1>>>(Ad1, Bd1, Cd1, L_, M_, N_);
                    if (launchNum>1) {
                        checkCuda ( cudaMemcpyAsync(Ad0, Ah0, L_*M_ * sizeof(float), cudaMemcpyHostToDevice, stream2) );
                    }
                    checkCuda ( cudaMemcpyAsync(Bd0, Bh0, M_*N_ * sizeof(float), cudaMemcpyHostToDevice, stream2) );
                    checkCuda ( cudaMemcpyAsync(Ch0, Cd0, L_*N_ * sizeof(float), cudaMemcpyDeviceToHost, stream3) );             
                }
                cudaDeviceSynchronize();
				//////////////////////////////////////////////
#ifndef _WIN32
                pthread_mutex_unlock(&f1);
#else
                ReleaseSemaphore(f1, 1, NULL);
#endif
            }
        }
    }

    if ((launchNum-1)%2 == 0) {
		if (launchNum>1)deserialize(Ch1, launchNum-2);
		cudaMemcpy(Ch0, Cd0, L_*N_ * sizeof(float), cudaMemcpyDeviceToHost);
		deserialize(Ch0, launchNum-1);
	}
	else {
		if (launchNum>1)deserialize(Ch0, launchNum-2);
		cudaMemcpy(Ch1, Cd1, L_*N_ * sizeof(float), cudaMemcpyDeviceToHost);
		deserialize(Ch1, launchNum-1);
	}

    timer = clock() - timer;

    cudaFreeHost(Ah0);
	cudaFreeHost(Bh0);
	cudaFreeHost(Ch0);
	cudaFree(Ad0);
	cudaFree(Bd0);
	cudaFree(Cd0);
	if (launchNum>1) {
		cudaFreeHost(Ah1);
		cudaFreeHost(Bh1);
		cudaFreeHost(Ch1);
		cudaFree(Ad1);
		cudaFree(Bd1);
		cudaFree(Cd1);
	}
    
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    TIMER = clock() - TIMER;
    #ifdef _OPENMP
    wall_timer = omp_get_wtime() - wall_timer;
    #endif
	printf("\n %d", launchNum);
    printf("\n%s: full time %f ms, wall %f sec; calc time %f ms", devProp.name, (float)TIMER, wall_timer, (float)timer);

#ifndef _WIN32
	timer = timer/1000;
	TIMER = TIMER/1000;
#endif
    printf("\nCalc time:           %10.3lf s = %d ms", (float)timer/1000, timer);
	print_to_file("GPU_wall_time_w_allocs=%fsec ", (float)TIMER/1000);
	print_to_file("GPU_calc_time=%fsec ", (float)timer/1000);
}

float abss(float a) {
    return (a > 0) ? a : -a;
}

void CompareResults() {

    int calc = 0, index = 0;
    float max = 0;
	
	printf("\n\n");
    printf("*********************************************\n");
    printf("----------==========STATS==========----------\n");
    printf("*********************************************\n");
	
    for(int i = 0; i < L; i++)
    {
        for(int j = 0; j < N; j++)
        {
            if (abss(Cc[i][j] - Cg[i][j]) > 0.000008) {
                calc++;
                if (abss(Cc[i][j] - Cg[i][j]) > max) {
                    max = abss(Cc[i][j] - Cg[i][j]);
                    index = i*N + j;
                }
            }
        }
    }
    int j = (index)/N;
    int i = (index)%N;
	printf("\ncpu:                             %10.5lf", Cc[j][i]);
	printf("\ngpu:                             %10.5lf", Cg[j][i]);
	printf("\nmax deviation:                   %10.5lf", max);
    printf("\ndeviations that exceed 0.000008: %10.5lf%% \n", (float)calc/(M*N)*100);
	print_to_file("maxDev=%f ", max);
}

int main() {
    printf("*********************************************\n");
    printf("----------========Data Init========----------\n");
    printf("*********************************************\n");

    clock_t timer;
    timer = clock();
    AllocInitialData();
    FillInitialData();
    printf("\nInit&Fill time:      %10.3lf ms", (float)(clock() - timer));

    printf("\n\n");
    printf("*********************************************\n");
    printf("----------===========HOST==========----------\n");
    printf("*********************************************\n");

    CPU_2d_arrays();

    printf("\n\n");
    printf("*********************************************\n");
    printf("----------===========GPU===========----------\n");
    printf("*********************************************\n");
    
    MatMul_GPU_asynch(0);
	
    CompareResults();
    cudaDeviceReset();
	print_to_file("%s", "\n");

    return 0;
}

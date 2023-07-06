#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include "curand_kernel.h"
#include "cublas_v2.h"
#include "device_functions.h"
#include<time.h>
#include<iostream>
#include<thread>
#include<mutex>
#include<condition_variable>
#include <fstream>
#include <cooperative_groups.h>
#include <sstream>
#include <string>
#include <iomanip>
#include"semaphore.h"
#include<vector>

#include <stdio.h>

/****************************************************************************** 
 * @description: hyperparameter settings
 * @return {*}
 *******************************************************************************/
#define INDIVNUM 512//individual numbers
#define INDIVNUM_ISLAND 256//individual numbers per island
#define SHARED_CAPACITY 10240//
#define DIMENSION 50
#define CPU_THREAD_NUM 1000//CPU thread number
#define ITER_NUM 1000
#define INTERVAL_TRANSFER 50//the interval of knowledge transfer
#define INTERVAL_MIGRA 20//the interval of population migration
#define TRANSFER_NUM 50//the individual number for knowledge transfer
#define T 1000//task number
#define ISLAND_NUM 2//island number
#define VAL_TYPE 4 //float type, 8 is double type
#define BANKSIZE 32 
#define WARPSIZE 32
#define BLOCK_NUM ((INDIVNUM - 1) / (INDIV_PERBLOCK * 2) + 1)
#define F 0.5
#define CROSSOVER_RATE 0.6
#define ETA 2.f //the parameter for simulate binary crossover
#define B_BASEVAL 100
#define PM 2.f
#define DM 5.f //polynomial mutation parameters
#define THREAD_FOR_OPERA 416  //(SHARED_CAPACITY / (VAL_TYPE * DIMENSION * 3)) //每个block里一个thread服务一个个体
#define THREAD_FOR_TRANSFER 96 //用于数据传输的线程数
#define _THREAD_NUM (THREAD_FOR_OPERA + THREAD_FOR_TRANSFER) //The thread number for a kernel function
#define E 2.718282 
#define FULL_MASK 0xffffffff
#define TASK_TYPENUM 7 //evaluation function number
#define MIGRA_NUM 50
#define MIGRA_PERBLOCK MIGRA_NUM / BLOCK_NUM
#define MIGRA_PROP 1/8 //The proportion of individuals per block used for population migration
#define MIGRA_NUM (INDIV_PERBLOCK * MIGRA_PROP) //indivdual number of population migration
#define BESTNUM_PERTASK 50 
#define PI 3.1415926 
#define LOOPTIME 1 
#define SELECT_INTERVAL INDIV_PERBLOCK * ISLAND_NUM * INTERVAL_TRANSFER


#define INDIV_PERBLOCK 8//(SHARED_CAPACITY / (VAL_TYPE * 3 * DIMENSION))
#define STREAM_NUM 10 //stream number

using namespace std;


/****************************************************************************** 
 * @description: for bitonic sort
 * @return {*}
 *******************************************************************************/
struct DS {
	float eval[BANKSIZE];
	float* pointer[BANKSIZE];
	float* eval_pointer[BANKSIZE];
};


/****************************************************************************** 
 * @description: Some global variables on CPU
 * @return {*}
 *******************************************************************************/
semaphore pv(5);
bool achieve[T];
mutex iter_mutex;
condition_variable wait_line, finish_line;

int max_iter[T];
int tasks_type[T];
float** M_cpu = new float*[T];
float** b_cpu = new float*[T];
float** INDIVIDUALS_CPU = new float*[CPU_THREAD_NUM];
float** INDIV_VAL_CPU = new float*[CPU_THREAD_NUM];
float** INDIV_BEST_CPU = new float*[CPU_THREAD_NUM];
float** INDIVVAL_BEST_CPU = new float*[CPU_THREAD_NUM];
int achieve_num = 0, iter_time = 0, waiting_num = 0, transfer_num = T, finish_num = 0;


/****************************************************************************** 
 * @description: Some global variables on GPU
 * @return {*}
 *******************************************************************************/
int* task_type[T];
curandState* devStates;
int* select_interval[T];
float** M = new float*[T];
float** b = new float*[T];
__device__ float Weierstrass_para;
int** syncval = new int*[CPU_THREAD_NUM];//for block synchronization
DS** indiv_sort = new DS*[CPU_THREAD_NUM];
__constant__ float Range[TASK_TYPENUM * 2];//dimension range of task
float** indiv_val = new float*[CPU_THREAD_NUM];
float** INDIVIDUALS = new float*[CPU_THREAD_NUM];
float** INDIV_BEST_GPU = new float*[CPU_THREAD_NUM];
float** INDIVVAL_BEST_GPU = new float*[CPU_THREAD_NUM];




/****************************************************************************** 
 * @description: Generate random numbers from a uniform distribution
 * @param {float*} value
 * @param {curandState*} state
 * @return { }
 *******************************************************************************/
__device__ void rand_uniform(float* value, curandState* state) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	*value = curand_uniform(&state[tid]);
}

/****************************************************************************** 
 * @description: Population initialization
 * @param {curandState*} states
 * @param {float*} INDIVIDUALS
 * @param {int*} type
 * @param {float*} eval
 * @return { }
 *******************************************************************************/
__global__ void pop_init(curandState* states, float* INDIVIDUALS, int* type, float* eval) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const int t_n = gridDim.x * blockDim.x;
	while (tid < INDIVNUM * DIMENSION) {
		INDIVIDUALS[tid] = curand_uniform(&states[blockIdx.x * blockDim.x + threadIdx.x]) * (Range[*type * 2 + 1] - Range[*type * 2]) + Range[*type * 2];
		tid += t_n;
	}
	tid = blockIdx.x * blockDim.x + threadIdx.x;
	while (tid < INDIVNUM) {
		eval[tid] = INT_MAX;
		tid += t_n;
	}
}


/****************************************************************************** 
 * @description: Random number initialization, giving each RNG an initial seed
 * @param {curandState*} states
 * @param {int} seed
 * @return { }
 *******************************************************************************/
__global__ void curandInit(curandState* states, int seed) {
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	int interval = blockDim.x * gridDim.x;
	int total_randVal = _THREAD_NUM * BLOCK_NUM * STREAM_NUM;
	while (tid < total_randVal) {
		curand_init(seed, tid, 0, &states[tid]);
		tid += interval;
	}
}

/****************************************************************************** 
 * @description: Initialize random matrix for transforming decision variables
 * @param {curandState*} states
 * @param {float**} M
 * @param {float**} b
 * @param {int*} task_choice
 * @return { }
 *******************************************************************************/
__global__ void para_init(curandState* states, float** M, float** b, int* task_choice) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const int t_n = gridDim.x * blockDim.x;
	while (tid < T * DIMENSION) {
		int type = task_choice[tid / DIMENSION];
		int range = Range[type * 2 + 1] - Range[type * 2];
		M[tid / DIMENSION][tid % DIMENSION] = curand_uniform(&states[tid % (_THREAD_NUM * BLOCK_NUM * STREAM_NUM)]);
		b[tid / DIMENSION][tid % DIMENSION] = curand_uniform(&states[tid % (_THREAD_NUM * BLOCK_NUM * STREAM_NUM)]) * range;
		
		tid += t_n;
	}
}

/****************************************************************************** 
 * @description: Initialize the dimension range of the task
 * @return { }
 *******************************************************************************/
void rangeInit() {
	float range_val[TASK_TYPENUM * 2];
	range_val[0] = -100;
	range_val[1] = 100;
	range_val[2] = -50;
	range_val[3] = 50;
	range_val[4] = -50;
	range_val[5] = 50;
	range_val[6] = -50;
	range_val[7] = 50;
	range_val[8] = -100;
	range_val[9] = 100;
	range_val[10] = -0.5;
	range_val[11] = 0.5;
	range_val[12] = -500;
	range_val[13] = 500;
	cudaMemcpyToSymbol(Range, range_val, TASK_TYPENUM * 2 * sizeof(float));
}


/****************************************************************************** 
 * @description: Move the entire population of a task back to the CPU
 * @param {int} cthread_idx
 * @param {cudaStream_t*} stream
 * @return { }
 *******************************************************************************/
void popTransfer_(int cthread_idx, cudaStream_t* stream) {

		INDIVIDUALS_CPU[cthread_idx] = new float[INDIVNUM * DIMENSION];
		INDIV_VAL_CPU[cthread_idx] = new float[INDIVNUM];
		cudaMemcpyAsync(INDIVIDUALS_CPU[cthread_idx], INDIVIDUALS[cthread_idx], INDIVNUM * DIMENSION * sizeof(float), cudaMemcpyDeviceToHost, *stream);
		cudaMemcpyAsync(INDIV_VAL_CPU[cthread_idx], indiv_val[cthread_idx], INDIVNUM * sizeof(float), cudaMemcpyDeviceToHost, *stream);

}


/****************************************************************************** 
 * @description: Overall initialization before the evolution process starts, such as transferring data to global memory, initialization of GPU random number generator, etc.
 * @return { }
 *******************************************************************************/
void initialization() {

	//atomic variables for block synchronization
	int syncval_cpu[((INTERVAL_TRANSFER - 1) / LOOPTIME + 1) * ISLAND_NUM];
	for(int i = 0; i < ((INTERVAL_TRANSFER - 1) / LOOPTIME + 1) * ISLAND_NUM; ++i){
		syncval_cpu[i] = 0;
	}

	rangeInit();

	int* task_choice;
	cudaMalloc((void**)&task_choice, sizeof(int) * T);
	srand(0);

	for(int i = 0; i < CPU_THREAD_NUM; ++i){
		tasks_type[i] = rand() % 7;
		if(i == 0){
			tasks_type[i] = 6;
		}
	}
	srand(time(0));

	for (int i = 0; i < CPU_THREAD_NUM; ++i) {
        achieve[i] = false;
        max_iter[i] = 3;
		INDIVIDUALS[i] = new float[INDIVNUM * DIMENSION];
		indiv_val[i] = new float[INDIVNUM];
		indiv_sort[i] = new DS[(INDIVNUM - 1) / BANKSIZE + 1];//上取整
		cudaMalloc((void**)&select_interval[i], SELECT_INTERVAL * sizeof(int));
		cudaMalloc((void**)&INDIVIDUALS[i], INDIVNUM * DIMENSION * sizeof(float));
		cudaMalloc((void**)&indiv_val[i], INDIVNUM * sizeof(float));
		cudaMalloc((void**)&INDIV_BEST_GPU[i], BESTNUM_PERTASK * DIMENSION * sizeof(float));
		cudaMalloc((void**)&INDIVVAL_BEST_GPU[i], BESTNUM_PERTASK * sizeof(float));
		cudaMalloc((void**)&indiv_sort[i], ((INDIVNUM - 1) / BANKSIZE + 1) * BANKSIZE * (sizeof(float) + sizeof(float*) + sizeof(float*)));
		cudaMalloc((void**)&task_type[i], sizeof(int) * 2);

		cudaMemcpy(task_type[i], &tasks_type[i], sizeof(int), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&syncval[i], sizeof(int) * ((INTERVAL_TRANSFER - 1) / LOOPTIME + 1) * ISLAND_NUM);
		cudaMemcpy(syncval[i], &syncval_cpu, sizeof(int) * ((INTERVAL_TRANSFER - 1) / LOOPTIME + 1) * ISLAND_NUM, cudaMemcpyHostToDevice);

	}
		cudaMemcpy(task_choice, &tasks_type, sizeof(int) * T, cudaMemcpyHostToDevice);
	
	for (int i = 0; i < T; ++i) {
		float* devM_1d, *devb_1d;
		cudaMalloc((void**)&devM_1d, DIMENSION * sizeof(float));
		M_cpu[i] = devM_1d;
		cudaMalloc((void**)&devb_1d, DIMENSION * sizeof(float));
		b_cpu[i] = devb_1d;

	}
	cudaMalloc((void**)&M, sizeof(float*) * T);
	cudaMalloc((void**)&b, sizeof(float*) * T);
	cudaMemcpy(M, M_cpu, sizeof(float*) * T, cudaMemcpyHostToDevice);
	cudaMemcpy(b, b_cpu, sizeof(float*) * T, cudaMemcpyHostToDevice);


	float a_ = 1, b_ = 1, res = 0;
	for (int i = 0; i <= 20; ++i) {
		res += a_ * cosf(2 * PI * b_ * 0.5);
		a_ *= 0.5;
		b_ *= 3;
	}

	cudaMemcpyToSymbol(Weierstrass_para, &res, sizeof(float));

	int total_threadnum = _THREAD_NUM * BLOCK_NUM * STREAM_NUM;

	cudaMalloc((void**)&devStates, sizeof(curandState) * total_threadnum);

	int blocknum = STREAM_NUM;
	curandInit << <blocknum * BLOCK_NUM, _THREAD_NUM >> > (devStates, 0);

	para_init << <T, INDIVNUM >> > (devStates, M, b, task_choice);
}


/****************************************************************************** 
 * @description: For various types of variable exchange operations in GPU, mainly used for bitonic sort
 * @param {float**、float*、int*} a
 * @param {float**、float*、int*} b
 * @return { }
 *******************************************************************************/
__device__ void _swap(float** a, float** b) {
	float* temp = *a;
	*a = *b;
	*b = temp;
}
__device__ void _swap(float* a, float* b) {
	float temp = *a;
	*a = *b;
	*b = temp;
}
__device__ void _swap(int* a, int* b) {
	int temp = *a;
	*a = *b;
	*b = temp;
}


/****************************************************************************** 
 * @description: GPU based bitonic sort
 * @param {DS*} temp
 * @param {int} num
 * @return { }
 *******************************************************************************/
__device__ void bitonic_sort(DS* temp, int num) {
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int DS_idx = tid / BANKSIZE, inner_idx = tid % BANKSIZE;

	if (tid < num) {
		for (unsigned int i = 2; i <= num; i <<= 1) {
			for (unsigned int j = i >> 1; j > 0; j >>= 1) {
				unsigned int tid_comp = tid ^ j, comp_idx = tid_comp / BANKSIZE, compinner = tid_comp % BANKSIZE;

				if (tid_comp > tid) {
					if ((tid & i) == 0) {
						if (temp[DS_idx].eval[inner_idx] > temp[comp_idx].eval[compinner]) {
							//printf("%f, %f\n", temp[DS_idx].eval[inner_idx], *(temp[DS_idx].eval_pointer[inner_idx]));
							_swap(&temp[DS_idx].eval[inner_idx], &temp[comp_idx].eval[compinner]);
							_swap(&temp[DS_idx].pointer[inner_idx], &temp[comp_idx].pointer[compinner]);
							_swap(&temp[DS_idx].eval_pointer[inner_idx], &temp[comp_idx].eval_pointer[compinner]);

						}
					}
					else {
						if (temp[DS_idx].eval[inner_idx] < temp[comp_idx].eval[compinner]) {
							_swap(&temp[DS_idx].eval[inner_idx], &temp[comp_idx].eval[compinner]);
							_swap(&temp[DS_idx].pointer[inner_idx], &temp[comp_idx].pointer[compinner]);
							_swap(&temp[DS_idx].eval_pointer[inner_idx], &temp[comp_idx].eval_pointer[compinner]);
						}
					}
				}
				__syncthreads();
			}
		}
	}
}

/****************************************************************************** 
 * @description: population sort
 * @param {DS*} indiv_sort
 * @param {float*} indiv_val
 * @param {float*} INDIVIDUALS
 * @param {int*} type
 * @return { }
 *******************************************************************************/
__global__ void popSort(DS* indiv_sort, float* indiv_val, float* INDIVIDUALS, int* type) {
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	int DS_size = (INDIVNUM - 1) / BANKSIZE + 1;
	int t_n = blockDim.x * gridDim.x;

	__shared__ DS temp[(INDIVNUM - 1) / BANKSIZE + 1];
	while (tid < INDIVNUM) {
		temp[tid / BANKSIZE].eval[tid % BANKSIZE] = indiv_val[tid];
		temp[tid / BANKSIZE].pointer[tid % BANKSIZE] = &INDIVIDUALS[tid * DIMENSION];
		temp[tid / BANKSIZE].eval_pointer[tid % BANKSIZE] = &indiv_val[tid];
		tid += t_n;
	}
	__syncthreads();

	bitonic_sort(temp, INDIVNUM);

	tid = threadIdx.x + (blockIdx.x * blockDim.x);
	while (tid < DS_size) {
		indiv_sort[tid] = temp[tid];
		tid += t_n;
	}
}


__device__ void getSum(int new_dim, float* temp_middle, int indivs_num, int indiv_perblock = INDIV_PERBLOCK) {
	const int t_n = THREAD_FOR_OPERA;
	while (true) {
		int tid = threadIdx.x;
		int old_dim = new_dim;
		new_dim >>= 1;
		__syncthreads();
		if (new_dim == 0) {
			break;
		}
		while (tid < new_dim * indivs_num) {
			int cur_dim = tid / indivs_num;
			float temp_val = temp_middle[(cur_dim + new_dim) * indiv_perblock + tid % indivs_num];
			if (old_dim % 2 == 1 && cur_dim + new_dim == old_dim - 2) {
				temp_val += temp_middle[(old_dim - 1) * indiv_perblock + tid % indivs_num];
			}
			temp_middle[cur_dim * indiv_perblock + tid % indivs_num] += temp_val;
			tid += t_n;
		}
	}
}

__device__ void getMulti(int new_dim, float* temp_middle, int indivs_num, int indiv_perblock = INDIV_PERBLOCK) {
	const int t_n = THREAD_FOR_OPERA;
	while (true) {
		int tid = threadIdx.x;
		int old_dim = new_dim;
		new_dim >>= 1;
		__syncthreads();
		if (new_dim == 0) {
			break;
		}
		while (tid < new_dim * indivs_num) {
			int cur_dim = tid / indivs_num;
			float temp_val = temp_middle[(cur_dim + new_dim) * indiv_perblock + tid % indivs_num];
			if (old_dim % 2 == 1 && cur_dim + new_dim == old_dim - 2) {
				temp_val *= temp_middle[(old_dim - 1) * indiv_perblock + tid % indivs_num];
			}
			temp_middle[cur_dim * indiv_perblock + tid % indivs_num] *= temp_val;
			tid += t_n;
		}
	}

}

/****************************************************************************** 
 * @description: atomic operation
 * @param {float*} address
 * @param {float} val
 * @return { }
 *******************************************************************************/
__device__ float atomicMul(float* address, float val) 
{ 
  unsigned int* address_as_ull = (unsigned int*)address; 
  unsigned int old = *address_as_ull, assumed; 
  do { 
 assumed = old; 
 old = atomicCAS(address_as_ull, assumed, __float_as_int(val * __int_as_float(assumed))); 
 } while (assumed != old); return __int_as_float(old);
} 


/****************************************************************************** 
 * @description: Mutation operation launched by GeneticOpera module, polynomial mutation is used here
 * @param {int} indiv_num
 * @param {int} n
 * @param {curandState*} state
 * @param {float*} indivs_in_s
 * @param {float*} rand
 * @return { }
 *******************************************************************************/
__device__ void Mutation(int indiv_num, int n, curandState* state,  float* indivs_in_s, float* rand) {
	int tid = threadIdx.x;
	int t_n = THREAD_FOR_OPERA;
	int indiv_perblock = INDIV_PERBLOCK;
	int group_interval = indiv_perblock * DIMENSION;

	float range = rand[1] - rand[0];
    float P = 1.f;

	tid = threadIdx.x;
	
	while (tid < indiv_num * DIMENSION) {
		
		float r = curand_uniform(state);
		
        if (r < P) {
            
			float u = curand_uniform(state);
			int idx = tid % indiv_num;
			int indiv_idx = tid / indiv_num * indiv_perblock + idx;
			float delta = 1.f, inner_val_1, inner_val_2, delta_;
			float temp_indiv_1 = indivs_in_s[group_interval * 2 + indiv_idx];
			delta_ = 1.0f * (temp_indiv_1 - rand[0]) / range;
			inner_val_1 = 2.f * u;
			inner_val_2 = 1.f - 2.f * u;
			if (u > 0.5) {
				delta = -1.f;
				delta_ = 1.f * (rand[1] - temp_indiv_1) / range;
				inner_val_1 = 2.f * (1.f - u);
				inner_val_2 = 2.f * (u - 0.5f);
			}
			delta *= powf(inner_val_1 + inner_val_2 * powf((1 - delta_), DM + 1.f), 1.f / (DM + 1.f)) - 1;
			temp_indiv_1 = temp_indiv_1 + delta * range;
			if (temp_indiv_1 > rand[1]) {
				temp_indiv_1 = rand[1];
			}
			else if (temp_indiv_1 < rand[0]) {
				temp_indiv_1 = rand[0];
			}
			indivs_in_s[group_interval * 2 + indiv_idx] = temp_indiv_1;
            
		}
        
		tid += THREAD_FOR_OPERA;
	}
}

/**
 * @name: CrossOver
 * @description:
 		Simulated Binary Crossover
 * @param: 
		indiv_num: individuals number for crossover
		state: curandState*
		indivs_in_s: Individuals in a block, in units of all individuals in each dimension:
			The arrangement is: indiv_00, indiv_10, ...,indiv_01, indiv_11, ...；
			The size is: indiv_num * DIMENSION
		rand: Dimension upper and lower bounds, rand[0] is the lower bound and rand[1] is the upper bound
 * @return 
 */
__device__ void CrossOver(int indiv_num, int n, curandState* state,  float* indivs_in_s, float* rand, float* r, float* indivs) {
	
	//===============================================================================
	//Basic setting
	int tid = threadIdx.x;//Thread ID number, each thread is responsible for the calculation of some dimensions of an individual
	int t_n = THREAD_FOR_OPERA;//total threads number for this device function
	int indiv_perblock = INDIV_PERBLOCK;//individual number per block
	int group_interval = indiv_perblock * DIMENSION;//Array size of indivs_in_s
	
	//===============================================================================
	
	while(tid < indiv_num * DIMENSION){
		int idx = tid % indiv_num;
		int r0 = r[idx], r1 = r[idx + 1];
		int init_posi = (tid / indiv_num) * indiv_num;
		float temp_indiv_1;
		float temp_indiv_2 = indivs_in_s[group_interval * (1 - n) + init_posi + r0];
		float rate = curand_uniform(state);
		if(rate >= 0){
			float p = curand_uniform(state);
			float temp_val = 0.5f / (1.f - p);
			if (p <= 0.5) {
				temp_val = 2 * p;
			}
			float beta = powf(temp_val, 1.0f / (ETA + 1.f));
			if ((tid % indiv_num) % 2 == 1) {
				beta *= -1;
				r1 = r[idx - 1];
			}
			temp_indiv_1 =
				0.5 * ((1 + beta) * temp_indiv_2
					+ (1 - beta) * indivs_in_s[group_interval * (1 - n) + init_posi + r1]);
			if (temp_indiv_1 > rand[1]) {
				temp_indiv_1 = rand[1];
			}
			else if (temp_indiv_1 < rand[0]) {
				temp_indiv_1 = rand[0];
			}
		}
		else{
			temp_indiv_1 = temp_indiv_2;
		}
		indivs_in_s[group_interval * 2 + tid] = temp_indiv_1;
		

		
		tid += THREAD_FOR_OPERA;
	}
}

/****************************************************************************** 
 * @description: Selection operation, here adopts the form of parent-child individual comparison
 * @param {int} indiv_num
 * @param {int} n
 * @param {float*} indivs_eval
 * @param {float*} indivs_in_s
 * @return { }
 *******************************************************************************/
__device__ void Selection(int indiv_num, int n,  float* indivs_eval,  float* indivs_in_s) {
	__shared__ int temp_judge[INDIV_PERBLOCK];
	int tid = threadIdx.x;
	int t_n = THREAD_FOR_OPERA;
	int indiv_perblock = INDIV_PERBLOCK;
	int group_interval = indiv_perblock * DIMENSION;

	while (tid < indiv_num) {
		if (indivs_eval[tid + indiv_perblock * (1 - n)] >= indivs_eval[indiv_perblock * 2 + tid]) {
			indivs_eval[tid + indiv_perblock * (1 - n)] = indivs_eval[indiv_perblock * 2 + tid];
			temp_judge[tid] = 1;
		}
		else {
			temp_judge[tid] = 0;
		}
		tid += THREAD_FOR_OPERA;
	}
	__syncthreads();
	tid = threadIdx.x;
	while (tid < indiv_num * DIMENSION) {
		if (temp_judge[tid % indiv_num] == 1) {
			int indiv_idx = tid / indiv_num * indiv_perblock + tid % indiv_num;
			indivs_in_s[group_interval * (1 - n) + indiv_idx] = indivs_in_s[group_interval * 2 + indiv_idx];
		}
		tid += THREAD_FOR_OPERA;
	}
}

/****************************************************************************** 
 * @description: Preparation before crossover
 * @param {float*} r
 * @param {int} indiv_num
 * @param {int} t_n
 * @param {int} n
 * @param {curandState*} state
 * @param {int} init_thread
 * @return { }
 *******************************************************************************/
__device__ void CrossPrep(float* r, int indiv_num, int t_n, int n, curandState* state, int init_thread) {
	int indiv_perblock = INDIV_PERBLOCK;
	int tid = threadIdx.x - init_thread;
	int init_posi = (1 - n) * indiv_perblock * LOOPTIME;
	/*
	while (tid < LOOPTIME * indiv_num) {
		int r_idx = (tid / indiv_num) * indiv_perblock + tid % indiv_num;
		r[init_posi + r_idx] = tid % indiv_num;
		tid += t_n;
	}
	__syncthreads();
	tid = threadIdx.x - init_thread;
	int half_choose = indiv_num >> 1;
	while (tid < half_choose * LOOPTIME) {
		int r1 = curand(state) % indiv_num;
		int r2 = curand(state) % indiv_num;
		while (r2 == r1) {
			r2 = curand(state) % indiv_num;
		}
		float rand_u = curand_uniform(state);
		if(rand_u > 0){
			r[init_posi + (tid / half_choose) * indiv_perblock + r2] = r1;// + beta;
		}
		rand_u = curand_uniform(state);
		if(rand_u > 0){
			r[init_posi + (tid / half_choose) * indiv_perblock + r1] = r2;// +(1 - beta);
		}
		tid += t_n;
	}
	*/
	
	tid = threadIdx.x - init_thread;
	int half_choose = indiv_num;
	while (tid < half_choose * LOOPTIME) {////Randomly select parents, allowing coverage (here the crossover probability is 1)
		int r1 = curand(state) % indiv_num;
			r[init_posi + (tid / half_choose) * indiv_perblock + tid] = r1;
		tid += t_n;
	}
	__syncthreads();
	/*
	while (tid < LOOPTIME * indiv_num) {
		int r_idx = init_posi + (tid / indiv_num) * indiv_perblock + tid % indiv_num;
		if ((int)r[r_idx] != tid % indiv_num) {
			float r1 = r[r_idx] - (int)r[r_idx];
			beta[r_idx] = powf(0.5f / (1.f - r1), 1.0f / (ETA + 1.f));
			if (r1 <= 0.5) {
				beta[r_idx] = powf((2 * r1), 1.f / (ETA + 1.f));
			}
			if (r[r_idx] < tid % indiv_num) {
				beta[r_idx] *= -1;
			}
		}
		tid += t_n;
	}
	__syncthreads();
	*/
}

/****************************************************************************** 
 * @description: function for test
 * @param {float*} indivs_in_s
 * @param {float*} indiv_val
 * @param {int} init_thread
 * @param {int} n
 * @param {int} c
 * @return { }
 *******************************************************************************/
__device__ void indivval_test( float* indivs_in_s,  float* indiv_val, int init_thread, int n, int c){
	int tid = threadIdx.x - init_thread;
	if(tid < INDIV_PERBLOCK){
					if(fabs(indiv_val[INDIV_PERBLOCK * n + tid] - indiv_val[INDIV_PERBLOCK * 2 + tid]) >= 0.1){
						//printf("sth wrong on eval..\n");
					}
				float result = 0;
				float result_1 = 0;
				for(int i = 0; i < DIMENSION; ++i){
					if(fabs(indivs_in_s[INDIV_PERBLOCK * DIMENSION * n + i * INDIV_PERBLOCK + tid] - indivs_in_s[INDIV_PERBLOCK * DIMENSION * 2 + i * INDIV_PERBLOCK + tid]) >= 1){
						//printf("indiv wrong..\n");
					}
					float x1 = indivs_in_s[INDIV_PERBLOCK * DIMENSION * n + i * INDIV_PERBLOCK + tid];
					result += x1 * x1;
					
					result_1 += cosf(2 * 3.1415926 * x1);
				}
				result = -20 * expf(-0.2 * sqrtf(result / 50)) - expf(result_1 / 50) + 20 + 2.718282;
				if(fabs( indiv_val[INDIV_PERBLOCK * n + tid]- result) >= 0.1){
					printf("!!sth wrong..:%f,%f, %d, %d\n", result, indiv_val[tid], tid, c);
				}
	}
}


/****************************************************************************** 
 * @description: Data transfer operations, transfer from shared memory back to global memory
 * @param {float*} indivs
 * @param {float*} indivs_in_s
 * @param {float*} indivs_eval
 * @param {float*} eval
 * @param {curandState*} state
 * @param {int*} shuffle
 * @param {int} n
 * @param {int} thread_num
 * @return { }
 *******************************************************************************/
__device__ void SharedToGlobal(float* indivs,  float* indivs_in_s,  float* indivs_eval, float* eval, curandState* state, int* shuffle, int n, int thread_num) {
	int threads_for_opera = _THREAD_NUM - thread_num;
	int tid = threadIdx.x - threads_for_opera, t_n = thread_num;
	int indiv_perblock = INDIV_PERBLOCK, group_interval = indiv_perblock * DIMENSION;
	int init_posi = blockIdx.x * (group_interval * 2);

	while (tid < indiv_perblock) {
		shuffle[tid] = tid;
		tid += t_n;
	}
	
	__syncthreads();
	tid = threadIdx.x - threads_for_opera;
	
	if (tid == 0) {
		
		for (int k = indiv_perblock - 1; k >= 0; --k) {
			int target = curand(state) % (k + 1);
			int temp = shuffle[k];
			shuffle[k] = shuffle[target];
			shuffle[target] = temp;
		}
	}
	
	tid = threadIdx.x - threads_for_opera;
	__syncthreads();
	while (tid < indiv_perblock * DIMENSION) {
		indivs[init_posi + indiv_perblock * DIMENSION * n + shuffle[tid / DIMENSION] * DIMENSION + tid % DIMENSION] = indivs_in_s[group_interval * n + (tid % DIMENSION) * indiv_perblock + (tid / DIMENSION)];
		tid += t_n;
	}

	tid = threadIdx.x - threads_for_opera;
	while (tid < indiv_perblock) {
		eval[blockIdx.x * indiv_perblock * 2 + shuffle[tid] + indiv_perblock * n] = indivs_eval[indiv_perblock * n + tid];
		tid += t_n;
	}

}


/****************************************************************************** 
 * @description: fitness evaluation
 * @param {int} type
 * @param {float*} M_s
 * @param {float*} b_s
 * @param {float*} indivs_in_s
 * @param {int} indivs_num
 * @param {float*} eval
 * @param {int} indiv_perblock
 * @return { }
 *******************************************************************************/
__device__ void evaluation(int type, float* M_s, float* b_s,  float* indivs_in_s, int indivs_num,  float* eval, int indiv_perblock) {//函数评估
	int tid = threadIdx.x;
	int marray_size = (WARPSIZE / indiv_perblock + 1) * indiv_perblock;
	int offset = WARPSIZE % indivs_num;
	int remainder = (WARPSIZE / indivs_num + 1) * indivs_num;
	int step_size = WARPSIZE;
	if (WARPSIZE < indivs_num) {
		offset = 0;
		step_size = indivs_num;
	}
	extern __shared__ float temp_try[];
	tid = THREAD_FOR_OPERA - threadIdx.x - 1;
	while (tid < 3 * marray_size) {
		temp_try[tid] = 0;
		tid += THREAD_FOR_OPERA;
	}
	__syncthreads();
	
    switch (type) {
	case 0://Shpere
		tid = threadIdx.x;
		while (tid < DIMENSION * indivs_num) {
			int dimension = tid / indivs_num;
			float x1 = M_s[dimension] * (indivs_in_s[dimension * indiv_perblock + tid % indivs_num] + b_s[dimension]);
			float result = x1 * x1;
			atomicAdd(&temp_try[(offset * (tid / step_size) + (tid % step_size)) % remainder], result);
			tid += THREAD_FOR_OPERA;
		}
		__syncthreads();
		if (threadIdx.x < indivs_num) {
			tid = threadIdx.x;
			float temp_val = 0;
			while (tid < remainder) {
				temp_val += temp_try[tid];
				tid += indivs_num;
			}
			eval[threadIdx.x] = temp_val;
		}
		
		break;
	case 1://Rosenbrock
		  
		tid = threadIdx.x;
		while (tid < (DIMENSION - 1) * indivs_num) {
			int dimension = tid / indivs_num;
			float x_1 = M_s[dimension] * (indivs_in_s[dimension * indiv_perblock + tid % indivs_num] + b_s[dimension]);
			float x_2 = M_s[dimension + 1] * (indivs_in_s[(dimension + 1) * indiv_perblock + tid % indivs_num] + b_s[dimension + 1]);
			float middle_val1 = x_1 * x_1 - x_2, middle_val2 = x_1 - 1;
			float result = 100 * middle_val1 * middle_val1 + middle_val2 * middle_val2;
			atomicAdd(&temp_try[(offset * (tid / step_size) + (tid % step_size)) % remainder], result); 
			tid += THREAD_FOR_OPERA;
		}
		__syncthreads();
		if (threadIdx.x < indivs_num) {
			tid = threadIdx.x;
			float temp_val = 0;
			while (tid + indivs_num < remainder) {
				temp_val += temp_try[tid + indivs_num];
				tid += indivs_num;
			}
			eval[threadIdx.x] = temp_try[threadIdx.x] + temp_val;
		}
		
		break;
	case 2://Ackley
		tid = threadIdx.x;

		   float temp_0 = 0, temp_1 = 0;
		   while (tid < DIMENSION * indivs_num) {
			   int dimen = tid / indivs_num;
			   float x = M_s[dimen] * (indivs_in_s[dimen * indiv_perblock + tid % indivs_num] + b_s[dimen]);
			   int bank_idx = (offset * (tid / step_size) + (tid % step_size)) % remainder;
			   atomicAdd(&temp_try[bank_idx + marray_size], x * x);
			   atomicAdd(&temp_try[bank_idx + marray_size * 2], cospif(2 * x));
			   tid += THREAD_FOR_OPERA;
		   }
		   __syncthreads();
			if (threadIdx.x < indivs_num) {
				tid = threadIdx.x;
				float temp_val = 0;
				float temp_val_1 = 0;
				while (tid < remainder) {
					temp_val += temp_try[tid + marray_size];
					temp_val_1 += temp_try[tid + 2 * marray_size];
					tid += indivs_num;
				}
				float result = -20 * expf(-0.2 * sqrtf(temp_val / DIMENSION)) - expf(temp_val_1 / DIMENSION) + 20 + E;
				eval[threadIdx.x] = result;
		   }
		
		break;
	case 3://Rastrgin
		
		tid = threadIdx.x;
		while (tid < DIMENSION * indivs_num) {
			int dimension = tid / indivs_num;
			float x = M_s[dimension] * (indivs_in_s[dimension * indiv_perblock + tid % indivs_num] + b_s[dimension]);
			float result = x * x - 10 * cospif(2 * x) + 10;

			atomicAdd(&temp_try[(offset * (tid / step_size) + (tid % step_size)) % remainder], result);
			tid += THREAD_FOR_OPERA;
		}
		__syncthreads();
		if (threadIdx.x < indivs_num) {
			tid = threadIdx.x;
			float temp_val = 0;
			while (tid + indivs_num < remainder) {
				temp_val += temp_try[tid + indivs_num];
				tid += indivs_num;
			}
			eval[threadIdx.x] = temp_try[threadIdx.x] + temp_val;
		}
		
		break;
	case 4://Griewank
		
		
		tid = threadIdx.x;
		while (tid < marray_size) {
			temp_try[tid + marray_size * 2] = 1;
			tid += THREAD_FOR_OPERA;
		}
		__syncthreads();
		tid = threadIdx.x;
		while (tid < DIMENSION * indivs_num) {
			int dimen = tid / indivs_num;
			float x = M_s[dimen] * (indivs_in_s[dimen * indiv_perblock + tid % indivs_num] + b_s[dimen]);
			int bank_idx = (offset * (tid / step_size) + (tid % step_size)) % remainder;
			atomicAdd(&temp_try[bank_idx + marray_size], x * x);
			float old_val = atomicMul(&temp_try[bank_idx + marray_size * 2], cosf(x / sqrtf(tid / indivs_num + 1)));
			//printf("%f = %f mul %f\n", temp_try[bank_idx + marray_size * 2], old_val, cosf(x / sqrtf(tid / indivs_num + 1)));
			tid += THREAD_FOR_OPERA;
		}
		__syncthreads();
		if (threadIdx.x < indivs_num) {
			tid = threadIdx.x;
			float temp_val_1 = 0, temp_val_2 = 1;
			while (tid < remainder) {
				temp_val_1 += temp_try[tid + marray_size];
				temp_val_2 *= temp_try[marray_size * 2 + tid];
				tid += indivs_num;
			}
			eval[threadIdx.x] = 1 + temp_val_1 / 4000 - temp_val_2;
			
		}

		
		break;

	case 5://Weierstrass
		
		tid = threadIdx.x;
		while (tid < DIMENSION * indivs_num) {
			int dimension = tid / indivs_num;
			float x = M_s[dimension] * (indivs_in_s[dimension * indiv_perblock + tid % indivs_num] + b_s[dimension]);
			float temp_val = 2 * (x + 0.5);
			float result = 0;
			float pow_val1 = 1, pow_val2 = 1;

			for (int k = 0; k <= 20; ++k) {
				result += pow_val1 * cospif(temp_val * pow_val2);
				pow_val1 *= 0.5;
				pow_val2 *= 3;
			}
			atomicAdd(&temp_try[(offset * (tid / step_size) + (tid % step_size)) % remainder], result);
			tid += THREAD_FOR_OPERA;
		}
		__syncthreads();
		if (threadIdx.x < indivs_num) {
			tid = threadIdx.x;
			float temp_val = 0;
			while (tid < remainder) {
				temp_val += temp_try[tid];
				tid += indivs_num;
			}
			eval[threadIdx.x] = temp_val - DIMENSION * Weierstrass_para;
		}
		
		break;
	case 6://Schwefel
		
		tid = threadIdx.x;
		while (tid < DIMENSION * indivs_num) {
			int dimension = tid / indivs_num;
			float x = M_s[dimension] * (indivs_in_s[dimension * indiv_perblock + tid % indivs_num] + b_s[dimension]);
			float result = x * sinf(sqrtf(fabsf(x)));
			atomicAdd(&temp_try[(offset * (tid / step_size) + (tid % step_size)) % remainder], result);
			tid += THREAD_FOR_OPERA;
		}
		__syncthreads();
		if (threadIdx.x < indivs_num) {
			tid = threadIdx.x;
			float temp_val = 0;
			while (tid < remainder) {
				temp_val += temp_try[tid];
				tid += indivs_num;
			}

			eval[threadIdx.x] = 418.9829f * DIMENSION - temp_val;eval[threadIdx.x] = 418.9829f * DIMENSION - temp_val;
			
		}
		
		break;
	}
}

/****************************************************************************** 
 * @description: Parity exchange sorting, mainly used for island migration on GeneticOpera module
 * @param {float*} indivs_val
 * @param {int*} idx
 * @param {int} tid
 * @param {int} num
 * @return { }
 *******************************************************************************/
__device__ void paritySort(float* indivs_val, int* idx, int tid, int num) {
    int max_num = ((num / 2) / WARPSIZE + 1) * WARPSIZE;
	if (tid < (max_num)) {
		if (tid % 2 == 1) {
			tid += num / 2;
			tid += tid % 2;
		}
		for (int i = 0; i < num; i++) {
			int temp_id = tid + (i % 2);
			if (temp_id + 1 < num) 
            {
                if(indivs_val[idx[temp_id]] > indivs_val[idx[temp_id + 1]]){
			        _swap(&idx[temp_id], &idx[temp_id + 1]);
                }
			}
			if(max_num > WARPSIZE){
				__syncthreads();
			}
		}
	}
}

/****************************************************************************** 
 * @description: island migration
 * @param {int*} idx_forSort
 * @param {float*} indivs_eval
 * @param {float*} indivs_in_s
 * @param {int} indiv_num
 * @return { }
 *******************************************************************************/
__device__ void IslandMigration(int* idx_forSort, float* indivs_eval, float* indivs_in_s, int indiv_num) {
	int t_n = _THREAD_NUM;
	//===================================sort======================================
	int tid = threadIdx.x;
	int indiv_perblock = INDIV_PERBLOCK;
	int arraySize = indiv_perblock;
	while (tid < 2 * arraySize) {
		idx_forSort[tid] = tid % arraySize;
		tid += t_n;
	}
	__syncthreads();


	tid = threadIdx.x;
    int max_num = ((indiv_num / 2) / WARPSIZE + 1) * WARPSIZE;
	while (tid < max_num) {
		paritySort(indivs_eval, idx_forSort, tid, indiv_num);
	
		tid += t_n;
	}
	tid = threadIdx.x - max_num;
	while (tid < max_num && tid >= 0) {
		paritySort(indivs_eval + indiv_perblock, idx_forSort + arraySize, tid, indiv_num);
		tid += t_n;
	}

	__syncthreads();
	
	//===================================isoland migration======================================
	tid = threadIdx.x;
	__shared__ int temp[MIGRA_NUM * 2];
	int migra_num = indiv_num * MIGRA_PROP;
	if (indiv_num < indiv_num) {
		migra_num = indiv_num * MIGRA_PROP;
	}

	while (tid < migra_num) {
		int selfId = indiv_num + idx_forSort[tid + arraySize + indiv_num - migra_num];
		int targetId = idx_forSort[tid];
		if (indivs_eval[targetId] < indivs_eval[selfId]) {
			indivs_eval[selfId] = indivs_eval[targetId];
			temp[tid] = 1;
		}
		else {
			temp[tid] = 0;
		}
		tid += t_n;
	}
	tid = threadIdx.x - migra_num;
	while (tid < migra_num) {
		int targetId = indiv_num + idx_forSort[arraySize + tid];
		int selfId = idx_forSort[tid + indiv_num - migra_num];
		if (indivs_eval[targetId] < indivs_eval[selfId]) {
			indivs_eval[selfId] = indivs_eval[targetId];
			temp[migra_num + tid] = 1;
		}
		else {
			temp[migra_num + tid] = 0;
		}
		tid += t_n;
	}

	__syncthreads();
	tid = threadIdx.x;
	while (tid < migra_num * DIMENSION) {
		int init_dimen = (tid / migra_num) * indiv_perblock;
		if (temp[tid % migra_num] == 1) {
			indivs_in_s[indiv_perblock * DIMENSION + init_dimen + idx_forSort[arraySize + indiv_num - tid % migra_num - 1]] = indivs_in_s[init_dimen + idx_forSort[tid % migra_num]];
		}
		if (temp[migra_num + tid % migra_num] == 1) {
			indivs_in_s[init_dimen + idx_forSort[indiv_num - tid % MIGRA_NUM - 1]] = indivs_in_s[indiv_perblock * DIMENSION + init_dimen + idx_forSort[arraySize + tid % migra_num]];
		}
		tid += t_n;
	}
	
}

/****************************************************************************** 
 * @description:  Data transfer operations, transfer from global memory to shared memory
 * @param {int} n
 * @param {float*} indivs_in_s
 * @param {float*} indivs_eval
 * @param {float*} indivs
 * @param {float*} eval
 * @param {int*} select_interval
 * @param {int} iter
 * @return { }
 *******************************************************************************/
__device__ void SelectFromBlocks(int n,  float* indivs_in_s,  float* indivs_eval, float* indivs, float* eval, int* select_interval, int iter) {
	int tid = threadIdx.x - THREAD_FOR_OPERA;
	int indiv_perblock = INDIV_PERBLOCK, group_interval = indiv_perblock * DIMENSION;
	__shared__ int targetblock[INDIV_PERBLOCK];

	int temp_k = 0;
	while (tid < indiv_perblock) {
		targetblock[tid] = (blockIdx.x + select_interval[iter * indiv_perblock + tid]) % gridDim.x;
		
		indivs_eval[indiv_perblock * n + tid] = eval[indiv_perblock * targetblock[tid] * 2 + indiv_perblock * n + tid];
		tid += THREAD_FOR_TRANSFER;
	}

	__syncthreads();
	tid = threadIdx.x - THREAD_FOR_OPERA;
	while (tid < indiv_perblock * DIMENSION) {
		int indiv_id = tid / DIMENSION;
		indivs_in_s[group_interval * n + (tid % DIMENSION) * indiv_perblock + indiv_id] = indivs[group_interval * targetblock[indiv_id] * 2 + indiv_perblock * DIMENSION * n + tid];
		tid += THREAD_FOR_TRANSFER;
	}

}

/****************************************************************************** 
 * @description: function for test
 * @param {float*} INDIVIDUALS
 * @param {float*} indivs_val
 * @return { }
 *******************************************************************************/
__device__ void eval_test(float* INDIVIDUALS, float* indivs_val){
	
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid - THREAD_FOR_OPERA == 0){
	bool f = false;
	for (int i = 0; i < 50; ++i) {
		float result = 0;418.9829 * DIMENSION;
		float result_1 = 0;
		for (int j = 0; j < DIMENSION; ++j) {
			//result -= INDIVIDUALS_CPU[task_debug][i * DIMENSION + j] * sinf(sqrtf(fabsf(INDIVIDUALS_CPU[task_debug][i * DIMENSION + j])));
			result += INDIVIDUALS[i * DIMENSION + j] * INDIVIDUALS[i * DIMENSION + j];
			result_1 += cosf(2 * 3.1415926 * INDIVIDUALS[i * DIMENSION + j]);
		}
		//printf("num:%d, eval: %f\n\n", i, indivs_val[i]);
		result = -20 * expf(-0.2 * sqrtf(result / 50)) - expf(result_1 / 50) + 20 + 2.718282;
		if(fabs(result - indivs_val[i]) > 0.1){
			f = true;
			printf("result:%f, %f, %d\n", result, indivs_val[i], i);
			//break;
		}
	}
		printf("-------------------------------------------\n");
	}
}
 
/****************************************************************************** 
 * @description: Kernel(O) module
 * @param {float*} indivs, the entire population in global memory
 * @param {volatile int*} syncval, atomic variable for block synchronization
 * @param {curandState*} states
 * @param {int*} type, task type
 * @param {float*} M
 * @param {float*} b
 * @param {float*} eval
 * @param {int*} select_interval
 * @return { }
 *******************************************************************************/ 
__global__ void GeneticOpera(float* indivs, volatile int* syncval, curandState* states, int* type, float* M, float* b, float* eval, int* select_interval) {
	
	
	 __shared__ float indivs_in_s[INDIV_PERBLOCK * 3 * DIMENSION];
	 __shared__ float indivs_eval[INDIV_PERBLOCK * 3];
	__shared__ int idx_forSort[INDIV_PERBLOCK * 2];
	__shared__ int shuffle[INDIV_PERBLOCK];
	__shared__ float M_s[DIMENSION], b_s[DIMENSION];

	float range[2];
	range[0] = Range[*type * 2];
	range[1] = Range[*type * 2 + 1];

	curandState state = states[blockDim.x * blockIdx.x + threadIdx.x];
	int tid = threadIdx.x;
	int indiv_perblock = INDIV_PERBLOCK;
	int group_interval = indiv_perblock * DIMENSION;
	int init_posi = blockIdx.x * (group_interval * 2);
	int indiv_num = indiv_perblock;

	while (tid < indiv_num * DIMENSION) {
		
		indivs_in_s[(tid % DIMENSION) * indiv_perblock + (tid / DIMENSION)] = indivs[init_posi + tid];
		tid += _THREAD_NUM;
	}
	tid = threadIdx.x;
	while (tid < indiv_num) {
		indivs_eval[tid] = eval[blockIdx.x * indiv_perblock * 2 + tid];
		tid += _THREAD_NUM;
	}
	tid = _THREAD_NUM - threadIdx.x - 1;
	while (tid < DIMENSION) {
		M_s[tid] = M[tid];
		b_s[tid] = b[tid];
		tid += _THREAD_NUM;
	}
	int iter_time = ((INTERVAL_TRANSFER - 1) / LOOPTIME + 1) * 2;
	int n = 1;

	__shared__ float r[LOOPTIME * INDIV_PERBLOCK * 2];
	
	CrossPrep(r, indiv_num, _THREAD_NUM, n, &state, 0);

	int interval_migra = LOOPTIME;

	int migra_time = (INTERVAL_MIGRA / LOOPTIME) * 2;
	for (int i = 0; i < iter_time; ++i) {
		if (iter_time - i == 2) {
			interval_migra = (INTERVAL_TRANSFER * 2 - LOOPTIME * i) / (2);
		}
		//DataProcess and DataTransfer are performed in parallel
			if (threadIdx.x < THREAD_FOR_OPERA) {
				for (int j = 0; j < interval_migra; ++j) {
					//mutation
					tid = threadIdx.x;
					__syncthreads();
					
					//indivval_test(indivs_in_s, indivs_eval, 0, 1 - n, i);
					//crossover
					CrossOver(indiv_num, n, &state, indivs_in_s, range, r + (1 - n) * indiv_perblock * LOOPTIME, indivs);
					__syncthreads();
					tid = threadIdx.x;

					Mutation(indiv_num, n, &state, indivs_in_s, range);
                    
					__syncthreads();
					//filter + update
					evaluation(*type, M_s, b_s, indivs_in_s + 2 * group_interval, indiv_num, indivs_eval + indiv_perblock * 2, indiv_perblock);//1s左右时间
					
					__syncthreads();
					Selection(indiv_num, n, indivs_eval, indivs_in_s);
					
					
				}


			}
			else {

				
				tid = threadIdx.x - THREAD_FOR_OPERA;

				if (i > 0) {

					SharedToGlobal(indivs, indivs_in_s, indivs_eval, eval, &state, shuffle, n, THREAD_FOR_TRANSFER);
				}
				else{
					__syncthreads();
					__syncthreads();
				}



				tid = threadIdx.x - THREAD_FOR_OPERA;
				if (tid % THREAD_FOR_TRANSFER == 0) {//block synchronization
					atomicAdd((int*)(syncval + i), 1);
				}
					
				__syncthreads();
				CrossPrep(r, indiv_num, THREAD_FOR_TRANSFER, 1 - n, &state, THREAD_FOR_OPERA);
                while (*(syncval + i) % gridDim.x != 0) {
                }
                __syncthreads();
				SelectFromBlocks(n, indivs_in_s, indivs_eval, indivs, eval, select_interval, i);
				
				


				__syncthreads();
					
               if(*type == 4){
				__syncthreads();
			   }
				
			}
			
          
		
		
			n = (n + 1) % 2;
		__threadfence();
		if ((i + 1) % migra_time == 0 && i != iter_time - 1 && i % (INTERVAL_MIGRA * 2) == 0) {
			IslandMigration(idx_forSort, indivs_eval, indivs_in_s, indiv_num);	
		}
		
		__syncthreads();
		

	}
	SharedToGlobal(indivs, indivs_in_s, indivs_eval, eval, &state, shuffle, n, _THREAD_NUM);
	states[blockDim.x * blockIdx.x + threadIdx.x] = state;
			   
}

/****************************************************************************** 
 * @description: Independent Fitness evaluation
 * @param {float*} indivs_
 * @param {float*} eval
 * @param {float*} M
 * @param {float*} b
 * @param {int*} type
 * @return { }
 *******************************************************************************/
__global__ void evaluate(float* indivs_, float* eval, float* M, float* b, int* type) {

	int indiv_perblock = INDIV_PERBLOCK * 2, unit = INDIV_PERBLOCK * 2;
	int init_indiv = blockIdx.x * indiv_perblock;

	if (blockIdx.x == BLOCK_NUM - 1) {
		indiv_perblock = INDIVNUM - unit * (BLOCK_NUM - 1);
	}
	__shared__ float indivs[INDIV_PERBLOCK * 2 * DIMENSION];
	__shared__ float indivs_val[INDIV_PERBLOCK * 2];

	int tid = threadIdx.x;
	int t_n = blockDim.x;
	while (tid < indiv_perblock * DIMENSION) {
		indivs[(tid % DIMENSION) * unit + (tid / DIMENSION)] = indivs_[init_indiv * DIMENSION + tid];
		tid += t_n;
	}
	tid = threadIdx.x;
	while (tid < indiv_perblock) {
		indivs_val[tid] = eval[init_indiv + tid];
		tid += t_n;
	}
	tid = blockDim.x - threadIdx.x - 1;
	__shared__ float M_s[DIMENSION], b_s[DIMENSION];
	while (tid < DIMENSION) {
		M_s[tid] = M[tid];
		b_s[tid] = b[tid];
		tid += t_n;
	}
	__syncthreads();

	evaluation(*type, M_s, b_s, indivs, indiv_perblock, indivs_val, unit);


	__syncthreads();
	tid = threadIdx.x;
	while (tid < indiv_perblock * DIMENSION) {
		int target_tid = init_indiv * DIMENSION + tid;
		indivs_[target_tid] = indivs[(tid / DIMENSION) + (tid % DIMENSION) * unit];
		tid += t_n;
	}
	tid = threadIdx.x;
	while (tid < indiv_perblock) {
		eval[init_indiv + tid] = indivs_val[tid];
		//printf("eval:%f, type:%d\n", eval[init_indiv + tid], *type);
		tid += t_n;
	}
}

/****************************************************************************** 
 * @description: function for test
 * @param {float*} INDIVIDUALS
 * @param {int} cthread_idx
 * @param {int*} type
 * @param {DS*} indiv_sort
 * @param {float*} indivs_val
 * @return { }
 *******************************************************************************/
__global__ void test_func(float* INDIVIDUALS, int cthread_idx, int* type, DS* indiv_sort, float* indivs_val) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const int t_n = gridDim.x * blockDim.x;
	//while(tid < DIMENSION){
	//	printf("%f ", INDIVIDUALS[tid]);//*(indiv_sort[0].pointer[0] + tid));
	//	tid += t_n;
	//}
	tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid == 0){
		printf("%f ", *(indiv_sort[0].eval_pointer[0]));
	}
	while(tid < INDIVNUM){
		if(*(indiv_sort[0].eval_pointer[0]) > *(indiv_sort[tid / 32].eval_pointer[tid % 32])){
			printf("sth wrong on sort..\n");
		}
		tid += t_n;
	}
	/*
	while (tid < INDIVNUM) {
		printf("val:%f, %f, %f\n", *(indiv_sort[tid / BANKSIZE].eval_pointer[tid % BANKSIZE]), indiv_sort[tid / BANKSIZE].eval[tid % BANKSIZE], indivs_val[tid]);
		//if (*(indiv_sort[(tid / DIMENSION) / BANKSIZE].pointer[(tid / DIMENSION) % BANKSIZE] + tid % DIMENSION) < Range[type[0] * 2]) {
		//	printf("sth wrong....\n");
		//}

		//printf("INDIVS:%f, tid:%d\n", INDIVIDUALS[tid], cthread_idx);
		//if (INDIVIDUALS[tid] > 500) {
		//	printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!, %f\n", INDIVIDUALS[tid]);
		//}
		tid += t_n;
	}
	*/
}

/****************************************************************************** 
 * @description: knowledge transfer
 * @param {curandState*} state
 * @param {int} *type
 * @param {float*} M
 * @param {float*} b
 * @param {DS*} indivs_sort_1
 * @param {DS*} indivs_sort_2
 * @return { }
 *******************************************************************************/
__global__ void knowledgeTransfer(curandState* state, int *type, float* M, float* b, DS* indivs_sort_1, DS* indivs_sort_2) {

	__shared__ float indivs_self[int(TRANSFER_NUM / BLOCK_NUM) * DIMENSION];
	__shared__ float indivs_target[int(TRANSFER_NUM / BLOCK_NUM) * DIMENSION];

	__shared__ float indivs_eval_t[TRANSFER_NUM / BLOCK_NUM];


	int indiv_perblock = TRANSFER_NUM / BLOCK_NUM;
	float range_1 = Range[type[0] * 2 + 1] - Range[type[0] * 2];
	float range_2 = Range[type[1] * 2 + 1] - Range[type[1] * 2];
	int tid = threadIdx.x;
	int t_n = blockDim.x;
	int init_posi = blockIdx.x * indiv_perblock;
	while (tid < indiv_perblock * DIMENSION) {
		indivs_self[(tid % DIMENSION) * indiv_perblock + (tid / DIMENSION)] = *(indivs_sort_1[(init_posi + tid / DIMENSION) / BANKSIZE].pointer[(init_posi + tid / DIMENSION) % BANKSIZE] + tid % DIMENSION);
		indivs_target[(tid % DIMENSION) * indiv_perblock + (tid / DIMENSION)] = *(indivs_sort_2[(init_posi + tid / DIMENSION) / BANKSIZE].pointer[(init_posi + tid / DIMENSION) % BANKSIZE] + tid % DIMENSION);
		
		tid += t_n;
	}

	tid = threadIdx.x;
	__shared__ float M_s[DIMENSION], b_s[DIMENSION];
	while (tid < DIMENSION) {
		M_s[tid] = M[tid];
		b_s[tid] = b[tid];
		tid += t_n;
	}
	__syncthreads();
	tid = threadIdx.x;
	while (tid < indiv_perblock * DIMENSION) {
		float rand_val = curand_uniform(&state[threadIdx.x + blockDim.x * blockIdx.x]);

		if (rand_val > CROSSOVER_RATE) {
			indivs_self[tid] = Range[type[0] * 2] + (indivs_target[tid] - Range[type[1] * 2]) / range_2 * range_1;
		}
		tid += t_n;
	}
	__syncthreads();
	evaluation(type[0], M_s, b_s, indivs_self, indiv_perblock, indivs_eval_t, indiv_perblock);
	__syncthreads();
	tid = threadIdx.x;
	__shared__ int temp[TRANSFER_NUM / BLOCK_NUM];
	while (tid < indiv_perblock) {
		int target_tid = blockIdx.x * indiv_perblock + INDIVNUM - TRANSFER_NUM + tid;
		if (indivs_eval_t[tid] < *indivs_sort_1[target_tid / BANKSIZE].eval_pointer[target_tid % BANKSIZE]) {
			*indivs_sort_1[target_tid / BANKSIZE].eval_pointer[target_tid % BANKSIZE] = indivs_eval_t[tid];
			temp[tid] = 1;
		}
		else {
			temp[tid] = 0;
		}
		tid += t_n;
	}
	__syncthreads();
	
	tid = threadIdx.x;
	while (tid < indiv_perblock * DIMENSION) {
		init_posi = (blockIdx.x * indiv_perblock + INDIVNUM - TRANSFER_NUM);
		if (temp[tid / DIMENSION] == 1) {
			*(indivs_sort_1[(init_posi + tid / DIMENSION) / BANKSIZE].pointer[(init_posi + tid / DIMENSION) % BANKSIZE] + tid % DIMENSION) = indivs_self[(tid / DIMENSION) + (tid % DIMENSION) * indiv_perblock];
		}
		tid += t_n;
	}
	
}

/****************************************************************************** 
 * @description: Initial select_interval
 * @param {int*} select_interval
 * @param {curandState*} state
 * @return { }
 *******************************************************************************/
__global__ void intervalRand(int* select_interval, curandState* state){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int t_n = blockDim.x * gridDim.x;
	while(tid < SELECT_INTERVAL){
		select_interval[tid] = curand(&state[tid]) % BLOCK_NUM;
		tid += t_n;
	}
}

/****************************************************************************** 
 * @description: Transfer best individual back to CPU
 * @param {float*} indiv_best
 * @param {float*} indival_best
 * @param {DS*} indiv_sort
 * @return { }
 *******************************************************************************/
__global__ void messageTransfer(float* indiv_best, float* indival_best, DS* indiv_sort) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int t_n = gridDim.x * blockDim.x;
	while (tid < BESTNUM_PERTASK * DIMENSION) {
		int id = tid / DIMENSION;
		indiv_best[tid] = *(indiv_sort[id / BANKSIZE].pointer[id % BANKSIZE] + tid % DIMENSION);
		tid += t_n;
	}
	tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < BESTNUM_PERTASK) {
		indival_best[tid] = indiv_sort[tid / BANKSIZE].eval[tid % BANKSIZE];
		//printf("%f, %f\n", indiv_sort[tid / BANKSIZE].eval[tid % BANKSIZE], *(indiv_sort[tid / BANKSIZE].eval_pointer[tid % BANKSIZE]));
		tid += t_n;
	}
}

int last_transfer = transfer_num;
DS record[T][2][ITER_NUM / INTERVAL_TRANSFER + 1];

/****************************************************************************** 
 * @description: Iteration of each task, each CPU thread is responsible for a task
 * @param {int} task_idx
 * @param {int} cthread_idx
 * @param {cudaStream_t*} streams
 * @return { }
 *******************************************************************************/
void iter(int task_idx, int cthread_idx, cudaStream_t* streams) {
	mutex cond_mutex_w, cond_mutex_f;
	unique_lock<mutex> cond_lock_w(cond_mutex_w);
	unique_lock<mutex> cond_lock_f(cond_mutex_f);
	
	int blocknum = ISLAND_NUM * VAL_TYPE, threadnum = INDIVNUM_ISLAND * _THREAD_NUM * DIMENSION / SHARED_CAPACITY;

	pop_init << <BLOCK_NUM, _THREAD_NUM, 0, streams[cthread_idx % STREAM_NUM] >> > (devStates+ _THREAD_NUM * BLOCK_NUM * (cthread_idx % STREAM_NUM), INDIVIDUALS[cthread_idx], task_type[cthread_idx], indiv_val[cthread_idx]);
	int memory_cost = 3 * ((int)(WARPSIZE / (2 * INDIV_PERBLOCK) + 1)) * (2 * INDIV_PERBLOCK) * sizeof(float);
	evaluate<<<BLOCK_NUM, THREAD_FOR_OPERA, memory_cost, streams[cthread_idx % STREAM_NUM]>>>(INDIVIDUALS[cthread_idx], indiv_val[cthread_idx], M_cpu[task_idx], b_cpu[task_idx], task_type[cthread_idx]);
	
		
	popSort << <1, _THREAD_NUM, 0, streams[cthread_idx % STREAM_NUM] >> >(indiv_sort[cthread_idx], indiv_val[cthread_idx], INDIVIDUALS[cthread_idx], task_type[cthread_idx]);
	
	cudaStreamSynchronize(streams[cthread_idx % STREAM_NUM]);
	
	
	for (int i = 0; i < ITER_NUM / INTERVAL_TRANSFER; ++i) {
		intervalRand<<<3, _THREAD_NUM, 0, streams[cthread_idx % STREAM_NUM]>>>(select_interval[cthread_idx], devStates+ _THREAD_NUM * BLOCK_NUM * (cthread_idx % STREAM_NUM));
		
		GeneticOpera << <BLOCK_NUM, _THREAD_NUM, 3 * ((int)(WARPSIZE / INDIV_PERBLOCK + 1)) * INDIV_PERBLOCK * sizeof(float), streams[cthread_idx % STREAM_NUM]  >> > (INDIVIDUALS[cthread_idx], (volatile int*)syncval[cthread_idx], devStates+ _THREAD_NUM * BLOCK_NUM * (cthread_idx % STREAM_NUM), task_type[cthread_idx], M_cpu[task_idx], b_cpu[task_idx], indiv_val[cthread_idx], select_interval[cthread_idx]);
		
		popSort << <1, _THREAD_NUM, 0, streams[cthread_idx % STREAM_NUM] >> >(indiv_sort[cthread_idx], indiv_val[cthread_idx], INDIVIDUALS[cthread_idx], task_type[cthread_idx]);
		cudaStreamSynchronize(streams[cthread_idx % STREAM_NUM]);
		
		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "\n%s\n", cudaGetErrorString(cudaStatus));
		}
		
		iter_mutex.lock();
		if (!achieve[task_idx]) {
			achieve_num += 1;
			achieve[task_idx] = true;
		}
		iter_mutex.unlock();
	
		if (achieve_num == transfer_num || i == max_iter[task_idx] || i == ITER_NUM / INTERVAL_TRANSFER - 1) {
			int target = rand() % T;
			while (target == task_idx) {
				target = rand() % T;
			}
			cudaMemcpyAsync(&task_type[cthread_idx][1], &tasks_type[target], sizeof(int), cudaMemcpyHostToDevice, streams[cthread_idx % STREAM_NUM]);
			if (achieve[task_idx]) {
				iter_mutex.lock();
				waiting_num += 1;
				iter_mutex.unlock();
				if (waiting_num == transfer_num) {
                    last_transfer = transfer_num;
					finish_num = 0;
					wait_line.notify_all();
				}
				else {
					wait_line.wait(cond_lock_w, []{return waiting_num == transfer_num ? true : false;});
				}
                int memory_cost = 3 * ((int)(WARPSIZE / (TRANSFER_NUM / BLOCK_NUM) + 1)) * (TRANSFER_NUM / BLOCK_NUM) * sizeof(float);
				knowledgeTransfer << < BLOCK_NUM, THREAD_FOR_OPERA, memory_cost, streams[cthread_idx % STREAM_NUM]>> > (devStates+ _THREAD_NUM * BLOCK_NUM * cthread_idx % STREAM_NUM, task_type[cthread_idx], M_cpu[task_idx], b_cpu[task_idx], indiv_sort[cthread_idx], indiv_sort[target]);

				iter_mutex.lock();
				achieve_num -= 1;
				achieve[task_idx] = false;
				max_iter[task_idx] += 3;
                    finish_num += 1;
				if (achieve_num == 0) {
					iter_time += 1;
					waiting_num = 0;
					finish_line.notify_all();
				    iter_mutex.unlock();
				}
                else{
					iter_mutex.unlock();
					finish_line.wait(cond_lock_f, []{return finish_num == last_transfer ? true:false;});
                }
			}
		}
			
	}
	
	iter_mutex.lock();
	transfer_num -= 1;
	if (transfer_num == waiting_num) {
		wait_line.notify_all();
	}
	iter_mutex.unlock();
	
	popTransfer_(cthread_idx, &streams[cthread_idx % STREAM_NUM]);
	messageTransfer<<<BLOCK_NUM, _THREAD_NUM, 0, streams[cthread_idx % STREAM_NUM]>>>(INDIV_BEST_GPU[cthread_idx], INDIVVAL_BEST_GPU[cthread_idx], indiv_sort[cthread_idx]);
}

int main()
{
	int dev = 0;
	int supportsCoopLaunch = 0;
	
	cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
	printf("> %d\n", supportsCoopLaunch);
	
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("> Using Device %d: %s\n", dev, deviceProp.name);
	cudaSetDevice(dev);

	
	if (deviceProp.major < 3 || (deviceProp.major == 3 && deviceProp.minor < 5))
	{
		if (deviceProp.concurrentKernels == 0)
		{
			printf("> GPU does not support concurrent kernel execution (SM 3.5 "
				"or higher required)\n");
			printf("> CUDA kernel runs will be serialized\n");
		}
		else
		{
			printf("> GPU does not support HyperQ\n");
			printf("> CUDA kernel runs will have limited concurrency\n");
		}
	}

	//printf("> Compute Capability %d.%d hardware with %d multi-processors\n",
	//	deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

	
	cudaEvent_t time_start, time_stop;
	cudaEventCreate(&time_start);
	cudaEventCreate(&time_stop);
	cudaEventRecord(time_start, 0);
	cudaEventSynchronize(time_start);
	initialization();


	cudaStream_t stream[STREAM_NUM];
	for (int i = 0; i < STREAM_NUM; ++i) {
		cudaStreamCreate(&stream[i]);
	}
	

	mutex t_mutex;
	//pthread_t threads[CPU_THREAD_NUM];
    thread threads[CPU_THREAD_NUM];
	int task_idx = 0;



	for (int i = 0; i < CPU_THREAD_NUM; ++i) {
        
		threads[i] = thread(iter, task_idx, i, stream);
		t_mutex.lock();
		task_idx += 1;
		t_mutex.unlock();
	}
	for (int i = 0; i < CPU_THREAD_NUM; ++i) {
		threads[i].join();
	}
	
	cudaEventRecord(time_stop, 0);
	cudaEventSynchronize(time_stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, time_start, time_stop);
	printf("\nruntime=%f ms, %f, %f \n", elapsedTime, time_start, time_stop);
    //popTransfer();
	int task_debug = -1;
	for (int i = 0; i < T; ++i) {
		if (tasks_type[i] == 6) {
			task_debug = i;
			cudaDeviceSynchronize();
			break;
		}
	}
	
	//===================================parameter discussion:time
	char* T_num = new char[5], *DIMEN = new char[5];
	sprintf(DIMEN,"%d",DIMENSION);
	ofstream file;
	file.open("time_dimension_self.txt", ios::app);
	file << elapsedTime << ", " << "#" << DIMEN;
	file << endl;
	file.close();
	//===================================
	/*
	//===================================parameter discussion
	ofstream file;
	file.open("time_indivsperblock.txt", ios::app);
	file << elapsedTime << ", " << "\\" << INDIV_PERBLOCK << ", " << STREAM_NUM;
	file << endl;
	file.close();
	//===================================
	*/

	/*
	string s;
	stringstream ss;
	ss << setprecision(8) << elapsedTime;
	s = ss.str();
	ss.clear();
	ofstream file;
	char* T_num = new char[5], *IPB_num = new char[2];
	sprintf(T_num,"%d",T);
	sprintf(IPB_num, "%d", INDIV_PERBLOCK);
	for(int k = 0; k < T; ++k){
	int min = INT_MAX, max = -1;
	for(int i = 0; i < ITER_NUM / INTERVAL_TRANSFER + 1; ++i){
		if(record[k][0][i].eval[0] < min){
			min = record[k][0][i].eval[0];
		}
		if(record[k][1][i].eval[0] > max){
			max = record[k][1][i].eval[0];
		}
	}
	file.open(string("./data_record_T_ANO_") + T_num + string(".txt"), ios::app);
	for(int i = 0; i < ITER_NUM / INTERVAL_TRANSFER + 1; ++i){
		//printf("%f ", record[k][0][i].eval[0]);
		//if(record[k][0][i].eval[0] < 0){
		//	printf("task_val: %f, task_id:%d, task_type:%d\n", record[k][0][i].eval[0], k, tasks_type[k]);
		//	break;
		//}
		file << record[k][0][i].eval[0] << ' ';
	}
	file << record[k][1][0].eval[0] << ' ';
	file << endl;
	file.close();
	}
	file.open(string("./data_record_T_ANO_") + T_num + string(".txt"), ios::app);
	file << endl;
	file.close();
	//printf("count：%d\n", pv.getCount());
	*/
	/*
	string s;
	stringstream ss;
	ss << setprecision(8) << elapsedTime;
	s = ss.str();
	ss.clear();
	ofstream file;
	file.open("./data.dat", ios::app);
	file << "runtime: " << s << " dimension: " << DIMENSION << endl;
	file.close();
	
	bool f = false;
	for (int i = 0; i < BESTNUM_PERTASK; ++i) {
		float result = 0;
		float result_1 = 0;
		printf("\n\n\n%d, [", tasks_type[task_debug]);
		for (int j = 0; j < DIMENSION; ++j) {
			printf("%f ", INDIV_BEST_CPU[task_debug][i * DIMENSION + j]);
			result += INDIVIDUALS_CPU[task_debug][i * DIMENSION + j] * INDIVIDUALS_CPU[task_debug][i * DIMENSION + j];
			result_1 += cosf(2 * 3.1415926 * INDIVIDUALS_CPU[task_debug][i * DIMENSION + j]);
		}
		printf("] \n");
		printf("eval: %f\n\n", INDIVVAL_BEST_CPU[task_debug][i]);
		result = -20 * expf(-0.2 * sqrtf(result / 50)) - expf(result_1 / 50) + 20 + 2.718282;
		if(fabs(result - INDIV_VAL_CPU[task_debug][i]) > 1){
			f = true;
			printf("result:%f\n", result);
			break;
		}
	}
	if(f){
		printf("sth wrong..\n");
	}
	*/
	
}
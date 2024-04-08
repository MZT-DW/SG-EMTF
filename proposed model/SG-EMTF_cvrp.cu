//此版本为修改shared_memory为动态内存前的版本
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include "curand_kernel.h"
#include "cublas_v2.h"
// #include "device_functions.h"
#include<time.h>
#include<iostream>
#include<thread>
#include<mutex>
#include<condition_variable>
#include <fstream>
#include <cooperative_groups.h>
#include <sstream>
#include <string>
#include"semaphore.h"
#include<vector>
#include<map>
#include<algorithm>
#include<unistd.h>
#include<sys/time.h>
#include<queue>
#include<chrono>

#include <stdio.h>

#define RAND_MAX 32767
#define INDIVNUM 512//任务个体数
#define INDIVNUM_ISLAND 256//种群个体数
#define SHARED_CAPACITY 10240//共享内存大小
#define DIMENSION 500//维度大小
#define CPU_THREAD_NUM 1000//CPU线程数
#define ITER_NUM 1000//迭代数
#define INTERVAL_TRANSFER 50//知识迁移间隔
#define INTERVAL_MIGRA 25//种群迁移间隔
#define TRANSFER_NUM 50//知识迁移的个体数
#define T 1000//任务数
#define ISLAND_NUM 2//孤岛数
#define VAL_TYPE 4 //4字节浮点数变量，即float
#define BANKSIZE 32 //
#define WARPSIZE 32
#define BLOCK_NUM ((INDIVNUM - 1) / (INDIV_PERBLOCK * ISLAND_NUM) + 1)//最后一个Block可能人数不足
#define F 0.5
#define CROSSOVER_RATE 0.6
#define ETA 2.f //模拟二进制交叉参数
#define B_BASEVAL 100
#define PM 2.f
#define DM 5.f //多项式变异参数
#define THREAD_FOR_OPERA 416  //(SHARED_CAPACITY / (VAL_TYPE * DIMENSION * 3)) //每个block里一个thread服务一个个体
#define THREAD_FOR_TRANSFER 96 //用于数据传输的线程数
#define _THREAD_NUM (THREAD_FOR_OPERA + THREAD_FOR_TRANSFER) //GPU一个kernel的线程数
#define E 2.718282 
#define FULL_MASK 0xffffffff
#define TASK_TYPENUM 7 //评估函数个数
#define MIGRA_NUM 32
// #define MIGRA_PERBLOCK MIGRA_NUM / BLOCK_NUM
// #define MIGRA_PROP 1/8 //种群迁移的个体比例
// #define MIGRA_PERBLOCK ((MIGRA_NUM - 1) / INDIV_PERBLOCK + 1)//(INDIV_PERBLOCK * MIGRA_PROP) //种群迁移个数
#define BESTNUM_PERTASK 50 //传回GPU的种群个体排序个数
#define PI 3.1415926 
#define LOOPTIME 1 //每个Block内部数据传输间隙的循环次数
#define SELECT_INTERVAL INDIV_PERBLOCK * ISLAND_NUM * INTERVAL_TRANSFER
#define BATCH_NUM 100


#define INDIV_PERBLOCK 4//(SHARED_CAPACITY / (VAL_TYPE * 3 * DIMENSION))
#define STREAM_NUM 8 //流数量

#define CAPACITY 5

using namespace std;

//pthread_cond_t wait_line, finish_line;

struct DS {
	float eval[BANKSIZE];
	float* pointer[BANKSIZE];
	float* eval_pointer[BANKSIZE];
};

/*
struct Args{
	float* INDIVIDUALS;
	int* syncval;
	curandState* states;
	int* type;
	float* M;
	float* b;
	float* eval;
};
*/

int** paras = new int*[T];
int* tasks_split = new int[BATCH_NUM + 1];
float* st_time = new float[BATCH_NUM];
float** INDIVIDUALS = new float*[CPU_THREAD_NUM];
float** INDIV_BEST_GPU = new float*[CPU_THREAD_NUM];
float** INDIVVAL_BEST_GPU = new float*[CPU_THREAD_NUM];
float** middle_val = new float*[CPU_THREAD_NUM];
float** indiv_val = new float*[CPU_THREAD_NUM];
DS** indiv_sort = new DS*[CPU_THREAD_NUM];
curandState* devStates;
int** syncval = new int*[CPU_THREAD_NUM];
float** M = new float*[T];
float** dis = new float*[T];
float** b = new float*[T];
float** dis_cpu = new float*[T];
float** M_cpu = new float*[T];
float** b_cpu = new float*[T];
int tasks_type[T];
__constant__ float Range[TASK_TYPENUM * 2];
int achieve_num = 0, iter_time = 0, waiting_num = 0, transfer_num = T, finish_num = 0;
int max_iter[T];
__device__ float Weierstrass_para;
pthread_mutex_t iter_mutex=PTHREAD_MUTEX_INITIALIZER;
// mutex iter_mutex;
condition_variable wait_line, finish_line;
semaphore pv(5);
int* select_interval[T];
int* task_type[T];

bool achieve[T];



float** INDIVIDUALS_CPU = new float*[CPU_THREAD_NUM];
float** INDIV_VAL_CPU = new float*[CPU_THREAD_NUM];
float** INDIV_BEST_CPU = new float*[CPU_THREAD_NUM];
float** INDIVVAL_BEST_CPU = new float*[CPU_THREAD_NUM];


__device__ __forceinline__ void namedBarrierSync(int name, int numThreads){
	asm volatile("bar.sync %0, %1;": : "r"(name), "r"(numThreads):"memory");
}




__device__ void randDb(float down, float up, float* value, curandState* state) {//存在问题需要修改，种子是每个thread一个，这里只能用到一个kernel的种子
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	*value = curand_uniform(&state[tid]) * (up - down) + down;//curand(&state[tid]) % int(up - down) + down + float(curand(&state[tid]) % 1000) / 1000.0;
}

__device__ void rand_uniform(float* value, curandState* state) {

	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	*value = curand_uniform(&state[tid]);
}

__device__ void CrossOpera(float val_1, float val_2, float val_3, float* ret) {
	*ret = val_1 + F * (val_2 - val_3);
}

__global__ void pop_init(curandState* states, float* INDIVIDUALS, int* type, float* eval, int* para) {//种群初始化
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const int t_n = gridDim.x * blockDim.x;
	int customer_num = para[2], car_num = para[1], dimension = para[0];
	while (tid < INDIVNUM * dimension) {
		if(tid % dimension < customer_num){
			INDIVIDUALS[tid] = (tid % customer_num) + 1;
		}
		else{
			INDIVIDUALS[tid] = 0;
		}
		tid += t_n;
	}
	
	namedBarrierSync(0, _THREAD_NUM);
	
	if (tid < INDIVNUM) {//目前采用单线程
		int init_id = tid * customer_num;
		for (int k = customer_num - 1; k >= 0; --k) {
			int target = curand(states) % (k + 1);
			int temp = INDIVIDUALS[k + init_id];
			INDIVIDUALS[k + init_id] = INDIVIDUALS[target + init_id];
			INDIVIDUALS[target + init_id] = temp;
		}
		
	}
	
	tid = blockIdx.x * blockDim.x + threadIdx.x;
	while (tid < INDIVNUM) {
		eval[tid] = INT_MAX;
		tid += t_n;
	}
}



__global__ void curandInit(curandState* states, int seed) {//随机函数初始化
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	int interval = blockDim.x * gridDim.x;
	int total_randVal = _THREAD_NUM * BLOCK_NUM * STREAM_NUM;
	while (tid < total_randVal) {
		curand_init(seed, tid, 0, &states[tid]);
		tid += interval;
	}
	//if(threadIdx.x + (blockIdx.x * blockDim.x) == 0){
	//	printf("rand_val:%d, %d\n", curand(&states[threadIdx.x + (blockIdx.x * blockDim.x)]), seed);
	//	printf("rand_val:%d, %d\n", curand(&states[threadIdx.x + (blockIdx.x * blockDim.x)]), seed);
	//}
	//if(threadIdx.x + (blockIdx.x * blockDim.x) == 1){
	//	printf("rand_val:%d, %d\n", curand(&states[threadIdx.x + (blockIdx.x * blockDim.x)]), seed);
	//	printf("rand_val:%d, %d\n", curand(&states[threadIdx.x + (blockIdx.x * blockDim.x)]), seed);
	//}
}


__global__ void para_init(curandState* states, float** dis) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const int t_n = gridDim.x * blockDim.x;
	int dim_max = DIMENSION + 1;
	int dim = dim_max * dim_max;
	while (tid < T * dim) {
		int idx = tid % dim;
		if(idx % dim_max == idx / dim_max){
			dis[tid / dim][idx] = 0;
		}
		else{
			dis[tid / dim][idx] = curand_uniform(&states[tid % (_THREAD_NUM * BLOCK_NUM * STREAM_NUM)]) * 100;
		}
		tid += t_n;
	}
}

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

void popTransfer() {
	
	for (int i = 0; i < CPU_THREAD_NUM; ++i) {
		INDIVIDUALS_CPU[i] = new float[INDIVNUM * DIMENSION];
		INDIV_VAL_CPU[i] = new float[INDIVNUM];
		INDIV_BEST_CPU[i] = new float[BESTNUM_PERTASK * DIMENSION];
		INDIVVAL_BEST_CPU[i] = new float[BESTNUM_PERTASK];
		
		cudaMemcpy(INDIVIDUALS_CPU[i], INDIVIDUALS[i], INDIVNUM * DIMENSION * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(INDIV_VAL_CPU[i], indiv_val[i], INDIVNUM * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(INDIV_BEST_CPU[i], INDIV_BEST_GPU[i], BESTNUM_PERTASK * DIMENSION * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(INDIVVAL_BEST_CPU[i], INDIVVAL_BEST_GPU[i], BESTNUM_PERTASK * sizeof(float), cudaMemcpyDeviceToHost);
	}
}

void popTransfer_(int cthread_idx, cudaStream_t* stream) {
		INDIVIDUALS_CPU[cthread_idx] = new float[INDIVNUM * DIMENSION];
		INDIV_VAL_CPU[cthread_idx] = new float[INDIVNUM];
		//INDIV_BEST_CPU[i] = new float[BESTNUM_PERTASK * DIMENSION];
		//INDIVVAL_BEST_CPU[i] = new float[BESTNUM_PERTASK];

		cudaMemcpyAsync(INDIVIDUALS_CPU[cthread_idx], INDIVIDUALS[cthread_idx], INDIVNUM * DIMENSION * sizeof(float), cudaMemcpyDeviceToHost, *stream);
		cudaMemcpyAsync(INDIV_VAL_CPU[cthread_idx], indiv_val[cthread_idx], INDIVNUM * sizeof(float), cudaMemcpyDeviceToHost, *stream);
		//cudaMemcpy(INDIV_BEST_CPU[i], INDIV_BEST_GPU[i], BESTNUM_PERTASK * DIMENSION * sizeof(float), cudaMemcpyDeviceToHost);
		//cudaMemcpy(INDIVVAL_BEST_CPU[i], INDIVVAL_BEST_GPU[i], BESTNUM_PERTASK * sizeof(float), cudaMemcpyDeviceToHost);

}

void paraSave(float** M_cpu, float** b_cpu, int* task_choice){
	ofstream file;
	file.open("para.txt", ios::out);
	float* M_C = new float[DIMENSION * T], *b_C = new float[DIMENSION * T];
	for(int i = 0; i < T; ++i){
		cudaMemcpy(M_C + DIMENSION * i, M_cpu[i], DIMENSION * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(b_C + DIMENSION * i, b_cpu[i], DIMENSION * sizeof(float), cudaMemcpyDeviceToHost);
	}
	for(int j = 0; j < T; ++j){
		for(int i = 0; i < DIMENSION; ++i){
			file << M_C[j * DIMENSION + i] << ' ' << b_C[j * DIMENSION + i] << ' ';
		}
		file << endl;
	}
	file << endl;
	for(int i = 0; i < T; ++i){
		file << task_choice[i] << ' ';
	}
	file << endl;
	file.close();
}

void getRandNum(int* a, int n, int min, int max){
	if(n == 0){
		return;
	}
	int rd = rand() % (max - min + 1) + min;
	for(int i = 0; i < n; ++i){
		while(1){
			bool flag = true;
			for(int j = 0; j < i; ++j){
				if(rd == a[j]){
					flag = false;
					break;
				}
			}
			if(flag){
				break;
			}
			rd = rand() % (max - min + 1) + min;
		}
		a[i] = rd;
	}
}

float getRandFloat(float min, float max){
	return ((float)(rand() % RAND_MAX) / (float)RAND_MAX) + min + float(rand() % int(max - min));
}

void initialization() {//数据传输至显存，随机数种子的初始化

		//初始化事件
    //pthread_cond_init(&wait_line, NULL);
    //pthread_cond_init(&finish_line, NULL);

					   //个体生成和初始化
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
	srand(0);
	for (int i = 0; i < CPU_THREAD_NUM; ++i) {
        achieve[i] = false;
        max_iter[i] = 1;
		INDIVIDUALS[i] = new float[INDIVNUM * DIMENSION];
		indiv_val[i] = new float[INDIVNUM];
		indiv_sort[i] = new DS[(INDIVNUM - 1) / BANKSIZE + 1];//上取整
		middle_val[i] = new float[INDIVNUM * DIMENSION];
		cudaMalloc((void**)&select_interval[i], SELECT_INTERVAL * sizeof(int));
		cudaMalloc((void**)&INDIVIDUALS[i], INDIVNUM * DIMENSION * sizeof(float));
		cudaMalloc((void**)&indiv_val[i], INDIVNUM * sizeof(float));
		cudaMalloc((void**)&INDIV_BEST_GPU[i], BESTNUM_PERTASK * DIMENSION * sizeof(float));
		cudaMalloc((void**)&INDIVVAL_BEST_GPU[i], BESTNUM_PERTASK * sizeof(float));
		cudaMalloc((void**)&indiv_sort[i], ((INDIVNUM - 1) / BANKSIZE + 1) * BANKSIZE * (sizeof(float) + sizeof(float*) + sizeof(float*)));
		cudaMalloc((void**)&middle_val[i], ((BLOCK_NUM * sizeof(int))));
		cudaMalloc((void**)&task_type[i], sizeof(int) * 2);

		cudaMemcpy(task_type[i], &tasks_type[i], sizeof(int), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&syncval[i], sizeof(int) * ((INTERVAL_TRANSFER - 1) / LOOPTIME + 1) * ISLAND_NUM);
		cudaMemcpy(syncval[i], &syncval_cpu, sizeof(int) * ((INTERVAL_TRANSFER - 1) / LOOPTIME + 1) * ISLAND_NUM, cudaMemcpyHostToDevice);

	}
		cudaMemcpy(task_choice, &tasks_type, sizeof(int) * T, cudaMemcpyHostToDevice);
	//适应度函数的参数矩阵

	for (int i = 0; i < T; ++i) {
		float* devM_1d, *devb_1d;
		cudaMalloc((void**)&devM_1d, DIMENSION * sizeof(float));
		M_cpu[i] = devM_1d;
		cudaMalloc((void**)&devb_1d, DIMENSION * sizeof(float));
		b_cpu[i] = devb_1d;
	}
	
	for (int i = 0; i < T; ++i) {
		float* devM_1d, *devb_1d;
		cudaMalloc((void**)&devM_1d, DIMENSION * sizeof(float));
		M_cpu[i] = devM_1d;
		cudaMalloc((void**)&devb_1d, DIMENSION * sizeof(float));
		b_cpu[i] = devb_1d;
	}
	for(int i = 0; i < T; ++i){
		float* devDis_1d;
		cudaMalloc((void**)&devDis_1d, (DIMENSION + 1) * (DIMENSION + 1) * sizeof(float));
		dis_cpu[i] = devDis_1d;
	}
	for(int i = 0; i < T; ++i){
		int* para_1d;
		cudaMalloc((void**)&para_1d, sizeof(int) * 3);
		paras[i] = para_1d;
	}

	cudaMalloc((void**)&M, sizeof(float*) * T);
	cudaMalloc((void**)&b, sizeof(float*) * T);
	cudaMalloc((void**)&dis, sizeof(float*) * T);
	cudaMemcpy(dis, dis_cpu, sizeof(float*) * T, cudaMemcpyHostToDevice);
	cudaMemcpy(M, M_cpu, sizeof(float*) * T, cudaMemcpyHostToDevice);
	cudaMemcpy(b, b_cpu, sizeof(float*) * T, cudaMemcpyHostToDevice);


	float a_ = 1, b_ = 1, res = 0;
	for (int i = 0; i <= 20; ++i) {
		res += a_ * cosf(2 * PI * b_ * 0.5);
		a_ *= 0.5;
		b_ *= 3;
	}
	//cudaMalloc((void**)&Weierstrass_para, sizeof(float));
	cudaMemcpyToSymbol(Weierstrass_para, &res, sizeof(float));

	//生成cuda上的随机数种子，为了保证随机性，每个个体分配一个初始种子
	int total_threadnum = _THREAD_NUM * BLOCK_NUM * STREAM_NUM;

	cudaMalloc((void**)&devStates, sizeof(curandState) * total_threadnum);

	int blocknum = STREAM_NUM;
	curandInit << <blocknum * BLOCK_NUM, _THREAD_NUM >> > (devStates, 0);
	//curandInit << <blocknum * BLOCK_NUM, _THREAD_NUM >> > (devStates, rand());

	para_init << <T, INDIVNUM >> > (devStates, dis);
	//curandInit << <blocknum * BLOCK_NUM, _THREAD_NUM >> > (devStates, rand());
	

	srand(time(0));
	curandInit << <blocknum * BLOCK_NUM, _THREAD_NUM >> > (devStates, rand());

	paraSave(M_cpu, b_cpu, tasks_type);
}

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
//双调排序
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

__device__ void bitonic_sort(int* idx, float* indivs_val, int num) {//需要确保一个个体能分到一个线程
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < num) {
		for (unsigned int i = 2; i < num; i <<= 1) {
			for (unsigned int j = i >> 1; j > 0; j >>= 1) {
				unsigned int tid_comp = tid ^ j;

				if (tid_comp > tid && tid_comp < num) {
					if ((tid & i) == 0) {
						if (indivs_val[idx[tid]] > indivs_val[idx[tid_comp]]) {
							//printf("%f, %f\n", temp[DS_idx].eval[inner_idx], *(temp[DS_idx].eval_pointer[inner_idx]));
							_swap(&idx[tid], &idx[tid_comp]);

						}
					}
					else {
						if (indivs_val[idx[tid]] < indivs_val[idx[tid_comp]]) {
							_swap(&idx[tid], &idx[tid_comp]);
						}
					}
				}
				__syncthreads();
			}
		}
	}
	
}


__global__ void popSort(DS* indiv_sort, float* indiv_val, float* INDIVIDUALS, int* type) {
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	int DS_size = (INDIVNUM - 1) / BANKSIZE + 1;
	int t_n = blockDim.x * gridDim.x;

	//赋值个体适应度值到shared memory中
	__shared__ DS temp[(INDIVNUM - 1) / BANKSIZE + 1];//上取整
	while (tid < INDIVNUM) {//是单值单线程调用还是一次全部调用更快？
		temp[tid / BANKSIZE].eval[tid % BANKSIZE] = indiv_val[tid];
		temp[tid / BANKSIZE].pointer[tid % BANKSIZE] = &INDIVIDUALS[tid * DIMENSION];
		temp[tid / BANKSIZE].eval_pointer[tid % BANKSIZE] = &indiv_val[tid];
		tid += t_n;
	}
	__syncthreads();

	//双调排序
	bitonic_sort(temp, INDIVNUM);

	//赋值回global memory
	tid = threadIdx.x + (blockIdx.x * blockDim.x);
	while (tid < DS_size) {
		indiv_sort[tid] = temp[tid];
		tid += t_n;
	}
}

__device__ void getSum(int new_dim, float* temp_middle, int indivs_num, int indiv_perblock = INDIV_PERBLOCK) {
	const int t_n = THREAD_FOR_OPERA;
	while (true) {//数组求和
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
			if (old_dim % 2 == 1 && cur_dim + new_dim == old_dim - 2) {//如果维数为奇数，则多的一维并入最后一维
				temp_val += temp_middle[(old_dim - 1) * indiv_perblock + tid % indivs_num];
			}
			temp_middle[cur_dim * indiv_perblock + tid % indivs_num] += temp_val;
			tid += t_n;
		}
	}
}

__device__ void getMulti(int new_dim, float* temp_middle, int indivs_num, int indiv_perblock = INDIV_PERBLOCK) {
	const int t_n = THREAD_FOR_OPERA;
	while (true) {//数组求和
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
			if (old_dim % 2 == 1 && cur_dim + new_dim == old_dim - 2) {//如果维数为奇数，则多的一维并入最后一维
				temp_val *= temp_middle[(old_dim - 1) * indiv_perblock + tid % indivs_num];
			}
			temp_middle[cur_dim * indiv_perblock + tid % indivs_num] *= temp_val;
			tid += t_n;
		}
	}

}



__device__ float atomicMul(float* address, float val) 
{ 
  unsigned int* address_as_ull = (unsigned int*)address; 
  unsigned int old = *address_as_ull, assumed; 
  do { 
 assumed = old; 
 old = atomicCAS(address_as_ull, assumed, __float_as_int(val * __int_as_float(assumed))); 
 } while (assumed != old); return __int_as_float(old);
} 

__device__ void Mutation(int indiv_num, int n, curandState* state,  float* indivs_in_s, float* rand) {
	int tid = threadIdx.x;
	int t_n = THREAD_FOR_OPERA;
	int indiv_perblock = INDIV_PERBLOCK;
	int group_interval = indiv_perblock * DIMENSION;

	float range = rand[1] - rand[0];
    float P = 1.f;

	/*
	while(tid < indiv_num){
		r[tid + indiv_perblock * 3] = curand(state) % DIMENSION;
		tid += t_n;
	}
	*/
	tid = threadIdx.x;
	//变异
	while (tid < indiv_num * DIMENSION) {
		/*
		if(curand_uniform(state) > CROSSOVER_RATE && tid / indiv_num != int(r[tid % indiv_num + indiv_perblock * 3])){
			indivs_in_s[(tid / indiv_num) * indiv_perblock + tid % indiv_num + group_interval * 2] = indivs_in_s[(tid / indiv_num) * indiv_perblock + tid % indiv_num + group_interval * (1 - n)];
		}
		*/
		
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
				//randDb(range[0], range[1], &temp_indiv_1, states);
			}
			else if (temp_indiv_1 < rand[0]) {
				temp_indiv_1 = rand[0];
				//randDb(l_k, range[1], &temp_indiv_1, states);
			}
			indivs_in_s[group_interval * 2 + indiv_idx] = temp_indiv_1;
            
		}
        
		tid += THREAD_FOR_OPERA;
	}
}

/**
 * @name: CrossOver
 * @description:
 		模拟二进制交叉方案 
 * @param: 
		indiv_num:用于交叉的个体数量
		state:随机数种子，通过curand_uniform(state)等函数生成随机数；
		indivs_in_s: block内的个体，以每个维度的所有个体为单位:
			排列方式为：indiv_00, indiv_10, ...,indiv_01, indiv_11, ...；
			大小为：indiv_num * DIMENSION
		rand:维度上下限，rand[0]为下限，rand[1]为上限
 * @return 
 */
__device__ void CrossOver(int indiv_num, int n, curandState* state,  float* indivs_in_s, float* rand, float* r, float* indivs, int* para) {
	
	//===============================================================================
	//基本参数设置
	int tid = threadIdx.x;//线程Id号，每个线程负责一个个体的部分维度的计算
	int t_n = THREAD_FOR_OPERA;//线程总数
	int indiv_perblock = INDIV_PERBLOCK;//indivs_in_s的数组大小除以维度
	int group_interval = indiv_perblock * para[0];//indivs_in_s的最大允许数组大小
	
	//===============================================================================
	
	int dim = para[0];
	while(tid < indiv_num * dim){
		int init_posi = int(tid / indiv_num) * indiv_num;
		int idx = tid % indiv_num;//个体id
		int r0 = r[idx], r1 = r[(idx + 1) % indiv_num];
		float rate = curand_uniform(state);
		if(rate >= 0){
			indivs_in_s[group_interval * 2 + init_posi + idx] = indivs_in_s[group_interval * (1 - n) + init_posi + idx] + 1 * (indivs_in_s[group_interval * (1 - n) + init_posi + r0] - indivs_in_s[group_interval * (1 - n) + init_posi + r1]);
		}
		tid += THREAD_FOR_OPERA;
	}
	int temp[INDIV_PERBLOCK * DIMENSION > THREAD_FOR_OPERA ? INDIV_PERBLOCK * DIMENSION / THREAD_FOR_OPERA : 1];
	extern __shared__ int count[], zero_id[];
	tid = threadIdx.x;
	while(tid < indiv_num){
		int idx = tid % indiv_num, cur_dim = tid / indiv_num;
		count[tid] = 0;
		zero_id[tid] = 0;
		tid += THREAD_FOR_OPERA;
	}
	namedBarrierSync(2, THREAD_FOR_OPERA);
	tid = threadIdx.x;
	while(tid < indiv_num * dim){
		int idx = tid % indiv_num, cur_dim = tid / indiv_num;
		if(indivs_in_s[group_interval * 2 + cur_dim * indiv_num + idx] == 0){
			atomicAdd(&count[idx], 1);
		}
		if(count[idx] + 1 == para[2]){
			zero_id[idx] = cur_dim;
		}
		tid += THREAD_FOR_OPERA;
	}
	
	namedBarrierSync(2, THREAD_FOR_OPERA);
	tid = threadIdx.x;
	
	while(tid < indiv_num * dim){
		int idx = tid % indiv_num, cur_dim = tid / indiv_num;
		int cur_rank = 0, cur_val = indivs_in_s[group_interval * 2 + cur_dim * indiv_num + idx];
		if(cur_val != 0 || cur_dim > zero_id[idx]){
			for(int i = 0; i < dim; ++i){
				if(indivs_in_s[group_interval * 2 + i * indiv_num + idx] < cur_val || (indivs_in_s[group_interval * 2 + i * indiv_num + idx] == cur_val && i < cur_dim)){
					cur_rank += 1;
				}
			}
			temp[int(tid / THREAD_FOR_OPERA)] = cur_rank + 1;
		}
		else{
			temp[int(tid / THREAD_FOR_OPERA)] = 0;
		}
		tid += THREAD_FOR_OPERA;
	}
	tid = threadIdx.x;
	while(tid < indiv_num * dim){
		int idx = tid % indiv_num, cur_dim = tid / indiv_num;
		indivs_in_s[group_interval * 2 + cur_dim * indiv_num + idx] = 1;
		tid += THREAD_FOR_OPERA;
	}
	
}

__device__ void Selection(int indiv_num, int n,  float* indivs_eval,  float* indivs_in_s, int*para) {
	__shared__ int temp_judge[INDIV_PERBLOCK];
	int tid = threadIdx.x;
	int t_n = THREAD_FOR_OPERA;
	int indiv_perblock = INDIV_PERBLOCK;
	int group_interval = indiv_perblock * para[0];

	while (tid < indiv_num * para[0]) {
		if (indivs_eval[tid % indiv_num + indiv_perblock * (1 - n)] >= indivs_eval[indiv_perblock * 2 + tid % indiv_num]) {
			int indiv_idx = tid / indiv_num * indiv_perblock + tid % indiv_num;
			indivs_in_s[group_interval * (1 - n) + indiv_idx] = indivs_in_s[group_interval * 2 + indiv_idx];
			if(tid / indiv_num == 0){
				indivs_eval[tid % indiv_num + indiv_perblock * (1 - n)] = indivs_eval[indiv_perblock * 2 + tid % indiv_num];
			}
		}
		tid += THREAD_FOR_OPERA;
	}
}


__device__ void CrossPrep(float* r, int indiv_num, int t_n, int n, curandState* state, int init_thread) {
	int indiv_perblock = INDIV_PERBLOCK;
	int tid = threadIdx.x - init_thread;
	int init_posi = (1 - n) * indiv_perblock * LOOPTIME;
	
	tid = threadIdx.x - init_thread;
	int half_choose = indiv_num;
	while (tid < half_choose * LOOPTIME) {//随机选择双亲，允许覆盖(此处交叉概率取1）
		int r1 = curand(state) % indiv_num;
			r[init_posi + (tid / half_choose) * indiv_perblock + tid] = r1;// + beta;//整数表示下标，小数点后表示beta值
		tid += t_n;
	}
	namedBarrierSync(1, _THREAD_NUM - init_thread);
}

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
__device__ void SharedToGlobal(float* indivs,  float* indivs_in_s,  float* indivs_eval, float* eval, curandState* state, int* shuffle, int n, int n_island,  int thread_num, int* para) {
	int threads_for_opera = _THREAD_NUM - thread_num;
	int tid = threadIdx.x - threads_for_opera, t_n = thread_num;
	int indiv_perblock = INDIV_PERBLOCK, group_interval = indiv_perblock * para[0];
	int init_posi = blockIdx.x * (group_interval * ISLAND_NUM);
	//洗牌算法
	while (tid < indiv_perblock) {
		shuffle[tid] = tid;
		tid += t_n;
	}
	
	namedBarrierSync(1, thread_num);
	tid = threadIdx.x - threads_for_opera;
	
	if (tid == 0) {//目前采用单线程
		
		for (int k = indiv_perblock - 1; k >= 0; --k) {
			int target = curand(state) % (k + 1);
			int temp = shuffle[k];
			shuffle[k] = shuffle[target];
			shuffle[target] = temp;
		}
	}
	
	tid = threadIdx.x - threads_for_opera;
		//if(blockIdx.x == 0 && threadIdx.x > THREAD_FOR_OPERA && t_n == _THREAD_NUM){
			//printf("tid:%d\n", tid);
		//}
	namedBarrierSync(1, thread_num);
	while (tid < indiv_perblock * para[0]) {
		indivs[init_posi + indiv_perblock * para[0] * n_island + shuffle[tid / para[0]] * para[0] + tid % para[0]] = indivs_in_s[group_interval * n + (tid % para[0]) * indiv_perblock + (tid / para[0])];
		tid += t_n;
	}

	tid = threadIdx.x - threads_for_opera;
	while (tid < indiv_perblock) {
		eval[blockIdx.x * indiv_perblock * ISLAND_NUM + shuffle[tid] + indiv_perblock * n_island] = indivs_eval[indiv_perblock * n + tid];
		tid += t_n;
	}

}

__device__ void reduce_sum(float* x, int len, int tid, int idx){
	int total_tnum = len >> 1;
	// printf("init total_tnum: %d, %d, %d, %d\n", total_tnum, tid, idx, len);
	if (tid < total_tnum){
		for(int i = len >> 1; i > 0; i >>= 1){
			if(tid < i){
				x[tid] += x[tid + i];
			}
			// if(len % 2 == 1 && tid == 0){
			// 	x[tid] += x[len - 1];
			// }
			// len >>= 1;
			// printf("total_tnum: %d, %d, %d, %d\n", total_tnum, tid, idx, len);
			// __threadfence();
			// namedBarrierSync(idx, total_tnum);
		}
	}
}

__device__ void evaluation(float* dis_s, float* indivs_in_s, int indivs_num,  float* eval, int indiv_perblock, int* para, int syncid) {//函数评估
	int tid = threadIdx.x;
	
	int dim = para[0];
	tid = threadIdx.x;

	extern __shared__ float car_num[];
	while(tid < indivs_num){
		car_num[tid] = 0;
		tid += THREAD_FOR_OPERA;
	}
	extern __shared__ float cost[];
	while(tid < indivs_num * dim){
		cost[tid] = 0;
		tid += THREAD_FOR_OPERA;
	}
	namedBarrierSync(syncid, THREAD_FOR_OPERA);
	
	tid = threadIdx.x;
	
	while(tid < dim * indivs_num){
		int cur_dim = tid / indivs_num, cur_indiv = tid % indivs_num;
		int source_id = indivs_in_s[indiv_perblock * cur_dim + cur_indiv];
		int target_id = indivs_in_s[indiv_perblock * (cur_dim + 1) + cur_indiv];
		if(indivs_in_s[indiv_perblock * cur_dim + cur_indiv] == 0){
			atomicAdd(&car_num[cur_indiv], 1);
		}
		if(cur_dim < dim - 1){
			cost[cur_indiv * dim + cur_dim] = dis_s[source_id * (DIMENSION + 1) + target_id];
		}
		else if(cur_dim == dim - 1){
			cost[cur_indiv * dim + cur_dim] = dis_s[source_id * (DIMENSION + 1)];
		}
		tid += THREAD_FOR_OPERA;
	}
	namedBarrierSync(syncid, THREAD_FOR_OPERA);
	tid = threadIdx.x;
	while(tid < dim * indivs_num){
		reduce_sum(cost + int(tid / dim) * dim, dim, tid % dim, tid / dim + 1);
		tid += THREAD_FOR_OPERA;
	}
	if(threadIdx.x < indivs_num){
		cost[threadIdx.x * dim] += dis_s[int(indivs_in_s[threadIdx.x])];
	}
	namedBarrierSync(syncid, THREAD_FOR_OPERA);
	if(threadIdx.x < indivs_num){
		eval[threadIdx.x] = 0.5 * cost[threadIdx.x * dim] + 0.5 * car_num[threadIdx.x];
	}
	namedBarrierSync(syncid, THREAD_FOR_OPERA);
	tid = threadIdx.x;
	while(tid < indivs_num * dim){
		cost[tid] = 0;
		tid += THREAD_FOR_OPERA;
	}
	namedBarrierSync(syncid, THREAD_FOR_OPERA);
	tid = threadIdx.x;
	while(tid < indivs_num * dim){
		int cur_dim = tid / indivs_num, cur_indiv = tid % indivs_num;
		int custom_id = indivs_in_s[indiv_perblock * cur_dim + cur_indiv];
		atomicAdd(&cost[cur_indiv * dim + custom_id], 1);
		tid += THREAD_FOR_OPERA;
	}
	namedBarrierSync(syncid, THREAD_FOR_OPERA);
	tid = threadIdx.x;
	while(tid < indivs_num * para[1]){
		int cur_dim = tid / indivs_num, cur_indiv = tid % indivs_num;
		if(cost[cur_indiv * dim + cur_dim] == 0){
			eval[cur_indiv] = 10000000;
		}
		tid += THREAD_FOR_OPERA;
	}
	namedBarrierSync(syncid, THREAD_FOR_OPERA);
	tid = threadIdx.x;
	if(tid / indivs_num == 0){
		int cur_num = 0;
		for(int i = 0; i < dim; ++i){
			if(indivs_in_s[indiv_perblock * i + tid] == 0){
				cur_num = 0;
			}
			cur_num += 1;
			if(cur_num > CAPACITY){
				eval[tid] = 10000000;
				break;
			}
		}
	}
	namedBarrierSync(syncid, THREAD_FOR_OPERA);
	tid = threadIdx.x;
	while(tid < indivs_num * dim){
		int cur_dim = tid / indivs_num, cur_indiv = tid % indivs_num;
		int custom_id = indivs_in_s[indiv_perblock * cur_dim + cur_indiv];
		atomicAdd(&cost[cur_indiv * dim + custom_id], 1);
		tid += THREAD_FOR_OPERA;
	}
	namedBarrierSync(syncid, THREAD_FOR_OPERA);
	tid = threadIdx.x;
	while(tid < indivs_num * para[1]){
		int cur_dim = tid / indivs_num, cur_indiv = tid % indivs_num;
		if(cost[cur_indiv * dim + cur_dim] > 1){
			eval[cur_indiv] = 10000000;
		}
		tid += THREAD_FOR_OPERA;
	}
}


__device__ void paritySort(float* indivs_val, int* idx, int tid, int num) {//奇偶交换排序
    int max_num = ((num / 2) / WARPSIZE + 1) * WARPSIZE;
	if (tid < num) {
		// if (tid % 2 == 1) {
		// 	tid += num / 2;
		// 	tid += tid % 2;
		// }
		for (int i = 0; i < num; i++) {
			if(tid % 2 == 0){
				int temp_id = tid + (i % 2);
				if (temp_id + 1 < num) 
				{
					if(indivs_val[idx[temp_id]] > indivs_val[idx[temp_id + 1]]){
						_swap(&idx[temp_id], &idx[temp_id + 1]);
					}
				}
			}
			if(max_num > WARPSIZE){
				__syncthreads();
			}
		}
	}
}

__device__ void IslandMigration(int* idx_forSort, float* indivs_eval, float* indivs_in_s, int indiv_num, int* para) {
	int t_n = _THREAD_NUM;
	//===================================排序======================================
	int tid = threadIdx.x;
	int indiv_perblock = INDIV_PERBLOCK;
	int arraySize = indiv_perblock;
	//初始化
	while (tid < 2 * arraySize) {
		idx_forSort[tid] = tid % arraySize;
		tid += t_n;
	}
	__threadfence();
	__syncthreads();


	tid = threadIdx.x;
    int max_num = ((indiv_num / 2) / WARPSIZE + 1) * WARPSIZE;//__syncthreads()最好以warp为单位，否则容易出现问题
	while (tid < max_num) {
		paritySort(indivs_eval, idx_forSort, tid, indiv_num);
	
		tid += t_n;
	}
	tid = threadIdx.x - max_num;
	while (tid < max_num && tid >= 0) {
		paritySort(indivs_eval + indiv_perblock, idx_forSort + arraySize, tid, indiv_num);
		tid += t_n;
	}

	__threadfence();
	__syncthreads();
	
	//岛间迁移
	tid = threadIdx.x;
	int migra_num = MIGRA_NUM / gridDim.x;
	if (blockIdx.x < MIGRA_NUM % gridDim.x) {
		migra_num += 1;
	}
	
	
	tid = threadIdx.x;
	while (tid < migra_num * para[0]) {
		int migra_id = tid % migra_num;
		int init_dimen = (tid / migra_num) * indiv_perblock;
		int targetId = idx_forSort[migra_id], targetId_1 = indiv_num + idx_forSort[arraySize + migra_id];
		int selfId = indiv_num + idx_forSort[arraySize + indiv_num - migra_id - 1], selfId_1 = idx_forSort[indiv_num - migra_id - 1];

		if (indivs_eval[targetId] < indivs_eval[selfId]) {
			// printf("%d, %d\n", idx_forSort[arraySize + indiv_num - migra_id - 1], idx_forSort[migra_id]);
			indivs_in_s[indiv_perblock * para[0] + init_dimen + idx_forSort[arraySize + indiv_num - migra_id - 1]] = indivs_in_s[init_dimen + idx_forSort[migra_id]];
		}
		if (indivs_eval[targetId_1] < indivs_eval[selfId_1]) {
			// printf("-%d, %d\n", idx_forSort[indiv_num - tid % MIGRA_NUM - 1], idx_forSort[arraySize + migra_id]);
			indivs_in_s[init_dimen + selfId_1] = indivs_in_s[indiv_perblock * para[0] + init_dimen + idx_forSort[arraySize + migra_id]];
		}
		tid += t_n;
	}

	__threadfence();
	__syncthreads();

	tid = threadIdx.x;
	while (tid < migra_num) {
		int selfId = indiv_num + idx_forSort[arraySize + indiv_num - tid - 1];
		int targetId = idx_forSort[tid];
		if (indivs_eval[targetId] < indivs_eval[selfId]) {
			indivs_eval[selfId] = indivs_eval[targetId];
		}
		tid += t_n;
	}
	tid = threadIdx.x - migra_num;
	while (tid < migra_num) {
		int targetId = indiv_num + idx_forSort[arraySize + tid];
		int selfId = idx_forSort[indiv_num - tid - 1];
		if (indivs_eval[targetId] < indivs_eval[selfId]) {
			indivs_eval[selfId] = indivs_eval[targetId];
		}
		tid += t_n;
	}

	
}

__device__ void SelectFromBlocks(int n,  int n_island, float* indivs_in_s,  float* indivs_eval, float* indivs, float* eval, int* select_interval, int iter, int* para) {
	int tid = threadIdx.x - THREAD_FOR_OPERA;
	int indiv_perblock = INDIV_PERBLOCK, group_interval = indiv_perblock * para[0];
	__shared__ int targetblock[INDIV_PERBLOCK];

	int temp_k = 0;
	while (tid < indiv_perblock) {
		targetblock[tid] = (blockIdx.x + select_interval[iter * indiv_perblock + tid]) % gridDim.x;
		/*
		if (targetblock[tid] == gridDim.x - 1) {
			if (tid >= indiv_final[n]) {
				if(temp_k)
				do{
					targetblock[tid] = (gridDim.x - 1 + select_interval[iter * INDIV_PERBLOCK + tid + temp_k]) % gridDim.x;//如果再次指向gridDim.x-1的话，，感觉会死循环？
					temp_k += 1;
				}while(targetblock[tid] == gridDim.x - 1);
			}
			else{
				interval = indiv_final[0];
			}
		}
		*/
		indivs_eval[indiv_perblock * n + tid] = eval[indiv_perblock * targetblock[tid] * ISLAND_NUM + indiv_perblock * n_island + tid];
		tid += THREAD_FOR_TRANSFER;
	}

	namedBarrierSync(1, THREAD_FOR_TRANSFER);
	tid = threadIdx.x - THREAD_FOR_OPERA;
	while (tid < indiv_perblock * para[0]) {
		//从自己开始，往后读取第几个block? (tid / (2 * para[0])) % BLOCK_NUM + 1,不用+1
		int indiv_id = tid / para[0];
		/*
		if (targetblock[indiv_id] == gridDim.x - 1) {
				interval = indiv_final[0];
		}
		*/
		indivs_in_s[group_interval * n + (tid % para[0]) * indiv_perblock + indiv_id] = indivs[group_interval * targetblock[indiv_id] * ISLAND_NUM + indiv_perblock * DIMENSION * n_island + tid];
		tid += THREAD_FOR_TRANSFER;
	}

}


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
 

__global__ void GeneticOpera(float* indivs, volatile int* syncval, curandState* states, int* type, float* dis, float* eval, int* select_interval, int* para) {
	const int dis_dim = para[1] + 1, dimension = para[0];
	 extern __shared__ float indivs_in_s[];
	 __shared__ float indivs_eval[INDIV_PERBLOCK * 3];//保存值
	__shared__ int idx_forSort[INDIV_PERBLOCK * 2];
	__shared__ int shuffle[INDIV_PERBLOCK];

	float range[2];//最大值和最小值
	range[0] = Range[*type * 2];
	range[1] = Range[*type * 2 + 1];

	curandState state = states[blockDim.x * blockIdx.x + threadIdx.x];
	int tid = threadIdx.x;
	int indiv_perblock = INDIV_PERBLOCK;
	int group_interval = indiv_perblock * dimension;//SHARED_CAPACITY / (VAL_TYPE * 3);//个体向量、子个体向量、下一步迭代需要的向量：总共需要三份，一份一间隔
	int init_posi = blockIdx.x * (group_interval * ISLAND_NUM);//有双倍，分时进行
	int indiv_num = indiv_perblock;

	while (tid < indiv_num * dimension) {
		//tid代表的是以维度为连续空间的下标，后续可改进
		indivs_in_s[(tid % dimension) * indiv_perblock + (tid / dimension)] = indivs[init_posi + tid];
		tid += _THREAD_NUM;
	}

	tid = threadIdx.x;
	while (tid < indiv_num) {
		indivs_eval[tid] = eval[blockIdx.x * indiv_perblock * 2 + tid];
		tid += _THREAD_NUM;
	}
	int iter_time = ((INTERVAL_TRANSFER - 1) / LOOPTIME + 1) * ISLAND_NUM;
	int n = 1, n_island = 1;
	
	//创建r数组并设为-1，为后续的同步做准备
	__shared__ float r[LOOPTIME * INDIV_PERBLOCK * 2];//每个个体分到两个变量用于保存r1和r2;
	
	CrossPrep(r, indiv_num, _THREAD_NUM, n, &state, 0);
	
	int interval_migra = LOOPTIME;//数据传输间隙迭代次数

	int migra_time = (INTERVAL_MIGRA / LOOPTIME) * ISLAND_NUM;
	for (int i = 0; i < iter_time; ++i) {
		// if (iter_time - i == 2) {
		// 	interval_migra = (INTERVAL_TRANSFER * 2 - LOOPTIME * i) / (2);
		// }
		//遗传操作和数据加载并行进行，对线程进行划分
			if (threadIdx.x < THREAD_FOR_OPERA) {//前THREAD_FOR_OPERA个线程用来计算
				for (int j = 0; j < interval_migra; ++j) {
					//变异
					tid = threadIdx.x;
					
					//交叉,此处以维度为单位
					CrossOver(indiv_num, n, &state, indivs_in_s, range, r + (1 - n) * indiv_perblock * LOOPTIME, indivs, para);
					
					tid = threadIdx.x;
					namedBarrierSync(2, THREAD_FOR_OPERA);

					// Mutation(indiv_num, n, &state, indivs_in_s, range, para);
                    
					// namedBarrierSync(2, THREAD_FOR_OPERA);
					//筛选 + 更新
					evaluation(dis, indivs_in_s + 2 * group_interval, indiv_num, indivs_eval + indiv_perblock * 2, indiv_perblock, para, 3);//1s左右时间
					//变异与交叉需要异步，故此处需要线程同步
					namedBarrierSync(2, THREAD_FOR_OPERA);
					Selection(indiv_num, n, indivs_eval, indivs_in_s, para);
					
				}


			}
			else {//后THREAD_NUM - THREAD_FOR_OPERA个线程用来传输数据
				  //传回global memory

				
				tid = threadIdx.x - THREAD_FOR_OPERA;

				if (i > 0) {
					SharedToGlobal(indivs, indivs_in_s, indivs_eval, eval, &state, shuffle, n, n_island, THREAD_FOR_TRANSFER, para);
				}



                //if(threadIdx.x - THREAD_FOR_OPERA == 0 && blockIdx.x == 0){
                //    printf("here..1\n");
                //}
				//__threadfence();
				tid = threadIdx.x - THREAD_FOR_OPERA;
				if (tid % THREAD_FOR_TRANSFER == 0) {//传送完成，
					atomicAdd((int*)(syncval + i), 1);
					//保证同步,忙等待直到对应的同步锁值更新
				}
				__threadfence();
				namedBarrierSync(1, THREAD_FOR_TRANSFER);
				CrossPrep(r, indiv_num, THREAD_FOR_TRANSFER, 1 - n, &state, THREAD_FOR_OPERA);
                while (*(syncval + i) % gridDim.x != 0) {
                }
				namedBarrierSync(1, THREAD_FOR_TRANSFER);
				SelectFromBlocks(n, n_island, indivs_in_s, indivs_eval, indivs, eval, select_interval, i, para);
				
				
                //if(threadIdx.x - THREAD_FOR_OPERA  == 0 && blockIdx.x == 0){
                //   printf("here..2\n");
                //}
			
			}
			
         
		
		n_island = (n_island + 1) % ISLAND_NUM;
		n = (n + 1) % 2;
		// __threadfence();
		__syncthreads();
		if(i % migra_time < (ISLAND_NUM - 1) && i > (ISLAND_NUM - 1)){
			IslandMigration(idx_forSort, indivs_eval, indivs_in_s, indiv_num, para);	
		}
		// if (i != iter_time - 1 && i % (INTERVAL_MIGRA * ISLAND_NUM) == 0) {
		// 	IslandMigration(idx_forSort, indivs_eval, indivs_in_s, indiv_num);	
		// }
		
		__syncthreads();
		

	}
	// printf("!!!!!!!!!!!!!!!!!!!\n");
	// printf("%d, %d, %d\n", para[0], para[1], para[2]);
	
	SharedToGlobal(indivs, indivs_in_s, indivs_eval, eval, &state, shuffle, n, n_island,  _THREAD_NUM, para);
	states[blockDim.x * blockIdx.x + threadIdx.x] = state;
               // if(threadIdx.x - THREAD_FOR_OPERA == 0 && blockIdx.x == 0){
              //      printf("here..6\n");
               // }
			   
}

__global__ void evaluate(float* indivs_, float* eval, float* dis_cpu, int* type, int* para) {

	int indiv_perblock = INDIV_PERBLOCK * ISLAND_NUM, unit = INDIV_PERBLOCK * ISLAND_NUM;
	int init_indiv = blockIdx.x * indiv_perblock;
	const int dimension = para[0];

	if (blockIdx.x == BLOCK_NUM - 1) {
		indiv_perblock = INDIVNUM - unit * (BLOCK_NUM - 1);
	}
	extern __shared__ float indivs[];//大小不能超出indiv_perblock * DIMENSION
	__shared__ float indivs_val[INDIV_PERBLOCK * ISLAND_NUM];//大小不能超出indiv_perblock * DIMENSION

	//传值到shared memory
	int tid = threadIdx.x;
	int t_n = blockDim.x;
	while (tid < indiv_perblock * dimension) {
		//tid代表的是以维度为连续空间的下标，后续可改进
		indivs[(tid % dimension) * unit + (tid / dimension)] = indivs_[init_indiv * dimension + tid];
		tid += t_n;
	}
	tid = threadIdx.x;
	while (tid < indiv_perblock) {
		indivs_val[tid] = eval[init_indiv + tid];
		tid += t_n;
	}
	tid = blockDim.x - threadIdx.x - 1;
	const int dim = para[1] + 1;
	__syncthreads();

	evaluation(dis_cpu, indivs, indiv_perblock, indivs_val, unit, para, 0);


	__syncthreads();
	//传回global_memory
	tid = threadIdx.x;
	while (tid < indiv_perblock * para[0]) {
		int target_tid = init_indiv * para[0] + tid;
		indivs_[target_tid] = indivs[(tid / para[0]) + (tid % para[0]) * unit];
		tid += t_n;
	}
	tid = threadIdx.x;
	while (tid < indiv_perblock) {
		eval[init_indiv + tid] = indivs_val[tid];
		tid += t_n;
	}
}

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

// __global__ void knowledgeTransfer(curandState* state, int *type, float* M, float* b, DS* indivs_sort_1, DS* indivs_sort_2) {

// 	__shared__ float indivs_self[int((TRANSFER_NUM - 1) / BLOCK_NUM + 1) * DIMENSION];
// 	__shared__ float indivs_target[int((TRANSFER_NUM - 1) / BLOCK_NUM + 1) * DIMENSION];

// 	__shared__ float indivs_eval_t[((TRANSFER_NUM - 1) / BLOCK_NUM + 1)];//保存值


// 	int indiv_perblock = TRANSFER_NUM / BLOCK_NUM;
// 	if(blockIdx.x < TRANSFER_NUM % BLOCK_NUM){
// 		indiv_perblock += 1;
// 	}
// 	float range_1 = Range[type[0] * 2 + 1] - Range[type[0] * 2];
// 	float range_2 = Range[type[1] * 2 + 1] - Range[type[1] * 2];
// 	int tid = threadIdx.x;
// 	int t_n = blockDim.x;
// 	int init_posi = blockIdx.x * indiv_perblock;
// 	//传值到shared memory
// 	while (tid < indiv_perblock * DIMENSION) {
// 		//tid代表的是以维度为连续空间的下标，后续可改进
// 		indivs_self[(tid % DIMENSION) * indiv_perblock + (tid / DIMENSION)] = *(indivs_sort_1[(init_posi + tid / DIMENSION) / BANKSIZE].pointer[(init_posi + tid / DIMENSION) % BANKSIZE] + tid % DIMENSION);
// 		indivs_target[(tid % DIMENSION) * indiv_perblock + (tid / DIMENSION)] = *(indivs_sort_2[(init_posi + tid / DIMENSION) / BANKSIZE].pointer[(init_posi + tid / DIMENSION) % BANKSIZE] + tid % DIMENSION);
		
// 		tid += t_n;
// 	}

// 	tid = threadIdx.x;
// 	__shared__ float M_s[DIMENSION], b_s[DIMENSION];
// 	while (tid < DIMENSION) {
// 		M_s[tid] = M[tid];
// 		b_s[tid] = b[tid];
// 		tid += t_n;
// 	}
// 	__threadfence();
// 	__syncthreads();
// 	//交叉
// 	tid = threadIdx.x;
// 	while (tid < indiv_perblock * DIMENSION) {
// 		float rand_val = curand_uniform(&state[threadIdx.x + blockDim.x * blockIdx.x]);

// 		if (rand_val > CROSSOVER_RATE) {
// 			indivs_self[tid] = Range[type[0] * 2] + (indivs_target[tid] - Range[type[1] * 2]) / range_2 * range_1;
// 		}
// 		tid += t_n;
// 	}
// 	__threadfence();
// 	__syncthreads();
// 	//评估
// 	evaluation(type[0], M_s, b_s, indivs_self, indiv_perblock, indivs_eval_t, indiv_perblock);
// 	__threadfence();
// 	__syncthreads();
// 	//对比，替换
// 	tid = threadIdx.x;
// 	__shared__ int temp[((TRANSFER_NUM - 1) / BLOCK_NUM + 1)];//记录哪些个体可以被替换,并替换eval值
// 	while (tid < indiv_perblock) {
// 		int target_tid = blockIdx.x * indiv_perblock + INDIVNUM - TRANSFER_NUM + tid;
// 		if (indivs_eval_t[tid] < *indivs_sort_1[target_tid / BANKSIZE].eval_pointer[target_tid % BANKSIZE]) {
// 			*indivs_sort_1[target_tid / BANKSIZE].eval_pointer[target_tid % BANKSIZE] = indivs_eval_t[tid];
// 			temp[tid] = 1;
// 		}
// 		else {
// 			temp[tid] = 0;
// 		}
// 		tid += t_n;
// 	}
// 	__threadfence();
// 	__syncthreads();
	
// 	tid = threadIdx.x;
// 	while (tid < indiv_perblock * DIMENSION) {
// 		init_posi = (blockIdx.x * indiv_perblock + INDIVNUM - TRANSFER_NUM);
// 		if (temp[tid / DIMENSION] == 1) {
// 			*(indivs_sort_1[(init_posi + tid / DIMENSION) / BANKSIZE].pointer[(init_posi + tid / DIMENSION) % BANKSIZE] + tid % DIMENSION) = indivs_self[(tid / DIMENSION) + (tid % DIMENSION) * indiv_perblock];
// 		}
// 		tid += t_n;
// 	}
	
// }

__global__ void intervalRand(int* select_interval, curandState* state){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int t_n = blockDim.x * gridDim.x;
	while(tid < SELECT_INTERVAL){
		select_interval[tid] = curand(&state[tid]) % BLOCK_NUM;
		tid += t_n;
	}
}
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

struct TaskMessage{
	int cur_iter;
	queue<int> waiting_line;
	cudaEvent_t event, event_1;
	int record_1 = 0;
	pthread_mutex_t task_mutex=PTHREAD_MUTEX_INITIALIZER;
	// mutex task_mutex;
	pthread_cond_t cond;
	TaskMessage(int cur_iter = 0){
		this->cur_iter = cur_iter;	
		
		cudaEventCreate(&event);
		cudaEventCreate(&event_1);
		pthread_cond_init(&cond, nullptr);
	}
};
map<int, TaskMessage*> tasks_set;
vector<int> tasks_vec;
TaskMessage task_ms[T];
double response_time[T], receive_time[T], solve_time[T];

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int last_transfer = transfer_num;
DS record[T][ISLAND_NUM][ITER_NUM / INTERVAL_TRANSFER + 1];


void iter(int task_idx, int cthread_idx , cudaStream_t* streams) {
	cudaEvent_t finish_sign;
	cudaEventCreate(&finish_sign);
	pthread_mutex_lock(&iter_mutex);
	tasks_set[task_idx] = &task_ms[task_idx];
	tasks_vec.push_back(task_idx);
	pthread_mutex_unlock(&iter_mutex);
	
	//任务参数：总维度、车辆数、客户数
	int dimension = rand() % int(DIMENSION / 5) + DIMENSION * 4 / 5;
	int car_num = dimension / 10;//每辆车上限五个客户
	int customer_num = dimension - car_num;
	int cpu_para[3] = {dimension, car_num, customer_num};
	cudaMemcpyAsync(paras[task_idx], cpu_para, sizeof(int) * 3, cudaMemcpyHostToDevice, streams[task_idx % STREAM_NUM]);
	
	// mutex cond_mutex_w, cond_mutex_f;
	// unique_lock<mutex> cond_lock_w(cond_mutex_w);
	// unique_lock<mutex> cond_lock_f(cond_mutex_f);
	int wait_num = 0;
	int blocknum = ISLAND_NUM * VAL_TYPE, threadnum = INDIVNUM_ISLAND * _THREAD_NUM * DIMENSION / SHARED_CAPACITY;
	
	pop_init << <BLOCK_NUM, _THREAD_NUM, 0, streams[task_idx % STREAM_NUM] >> > (devStates+ _THREAD_NUM * BLOCK_NUM * (task_idx % STREAM_NUM), INDIVIDUALS[task_idx], task_type[task_idx], indiv_val[task_idx], paras[task_idx]);
	int memory_cost = 3 * ((int)(WARPSIZE / (2 * INDIV_PERBLOCK) + 1)) * (2 * INDIV_PERBLOCK) * sizeof(float) + INDIV_PERBLOCK * ISLAND_NUM * cpu_para[0] * (sizeof(float) + sizeof(int)) + INDIV_PERBLOCK * ISLAND_NUM * sizeof(int) * 2;
	// printf("memory_cost: %d, %d, %d\n", memory_cost, INDIV_PERBLOCK * ISLAND_NUM * cpu_para[0] * (sizeof(float) + sizeof(int)), cpu_para[0]);
	evaluate<<<BLOCK_NUM, THREAD_FOR_OPERA, memory_cost, streams[task_idx % STREAM_NUM]>>>(INDIVIDUALS[task_idx], indiv_val[task_idx], dis_cpu[task_idx], task_type[task_idx], paras[task_idx]);
	//种群排序
	//种群迭代
		
	// cudaDeviceSynchronize();
	// cudaError_t cudaStatus = cudaGetLastError();
	// if (cudaStatus != cudaSuccess) {
	// 	fprintf(stderr, "\n%s\n", cudaGetErrorString(cudaStatus));
	// 	exit(-1);
	// }
	popSort << <1, _THREAD_NUM, 0, streams[task_idx % STREAM_NUM] >> >(indiv_sort[task_idx], indiv_val[task_idx], INDIVIDUALS[task_idx], task_type[task_idx]);
	
	int iter_time = ITER_NUM / INTERVAL_TRANSFER;
	for (int i = 0; i < iter_time; ++i) {
		//进行遗传操作以及适应度评估（为了最大化shared memory效用，此处进行INTERVAL_TRANSFER次迭代）
		//cudaLaunchCooperativeKernel((void*)GeneticOpera, dimGrid, dimBlock, kernelArgs, 3 * ((int)(WARPSIZE / INDIV_PERBLOCK + 1)) * INDIV_PERBLOCK * sizeof(float), streams[task_idx % STREAM_NUM]);
		
		intervalRand<<<3, _THREAD_NUM, 0, streams[task_idx % STREAM_NUM]>>>(select_interval[task_idx], devStates+ _THREAD_NUM * BLOCK_NUM * (task_idx % STREAM_NUM));
		// printf("-------------iter_time: %d\n", i);
		GeneticOpera << <BLOCK_NUM, _THREAD_NUM, 3 * ((int)(WARPSIZE / INDIV_PERBLOCK + 1)) * INDIV_PERBLOCK * sizeof(float) + INDIV_PERBLOCK * 3 * cpu_para[0] * sizeof(float) + INDIV_PERBLOCK * cpu_para[0] * sizeof(int) + INDIV_PERBLOCK * ISLAND_NUM * sizeof(int), streams[task_idx % STREAM_NUM]  >> > (INDIVIDUALS[task_idx], (volatile int*)syncval[task_idx], devStates+ _THREAD_NUM * BLOCK_NUM * (task_idx % STREAM_NUM), task_type[task_idx], dis_cpu[task_idx], indiv_val[task_idx], select_interval[task_idx], paras[task_idx]);
		
		// cudaDeviceSynchronize();
		// cudaError_t cudaStatus = cudaGetLastError();
		// if (cudaStatus != cudaSuccess) {
		// 	fprintf(stderr, "\n%s\n", cudaGetErrorString(cudaStatus));
		// 	exit(-1);
		// }
		for(int j = 0; j < wait_num; ++j){
			int target_id = tasks_set[task_idx]->waiting_line.front();
			pthread_mutex_lock(&iter_mutex);
			tasks_set[task_idx]->waiting_line.pop();
			pthread_mutex_unlock(&iter_mutex);
			while(tasks_set[target_id]->record_1 == 0){
			}
			cudaEventSynchronize(tasks_set[target_id]->event_1);
			pthread_mutex_lock(&tasks_set[target_id]->task_mutex);
			tasks_set[target_id]->record_1 -= 1;
			pthread_mutex_unlock(&tasks_set[target_id]->task_mutex);
		}
		popSort << <1, _THREAD_NUM, 0, streams[task_idx % STREAM_NUM] >> >(indiv_sort[task_idx], indiv_val[task_idx], INDIVIDUALS[task_idx], task_type[task_idx]);
		
		cudaEventRecord(tasks_set[task_idx]->event, streams[task_idx % STREAM_NUM]);
		
		if(i == max_iter[task_idx]){
			pthread_mutex_lock(&iter_mutex);
			int target = tasks_vec[rand() % tasks_vec.size()];
			while((target == task_idx || !(tasks_set[target]->cur_iter < iter_time)) && tasks_vec.size() > 1){
				target = tasks_vec[rand() % tasks_vec.size()];
			}
			cudaMemcpyAsync(&task_type[task_idx][1], &tasks_type[target], sizeof(int), cudaMemcpyHostToDevice, streams[task_idx % STREAM_NUM]);
			tasks_set[target]->waiting_line.push(task_idx);
			pthread_cond_broadcast(&tasks_set[task_idx]->cond);
			wait_num = tasks_set[task_idx]->waiting_line.size();
			tasks_set[task_idx]->cur_iter += 1;
			if(target != task_idx && !(tasks_set[target]->cur_iter < iter_time)){
				pthread_cond_wait(&tasks_set[target]->cond, &iter_mutex);
			}
			
			pthread_mutex_unlock(&iter_mutex);
			int memory_cost = 3 * ((int)(WARPSIZE / (TRANSFER_NUM / BLOCK_NUM) + 1)) * (TRANSFER_NUM / BLOCK_NUM) * sizeof(float);

			cudaEventSynchronize(tasks_set[target]->event);
			
			// knowledgeTransfer << < BLOCK_NUM, THREAD_FOR_OPERA, memory_cost, streams[task_idx % STREAM_NUM]>> > (devStates+ _THREAD_NUM * BLOCK_NUM * task_idx % STREAM_NUM, task_type[task_idx], M_cpu[task_idx], b_cpu[task_idx], indiv_sort[task_idx], indiv_sort[target]);
			
			cudaEventRecord(tasks_set[task_idx]->event_1, streams[task_idx % STREAM_NUM]);
			pthread_mutex_lock(&tasks_set[task_idx]->task_mutex);
			tasks_set[task_idx]->record_1 += 1;
			pthread_mutex_unlock(&tasks_set[task_idx]->task_mutex);

			max_iter[task_idx] += 1;
		}
	}
	
	// iter_mutex.lock();
	// transfer_num -= 1;
	// if (transfer_num == waiting_num) {
	// 	wait_line.notify_all();
	// }
	// iter_mutex.unlock();

	pthread_mutex_lock(&iter_mutex);
	pthread_cond_broadcast(&tasks_set[task_idx]->cond);
	// tasks_set.erase(task_idx);
	pthread_mutex_unlock(&iter_mutex);
	cudaEventRecord(finish_sign, streams[task_idx % STREAM_NUM]);

	cudaEventSynchronize(finish_sign);

	solve_time[task_idx] = cpuSecond();
	pthread_mutex_lock(&iter_mutex);
	tasks_vec.erase(find(tasks_vec.begin(), tasks_vec.end(), task_idx));
	pthread_mutex_unlock(&iter_mutex);

	// popTransfer_(task_idx, &streams[task_idx % STREAM_NUM]);
	//messageTransfer<<<BLOCK_NUM, _THREAD_NUM, 0, streams[task_idx % STREAM_NUM]>>>(INDIV_BEST_GPU[task_idx], INDIVVAL_BEST_GPU[task_idx], indiv_sort[task_idx]);
}



unsigned int getMemoryUsage() {
  std::ifstream statm("/proc/self/statm");
  unsigned int physicalMem = 0;
  statm >> physicalMem;
  statm.close();
  return physicalMem * 4;
}

int GetSysMemInfo() {  //获取系统当前可用内存
        int mem_free = -1;//空闲的内存，=总内存-使用了的内存
        int mem_total = -1; //当前系统可用总内存
        int mem_buffers = -1;//缓存区的内存大小
        int mem_cached = -1;//缓存区的内存大小
        char name[20];
 
        FILE *fp;
        char buf1[128], buf2[128], buf3[128], buf4[128], buf5[128];
        int buff_len = 128;
        fp = fopen("/proc/meminfo", "r");
        if (fp == NULL) {
            std::cerr << "GetSysMemInfo() error! file not exist" << std::endl;
            return -1;
        }
        if (NULL == fgets(buf1, buff_len, fp) ||
            NULL == fgets(buf2, buff_len, fp) ||
            NULL == fgets(buf3, buff_len, fp) ||
            NULL == fgets(buf4, buff_len, fp) ||
            NULL == fgets(buf5, buff_len, fp)) {
            std::cerr << "GetSysMemInfo() error! fail to read!" << std::endl;
            fclose(fp);
            return -1;
        }
        fclose(fp);
        sscanf(buf1, "%s%d", name, &mem_total);
        sscanf(buf2, "%s%d", name, &mem_free);
        sscanf(buf4, "%s%d", name, &mem_buffers);
        sscanf(buf5, "%s%d", name, &mem_cached);
        int memLeft = mem_free + mem_buffers + mem_cached;
        return mem_total - mem_free;
}

int main()
{
	double avg_mem_used = 0, max_mem_used = 0;
	double already_used = (32488476 - 4600288) / pow(2, 10);
	double init_mem = (double(GetSysMemInfo()) / pow(2, 10)) - already_used;

	printf("InitMemoryUsage: %lf\n", double(GetSysMemInfo()) / pow(2, 20));
	int dev = 0;
	int supportsCoopLaunch = 0;
	//printf("count：%d\n", pv.getCount());
	
	cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
	printf("> %d\n", supportsCoopLaunch);
	
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("> Using Device %d: %s\n", dev, deviceProp.name);
	cudaSetDevice(dev);

	// check if device support hyper-q
	
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

	//创建并发流
	cudaStream_t stream[STREAM_NUM];
	for (int i = 0; i < STREAM_NUM; ++i) {
		cudaStreamCreate(&stream[i]);
	}
	

	//线程开始迭代
	mutex t_mutex;
	//pthread_t threads[CPU_THREAD_NUM];
    thread threads[CPU_THREAD_NUM];
	
	//寻找最佳迭代时间
	tasks_split[0] = 0;
	tasks_split[BATCH_NUM] = T;
	getRandNum(tasks_split + 1, BATCH_NUM - 1, 0, T);
	sort(tasks_split, tasks_split + BATCH_NUM);

	for(int i = 0; i < BATCH_NUM; ++i){
		st_time[i] = getRandFloat(0.f, 1000.f);//0~1000ms等待时间
	}

	//时间记录
	double time = 0;
	for(int i = 0; i < BATCH_NUM; ++i){
		
		int task_num = tasks_split[i + 1] - tasks_split[i];
		int task_init = tasks_split[i];
		for(int j = 0; j < task_num; ++j){
			receive_time[task_init + j] = time;
		}
		time += st_time[i];
		st_time[i] = time;
	}


	cudaEvent_t time_start, time_stop;
	cudaEventCreate(&time_start);
	cudaEventCreate(&time_stop);
	cudaEventRecord(time_start, 0);
	cudaEventSynchronize(time_start);
	double t_start = cpuSecond();


	initialization();

	cudaEventRecord(time_stop, 0);
	cudaEventSynchronize(time_stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, time_start, time_stop);
	printf("\nruntime=%f ms, %f, %f \n", elapsedTime, time_start, time_stop);

	int task_idx = 0;
	// unordered_map<int, TaskMessage> tasks_set;

	double task_start = cpuSecond();
	for(int i = 0; i < BATCH_NUM; ++i){
		
		int task_num = (i == BATCH_NUM - 1) ? (T - tasks_split[i]) : (tasks_split[i + 1] - tasks_split[i]);
		int task_init = tasks_split[i];
		printf("task_init: %d, task_num: %d\n", task_init, task_num);
		// double iStart = cpuSecond();
		int j_end = i == BATCH_NUM - 1 ? T : tasks_split[i + 1];
		for(int j = tasks_split[i]; j < j_end; ++j){
			response_time[task_idx] = cpuSecond();
			threads[j] = thread(iter, task_idx, j, stream);
			t_mutex.lock();
			task_idx += 1;
			t_mutex.unlock();
			
			// threads[j].detach();
		}
		double iElaps = cpuSecond() - task_start;
		if(BATCH_NUM > 1 && iElaps < st_time[i] / 1000.f){
			// sleep(st_time[i] / 1000.f - iElaps);
			this_thread::sleep_for(chrono::milliseconds(long(st_time[i] - iElaps * 1000.f)));
			printf("sleep time: %f\n", st_time[i] / 1000.f - iElaps);
		}
		printf("MemoryUsage: %lf\n", (double(GetSysMemInfo()) / pow(2, 10)) - already_used);
		if(max_mem_used < (double(GetSysMemInfo()) / pow(2, 10)) - already_used){
			max_mem_used = (double(GetSysMemInfo()) / pow(2, 10)) - already_used;
		}
		avg_mem_used += (double(GetSysMemInfo()) / pow(2, 10)) - already_used;
	}
	for (int i = 0; i < CPU_THREAD_NUM; ++i) {
		threads[i].join();
	}
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\n%s\n", cudaGetErrorString(cudaStatus));
	}

	// for (int i = 0; i < CPU_THREAD_NUM; ++i) {
    //     /*
	// 	args.task_idx = task_idx;
	// 	args.cthread_idx = i;
	// 	args.type = tasks_type[i];
    //     */
	// 	threads[i] = thread(iter, task_idx, i, stream);
	// 	t_mutex.lock();
	// 	task_idx += 1;
	// 	t_mutex.unlock();
	// }
	// for (int i = 0; i < CPU_THREAD_NUM; ++i) {
	// 	threads[i].join();
	// }
	cudaDeviceSynchronize();
	cudaEventRecord(time_stop, 0);
	cudaEventSynchronize(time_stop);
	elapsedTime;
	cudaEventElapsedTime(&elapsedTime, time_start, time_stop);
	printf("\nruntime=%f ms, %f, %f, %f \n", elapsedTime, time_start, time_stop, cpuSecond() - t_start);
    popTransfer();
	int task_debug = -1;
	for (int i = 0; i < T; ++i) {
		if (tasks_type[i] == 6) {
			task_debug = i;
			cudaDeviceSynchronize();
			break;
		}
	}
	double avg_rs_time = 0, avg_sl_time = 0, best = 1000000000000, worst = 0;
	for(int i = 0; i < T; ++i){
		avg_rs_time += response_time[i] * 1000.f - receive_time[i] - task_start * 1000.f;
		avg_sl_time += solve_time[i] * 1000.f - receive_time[i] - task_start * 1000.f;
		if (solve_time[i] * 1000.f - receive_time[i] - task_start * 1000.f < best){
			best = solve_time[i] * 1000.f - receive_time[i] - task_start * 1000.f;
		}
		if(solve_time[i] * 1000.f - receive_time[i] - task_start * 1000.f > worst){
			worst = solve_time[i] * 1000.f - receive_time[i] - task_start * 1000.f;
			// printf("worst....%d, %lf, %lf, %lf\n", i, worst, solve_time[i] * 1000.f - task_start * 1000.f, receive_time[i]);
		}
	}
	avg_rs_time /= T;
	avg_sl_time /= T;
	double std_ = 0;
	for(int i = 0; i < T; ++i){
		std_ += (solve_time[i] * 1000.f - receive_time[i] - task_start * 1000.f - avg_sl_time) * (solve_time[i] * 1000.f - receive_time[i] - task_start * 1000.f - avg_sl_time);
	}
	std_ = sqrt(std_ / T);
	printf("avg_rs_time: %lf ms, avg_sl_time: %lf ms, avg_mem_used: %lf, max_mem_used: %lf, init_mem_used: %lf\n", avg_rs_time, avg_sl_time, avg_mem_used / BATCH_NUM, max_mem_used, init_mem);
	printf("std_: %lf ms, best: %lf ms, worst: %lf ms\n", std_, best, worst);
	//===================================parameter discussion:time
	// char* T_num = new char[5], *DIMEN = new char[5];
	// sprintf(DIMEN,"%d",DIMENSION);
	// ofstream file;
	// file.open("time_dimension_self.txt", ios::app);
	// file << elapsedTime << ", " << "#" << DIMEN;
	// file << endl;
	// file.close();
	//===================================

	//==================================
	for(int i = 0; i < T; ++i){
		sort(INDIV_VAL_CPU[i], INDIV_VAL_CPU[i] + INDIVNUM);
	}
	ofstream file;
	file.open("output_0.txt", ios::app);
	for(int i = 0; i < T; ++i){
		for(int j = 0; j < INDIVNUM; ++j){
			file << INDIV_VAL_CPU[i][j] << ' ';
		}
		file << endl;
	}
	file << endl;
	file.close();
	//==================================
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
#include <vector>
#include <iostream>

#include "cuda_error_check.cuh"
#include "utils.cuh"

__global__ void swap_kernel(int* a, int* b, int n){
	const int tid = threadIdx.x + blockDim.x*blockIdx.x;
	const int nThreads = blockDim.x*gridDim.x;
	const int iter = n%nThreads == 0? n/nThreads : n/nThreads+1;

	for(int i = 0; i < iter; i++){
		int id = tid + i*nThreads;
		if(id < n){
			int temp = a[id];
			a[id] = b[id];
			b[id] = temp;
		}
	}
}

__global__ void impl1_incore_kernel(edge* edges, int nEdges, int* distance, int* anyChange){
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	const int nThreads = blockDim.x * gridDim.x;
	const int iter = nEdges%nThreads == 0? nEdges/nThreads : nEdges/nThreads+1;	

	for(int i = 0; i < iter; i++){
		int id = tid + i*nThreads;
		if(id < nEdges){
			int u = edges[id].src;
			int v = edges[id].dest;
			int w = edges[id].w;
			int temp_dist = distance[u]+w;
			if(distance[u] == INF) continue;
			if(temp_dist < distance[v]){
				atomicMin(&distance[v], temp_dist);
				*anyChange = 1;
			}
		}
	}
}

__global__ void impl1_outcore_kernel(edge* edges, int nEdges, int* distance_cur, int* distance_prev, int* anyChange){
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	const int nThreads = blockDim.x * gridDim.x;
	const int iter = nEdges%nThreads == 0? nEdges/nThreads : nEdges/nThreads+1;	

	for(int i = 0; i < iter; i++){
		int id = tid + i*nThreads;
		if(id < nEdges){
			int u = edges[id].src;
			int v = edges[id].dest;
			int w = edges[id].w;
			if(distance_prev[u] == INF) continue;
			if(distance_prev[u]+w < distance_cur[v]){
				atomicMin(&distance_cur[v], distance_prev[u]+w);
				*anyChange = 1;
			}
		}
	}
}

__device__ void segmented_scan_min_kernel(int* result, int* node){
	const int tid = threadIdx.x;
	const int lane = tid & 31;
	if(lane >= 1 && node[tid] == node[tid - 1]) 
		result[tid] = min(result[tid],result[tid-1]);
	if(lane >= 2 && node[tid] == node[tid - 2])
		result[tid] = min(result[tid],result[tid-2]);
	if(lane >= 4 && node[tid] == node[tid - 4])
		result[tid] = min(result[tid],result[tid-4]);
	if(lane >= 8 && node[tid] == node[tid - 8])
		result[tid] = min(result[tid],result[tid-8]);
	if(lane >= 16 && node[tid] == node[tid - 16])
		result[tid] = min(result[tid],result[tid-16]);
}

__global__ void impl1_outcore_shmem_kernel(edge* edges, int nEdges, int* distance_cur, int* distance_prev, int* anyChange){
        const int tid = blockDim.x*blockIdx.x + threadIdx.x;
        const int nThreads = blockDim.x * gridDim.x;
	const int lane = tid & 31;
	const int iter = nEdges%nThreads == 0? nEdges/nThreads : nEdges/nThreads+1;

	__shared__ int s_node[1024]; //index of each destination node
	__shared__ int s_distance[1024]; //distance of each dest node

	for(int i = 0; i < iter; i++){
		int id = tid + i*nThreads;
		if(id < nEdges){
			int u = edges[id].src;
			int v = edges[id].dest;
			int w = edges[id].w;
			s_node[threadIdx.x] = v;
			if(distance_prev[u] == INF)
				s_distance[threadIdx.x] = INF;
			else
				s_distance[threadIdx.x] = distance_prev[u]+w;
			__syncthreads();

			// get the min distance for dest node
			segmented_scan_min_kernel(s_distance,s_node);
			if(lane == 31 || s_node[threadIdx.x] != s_node[threadIdx.x+1]){
				int result = s_distance[threadIdx.x];
				if(result < distance_cur[v]){
					atomicMin(&distance_cur[v],result);
					*anyChange = 1;
				}
			}
		}
	}
}

void impl1_incore(int* results, edge* h_edges, int nEdges, int n, int blockSize, int blockNum, char* deviceName){
	int nb = n*sizeof(int);
	int* d_anyChange = NULL;
	edge* d_edges = NULL;
	int* d_distance = NULL;
	cudaMalloc((void**)&d_edges,nEdges*sizeof(edge));
	cudaMalloc((void**)&d_distance,nb);
	cudaMalloc((void**)&d_anyChange,sizeof(int));
	cudaMemcpy(d_edges,h_edges,nEdges*sizeof(edge),cudaMemcpyHostToDevice);
	cudaMemcpy(d_distance,results,nb,cudaMemcpyHostToDevice);

	int nIter = 0;
	Timer timer;
	double time = 0.0; 
	for(int i = 0; i < n-1; i++){
		nIter++;
		cudaMemset(d_anyChange, 0,sizeof(int));
		timer.set();
		impl1_incore_kernel<<<blockNum,blockSize>>>(d_edges,nEdges,d_distance,d_anyChange);
		cudaDeviceSynchronize();
		time += timer.get();
	
		//break from loop if no changes
		int anyChange = 0;
		cudaMemcpy(&anyChange,d_anyChange, sizeof(int),cudaMemcpyDeviceToHost);
		if(!anyChange) break;
	}
	cout << "The total computation kernel time on GPU " << deviceName << " is " << time << " milli-seconds\n";
	cout << "Number of iterations: " << nIter << ", average computation time: " << (time/nIter) << " milli-seconds\n";
	
	cudaMemcpy(results,d_distance,nb,cudaMemcpyDeviceToHost);
	
	cudaFree(d_edges);
	cudaFree(d_distance);
	cudaFree(d_anyChange);
}

void impl1_outcore(int* distance, edge* h_edges, int nEdges, int n, int blockSize, int blockNum, bool useShmem, char* deviceName){
	int nb = n*sizeof(int);
	int* d_anyChange = NULL;
	edge* d_edges = NULL;
	int* d_distance_cur = NULL;
	int* d_distance_prev = NULL;
	cudaMalloc((void**)&d_edges,nEdges*sizeof(edge));
	cudaMalloc((void**)&d_distance_cur,nb);
	cudaMalloc((void**)&d_distance_prev,nb);
	cudaMalloc((void**)&d_anyChange,sizeof(int));
	cudaMemcpy(d_edges,h_edges,nEdges*sizeof(edge),cudaMemcpyHostToDevice);
	cudaMemcpy(d_distance_cur,distance,nb,cudaMemcpyHostToDevice);
	cudaMemcpy(d_distance_prev,distance,nb,cudaMemcpyHostToDevice);

	int nIter = 0;
    	Timer timer;
	double time = 0.0;
	for(int i = 0; i < n-1; i++){
		nIter++;
		cudaMemset(d_anyChange, 0,sizeof(int));
		timer.set();
		if(useShmem)
			impl1_outcore_shmem_kernel<<<blockNum,blockSize>>>(d_edges,nEdges,d_distance_cur,d_distance_prev,d_anyChange);
		else
			impl1_outcore_kernel<<<blockNum,blockSize>>>(d_edges,nEdges,d_distance_cur,d_distance_prev,d_anyChange);
		cudaDeviceSynchronize();
		time += timer.get();

		//break from loop if no changes
		int anyChange = 0;
		cudaMemcpy(&anyChange,d_anyChange, sizeof(int),cudaMemcpyDeviceToHost);
		if(!anyChange) break;
		else swap_kernel<<<blockNum,blockSize>>>(d_distance_prev,d_distance_cur,n);
	}
	cout << "The total computation kernel time on GPU " << deviceName << " is " << time << " milli-seconds\n";
	cout << "Number of iterations: " << nIter << ", average computation time: " << (time/nIter) << " milli-seconds\n";
	
	cudaMemcpy(distance,d_distance_cur,nb,cudaMemcpyDeviceToHost);
	
	cudaFree(d_edges);
	cudaFree(d_distance_prev);
	cudaFree(d_distance_cur);
	cudaFree(d_anyChange);
}

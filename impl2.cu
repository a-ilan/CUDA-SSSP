#include <vector>
#include <iostream>

#include "cuda_error_check.cuh"
#include "utils.cuh"

__device__ int d_nEdges; //number of edges left after filter

__global__ void copy_kernel(int* a, int* b, int n){
	const int tid = threadIdx.x + blockDim.x*blockIdx.x;
	const int nThreads = blockDim.x*gridDim.x;
	const int iter = n%nThreads == 0? n/nThreads : n/nThreads+1;

	for(int i = 0; i < iter; i++){
		int id = tid + i*nThreads;
		if(id < n){
			a[id] = b[id];
		}
	}
}

__global__ void bellmanford_incore_kernel(edge* edges, int* distance,int* changed){
	const int nEdges = d_nEdges;
	const int tid = threadIdx.x + blockDim.x*blockIdx.x;
	const int nThreads = blockDim.x*gridDim.x;
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
				changed[v] = 1;
			}
		}
	}
}

__global__ void bellmanford_outcore_kernel(edge* edges, int* distance_cur,int* distance_prev,int* changed){
        const int nEdges = d_nEdges;
        const int tid = blockDim.x*blockIdx.x + threadIdx.x;
        const int nThreads = blockDim.x*gridDim.x;
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
				changed[v] = 1;
			}
		}
	}
}

__global__ void warp_count_kernel(int* warp_count,edge* edges, int* changed, int nEdges){
	const int tid = threadIdx.x + blockDim.x*blockIdx.x;
	const int nThreads = blockDim.x*gridDim.x;
	const int lane = tid & 31;
	const int wid = tid >> 5;
	const int iter = nEdges%nThreads == 0? nEdges/nThreads : nEdges/nThreads+1;	
	
	for(int i = 0; i < iter; i++){
		int id = tid + i*nThreads;
		if(id < nEdges){
			int u = edges[id].src;
			//int v = edges[id].dest;
			int change = changed[u];

			int mask = __ballot(change);
			int count = __popc(mask);
	
			if(lane==0){
				atomicAdd(&warp_count[wid],count);
			}
		}
	}
}

__device__ int scan_warp_kernel(int* a){
	const int tid = threadIdx.x;
	const int lane = tid & 31;
	 if(lane >= 1) a[tid] += a[tid-1];
	 if(lane >= 2) a[tid] += a[tid-2];
	 if(lane >= 4) a[tid] += a[tid-4];
	 if(lane >= 8) a[tid] += a[tid-8];
	 if(lane >= 16) a[tid] += a[tid-16];
	return lane > 0 ? a[tid-1] : 0;
}

__global__ void scan_block_kernel(int* prefix_sum, int* warp_count){
	const int tid = threadIdx.x;
	const int wid = tid >> 5;
	const int lane = tid & 31;
	prefix_sum[tid] = warp_count[tid];
	__syncthreads();

	int val = scan_warp_kernel(prefix_sum);
	if(lane == 31) prefix_sum[wid] = prefix_sum[tid];
	__syncthreads();

	if(wid == 0) scan_warp_kernel(prefix_sum);
	__syncthreads();

	if(wid > 0) val += prefix_sum[wid-1];
	__syncthreads();

	prefix_sum[tid] = val;
}

__global__ void filter_edges_kernel(edge* filtered_edges, edge* all_edges, int* prefix_sum, int* warp_count, int* changed, int nEdges){
	const int tid = threadIdx.x + blockDim.x*blockIdx.x;
	const int nThreads = blockDim.x*gridDim.x;
	const int nWarps = nThreads%32 == 0? nThreads/32 : nThreads/32+1;
	const int wid = tid >> 5;
	const int lane = tid & 31;
	const int iter = nEdges%nThreads == 0? nEdges/nThreads : nEdges/nThreads+1;	
	int offset = prefix_sum[wid];

	//get the number of warps in the filtered list of edges
	d_nEdges = prefix_sum[nWarps-1]+warp_count[nWarps-1];

	for(int i = 0; i < iter; i++){
		int id = tid + i*nThreads;

		if(id < nEdges){
			int u = all_edges[id].src;
                	int v = all_edges[id].dest;
                	int w = all_edges[id].w;
	
			int change = changed[u];
			int mask = __ballot(change);
			int localid = __popc(mask<<(32-lane));
			
			if(change){
				filtered_edges[offset+localid].src = u;
				filtered_edges[offset+localid].dest = v;
				filtered_edges[offset+localid].w = w;
			}
			offset += __popc(mask);
		}
	}
}

void impl2_incore(int* distance, edge* edges, int nEdges, int n, int blockSize, int blockNum, char* deviceName){
	int nb = n*sizeof(int);
	int nThreads = blockSize*blockNum;
	int nWarps = nThreads%32 == 0? nThreads/32 : nThreads/32+1;

	edge* d_edges = NULL;
	edge* d_filtered_edges = NULL;
	int* d_distance = NULL;
	int* d_warp_count = NULL;
	int* d_prefix_sum = NULL;
	int* d_changed = NULL;
	cudaMalloc((void**)&d_edges,nEdges*sizeof(edge));
	cudaMalloc((void**)&d_filtered_edges,nEdges*sizeof(edge));
	cudaMalloc((void**)&d_distance,nb);
	cudaMalloc((void**)&d_warp_count,nWarps*sizeof(int));
	cudaMalloc((void**)&d_prefix_sum,nWarps*sizeof(int));
	cudaMalloc((void**)&d_changed,nb);
	cudaMemcpy(d_edges,edges,nEdges*sizeof(edge),cudaMemcpyHostToDevice);
	cudaMemcpy(d_filtered_edges,edges,nEdges*sizeof(edge),cudaMemcpyHostToDevice);
	cudaMemcpy(d_distance,distance,nb,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_nEdges,&nEdges,sizeof(int));	
	
	int nIter = 0;
	Timer timer;
	double computation_time = 0.0;
	double filtering_time = 0.0;
	for(int i = 0; i < n-1; i++){
		nIter++;
		cudaMemset(d_warp_count,0,nWarps*sizeof(int));
		cudaMemset(d_changed,0,nb);
		
		//process stage
		timer.set();
		bellmanford_incore_kernel<<<blockNum,blockSize>>>(d_filtered_edges,d_distance,d_changed);
		cudaDeviceSynchronize();
		computation_time += timer.get();

		//filter stage
		timer.set();
		warp_count_kernel<<<blockNum,blockSize>>>(d_warp_count,d_edges,d_changed, nEdges);
		cudaDeviceSynchronize();
		scan_block_kernel<<<1,nWarps>>>(d_prefix_sum,d_warp_count);
		cudaDeviceSynchronize();
		filter_edges_kernel<<<blockNum,blockSize>>>(d_filtered_edges,d_edges,d_prefix_sum,d_warp_count,d_changed,nEdges);
		cudaDeviceSynchronize();
		filtering_time += timer.get();
	
		//check if there is any edges left to process
		int left = 0;
		cudaMemcpyFromSymbol(&left,d_nEdges,sizeof(int));	
		if(left == 0) break;
	}
	cout << "The total computation kernel time on GPU " << deviceName << " is " << computation_time << " milli-seconds\n";
	cout << "The total filtering kernel time on GPU " << deviceName << " is " << filtering_time << " milli-seconds\n";
	cout << "Number of iterations: " << nIter << ", average computation time: " << (computation_time/nIter) << " milli-seconds\n";

	cudaMemcpy(distance,d_distance,nb,cudaMemcpyDeviceToHost);
	
	cudaFree(d_edges);
	cudaFree(d_filtered_edges);
	cudaFree(d_distance);
	cudaFree(d_warp_count);
	cudaFree(d_prefix_sum);
	cudaFree(d_changed);
}

void impl2_outcore(int* distance, edge* edges, int nEdges, int n, int blockSize, int blockNum, char* deviceName){
	int nb = n*sizeof(int);
	int nThreads = blockSize*blockNum;
	int nWarps = nThreads%32 == 0? nThreads/32 : nThreads/32+1;

	edge* d_edges = NULL;
	edge* d_filtered_edges = NULL;
	int* d_distance_cur = NULL;
	int* d_distance_prev = NULL;
	int* d_warp_count = NULL;
	int* d_prefix_sum = NULL;
	int* d_changed = NULL;
	cudaMalloc((void**)&d_edges,nEdges*sizeof(edge));
	cudaMalloc((void**)&d_filtered_edges,nEdges*sizeof(edge));
	cudaMalloc((void**)&d_distance_cur,nb);
	cudaMalloc((void**)&d_distance_prev,nb);
	cudaMalloc((void**)&d_warp_count,nWarps*sizeof(int));
	cudaMalloc((void**)&d_prefix_sum,nWarps*sizeof(int));
	cudaMalloc((void**)&d_changed,nb);
	cudaMemcpy(d_edges,edges,nEdges*sizeof(edge),cudaMemcpyHostToDevice);
	cudaMemcpy(d_filtered_edges,edges,nEdges*sizeof(edge),cudaMemcpyHostToDevice);
	cudaMemcpy(d_distance_cur,distance,nb,cudaMemcpyHostToDevice);
	cudaMemcpy(d_distance_prev,distance,nb,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_nEdges,&nEdges,sizeof(int));

	int nIter = 0;
	Timer timer;
	double computation_time = 0.0;
	double filtering_time = 0.0;
	for(int i = 0; i < n-1; i++){
		nIter++;
		cudaMemset(d_warp_count,0,nWarps*sizeof(int));
		cudaMemset(d_changed,0,nb);

		//process stage
		timer.set();
		bellmanford_outcore_kernel<<<blockNum,blockSize>>>(d_filtered_edges,d_distance_cur,d_distance_prev,d_changed);
		cudaDeviceSynchronize();
		computation_time += timer.get();

		//filter stage
		timer.set();
		warp_count_kernel<<<blockNum,blockSize>>>(d_warp_count,d_edges,d_changed, nEdges);
		cudaDeviceSynchronize();
		scan_block_kernel<<<1,nWarps>>>(d_prefix_sum,d_warp_count);
		cudaDeviceSynchronize();
		filter_edges_kernel<<<blockNum,blockSize>>>(d_filtered_edges,d_edges,d_prefix_sum,d_warp_count,d_changed,nEdges);
		cudaDeviceSynchronize();
		filtering_time += timer.get();

		//check if there is any edges left to process
		int left = 0;
		cudaMemcpyFromSymbol(&left,d_nEdges,sizeof(int));
		if(left == 0) break;
		else copy_kernel<<<blockNum,blockSize>>>(d_distance_prev,d_distance_cur,n);
	}
	cout << "The total computation kernel time on GPU " << deviceName << " is " << computation_time << " milli-seconds\n";
	cout << "The total filtering kernel time on GPU " << deviceName << " is " << filtering_time << " milli-seconds\n";
	cout << "Number of iterations: " << nIter << ", average computation time: " << (computation_time/nIter) << " milli-seconds\n";

	cudaMemcpy(distance,d_distance_cur,nb,cudaMemcpyDeviceToHost);

	cudaFree(d_edges);
	cudaFree(d_filtered_edges);
	cudaFree(d_distance_cur);
	cudaFree(d_distance_prev);
	cudaFree(d_warp_count);
	cudaFree(d_prefix_sum);
	cudaFree(d_changed);
}

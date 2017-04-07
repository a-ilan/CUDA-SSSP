#include <string>
#include <cstring>
#include <cstdlib>
#include <stdio.h>
#include <iostream>

#include "utils.hpp"

#define SSSP_INF 1073741824

/* sort by destination node */
void sort_by_dest(edge* edges, int nEdges, vector<initial_vertex> * peeps){
        int index = 0;
        for(int i = 0; i <  peeps->size(); i++){
                int nbrs = peeps->at(i).nbrs.size();
                for(int j = 0; j < nbrs; j++){
                        edges[index].src = peeps->at(i).nbrs[j].srcIndex;
                        edges[index].dest = i;
                        edges[index].w = peeps->at(i).nbrs[j].weight;
                        index++;
                }
        }
}

void sort_by_src(edge* edges, int nEdges){
	if(nEdges < 2) return;
	int pivot = edges[nEdges/2].src;

	int i,j;
	for(i=0, j=nEdges-1; ;i++, j--){
		while(edges[i].src < pivot) i++;
		while(edges[j].src > pivot) j--;

		if(i >= j) break;

		//swap i and j
		int temp = edges[i].src;
		edges[i].src = edges[j].src;
		edges[j].src = temp;
		temp = edges[i].dest;
		edges[i].dest = edges[j].dest;
		edges[j].dest = temp;
		temp = edges[i].w;
		edges[i].w = edges[j].w;
		edges[j].w = temp;
	}
	sort_by_src(edges,i);
	sort_by_src(edges+i,nEdges-i);
}

void testCorrectness(edge* edge, int* results, int nEdges, int nNodes){
	cout << "TESTING CORRECTNESS" << endl;
	int* solution = new int[nNodes];
	for (int i = 1; i < nNodes; i++) solution[i] = INF;
	solution[0] = 0;
	int change = 0;
	for (int i = 1; i < nNodes; i++){
		for (int j = 0; j < nEdges; j++){
			int u = edge[j].src;
			int v = edge[j].dest;
			int w = edge[j].w;
			if(solution[u]+w < solution[v]){
				solution[v] = solution[u]+w;
				change = 1;
			}
		}
		if(change == 0) break;
		change = 0;
	}

	int incorrect = 0;
	for(int i = 0; i < nNodes; i++){
		if(results[i] != solution[i]){
			incorrect++;
			cout<<"Correct: "<< solution[i] << " Yours: " << results[i] << endl;
		}
	}
	cout << "Correct: " << (nNodes-incorrect) << " Incorrect: " << incorrect << " Total: " << nNodes << endl;	
			
}

void saveResults(ofstream& outFile, int* solution, int n){
	for(int i = 0; i < n; i++){
		outFile << i << ": " << solution[i] << endl; 
	}
	outFile.close();
}

uint parse_graph(
		std::ifstream& inFile,
		std::vector<initial_vertex>& initGraph,
		const long long arbparam,
		const bool nondirected ) {

	const bool firstColumnSourceIndex = true;

	std::string line;
	char delim[3] = " \t";	//In most benchmarks, the delimiter is usually the space character or the tab character.
	char* pch;
	uint nEdges = 0;

	unsigned int Additionalargc=0;
	char* Additionalargv[ 61 ];

	// Read the input graph line-by-line.
	while( std::getline( inFile, line ) ) {
		if( line[0] < '0' || line[0] > '9' )	// Skipping any line blank or starting with a character rather than a number.
			continue;
		char cstrLine[256];
		std::strcpy( cstrLine, line.c_str() );
		uint firstIndex, secondIndex;

		pch = strtok(cstrLine, delim);
		if( pch != NULL )
			firstIndex = atoi( pch );
		else
			continue;
		pch = strtok( NULL, delim );
		if( pch != NULL )
			secondIndex = atoi( pch );
		else
			continue;

		uint theMax = std::max( firstIndex, secondIndex );
		uint srcVertexIndex = firstColumnSourceIndex ? firstIndex : secondIndex;
		uint dstVertexIndex = firstColumnSourceIndex ? secondIndex : firstIndex;
		if( initGraph.size() <= theMax )
			initGraph.resize(theMax+1);
		{ //This is just a block
		        // Add the neighbor. A neighbor wraps edges
			neighbor nbrToAdd;
			nbrToAdd.srcIndex = srcVertexIndex;

			Additionalargc=0;
			Additionalargv[ Additionalargc ] = strtok( NULL, delim );
			while( Additionalargv[ Additionalargc ] != NULL ){
				Additionalargc++;
				Additionalargv[ Additionalargc ] = strtok( NULL, delim );
			}
			initGraph.at(srcVertexIndex).distance = ( srcVertexIndex != arbparam ) ? SSSP_INF : 0;
			initGraph.at(dstVertexIndex).distance = ( dstVertexIndex != arbparam ) ? SSSP_INF : 0;
			nbrToAdd.weight = ( Additionalargc > 0 ) ? atoi(Additionalargv[0]) : 1;

			initGraph.at(dstVertexIndex).nbrs.push_back( nbrToAdd );
			nEdges++;
		}
		if( nondirected ) {
		        // Add the edge going the other way
			uint tmp = srcVertexIndex;
			srcVertexIndex = dstVertexIndex;
			dstVertexIndex = tmp;
			//swap src and dest and add as before
			
			neighbor nbrToAdd;
			nbrToAdd.srcIndex = srcVertexIndex;

			Additionalargc=0;
			Additionalargv[ Additionalargc ] = strtok( NULL, delim );
			while( Additionalargv[ Additionalargc ] != NULL ){
				Additionalargc++;
				Additionalargv[ Additionalargc ] = strtok( NULL, delim );
			}
			initGraph.at(srcVertexIndex).distance = ( srcVertexIndex != arbparam ) ? SSSP_INF : 0;
			initGraph.at(dstVertexIndex).distance = ( dstVertexIndex != arbparam ) ? SSSP_INF : 0;
			nbrToAdd.weight = ( Additionalargc > 0 ) ? atoi(Additionalargv[0]) : 1;
			initGraph.at(dstVertexIndex).nbrs.push_back( nbrToAdd );
			nEdges++;
		}
	}

	inFile.close();
	return nEdges;
}


struct timeval StartingTime;

void setTime(){
	gettimeofday( &StartingTime, NULL );
}

double getTime(){
	struct timeval PausingTime, ElapsedTime;
	gettimeofday( &PausingTime, NULL );
	timersub(&PausingTime, &StartingTime, &ElapsedTime);
	return ElapsedTime.tv_sec*1000.0+ElapsedTime.tv_usec/1000.0;	// Returning in milliseconds.
}


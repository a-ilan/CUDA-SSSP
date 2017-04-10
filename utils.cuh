#ifndef UTILS_CUH
#define UTILS_CUH

#include <fstream>
#include <vector>
#include <cstdlib>
#include <sys/time.h>

using namespace std;

#define INF 1073741824 //UINT_MAX

struct edge{
        int src;
        int dest;
        int w;
};

class neighbor {
public:
        unsigned int weight;
        unsigned int srcIndex;
};

class initial_vertex {
public:
	unsigned int distance;
	std::vector<neighbor> nbrs;
	initial_vertex():nbrs(0){}
};

class Timer {
	struct timeval startingTime;
public:
	void set();
	double get();
};

uint parse_graph(
	std::ifstream& inFile,
	std::vector<initial_vertex>& initGraph,
	const long long arbparam,
	const bool nondirected );

void sort_by_dest(edge* edges, int nEdges, vector<initial_vertex> * peeps);
void sort_by_src(edge* edges, int nEdges);
void testCorrectness(edge* edges, int* results, int nEdges, int nNodes);
void saveResults(ofstream& outFile, int* solution, int n);

#endif	//	PARSE_GRAPH_HPP

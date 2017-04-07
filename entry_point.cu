#include <cstring>
#include <stdexcept>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "cuda_error_check.cuh"
#include "utils.hpp"

#include "impl2.cu"
#include "impl1.cu"

enum class ProcessingType {Push, Neighbor, Own, Unknown};
enum SyncMode {InCore, OutOfCore};
enum SyncMode syncMethod;
enum SmemMode {UseSmem, UseNoSmem};
enum SmemMode smemMethod;
enum SortMode {sortDest,sortSrc};
enum SortMode sortMethod;

// Open files safely.
template <typename T_file>
void openFileToAccess( T_file& input_file, std::string file_name ) {
	input_file.open( file_name.c_str() );
	if( !input_file )
		throw std::runtime_error( "Failed to open specified file: " + file_name + "\n" );
}


// Execution entry point.
int main( int argc, char** argv )
{

	std::string usage =
		"\tCommand line arguments:\n\
			Input file: E.g., --input in.txt\n\
                        Block size: E.g., --bsize 512\n\
                        Block count: E.g., --bcount 192\n\
                        Output path: E.g., --output output.txt\n\
			Processing method: E.g., --method bmf (bellman-ford), or tpe (to-process-edge)\n\
			Shared memory usage: E.g., --usesmem yes, or no \n\
			Sync method: E.g., --sync incore, or outcore\n\
			sorting: E.g., --sort dest, or src\n";

	try {

		std::ifstream inputFile;
		std::ofstream outputFile;
		int selectedDevice = 0;
		int bsize = 0, bcount = 0;
		long long arbparam = 0;
		bool nonDirectedGraph = false;		// By default, the graph is directed.
		ProcessingType processingMethod = ProcessingType::Unknown;
		syncMethod = OutOfCore;
		smemMethod = UseNoSmem;
		sortMethod = sortDest;


		/********************************
		 * GETTING INPUT PARAMETERS.
		 ********************************/

		for( int iii = 1; iii < argc; ++iii )
			if ( !strcmp(argv[iii], "--method") && iii != argc-1 ) {
				if ( !strcmp(argv[iii+1], "bmf") )
				        processingMethod = ProcessingType::Push;
				else if ( !strcmp(argv[iii+1], "tpe") )
    				        processingMethod = ProcessingType::Neighbor;
				else{
					std::cerr << "\n Un-recognized method parameter value \n\n";
					return( EXIT_FAILURE );
				}   
			}
			else if ( !strcmp(argv[iii], "--sync") && iii != argc-1 ) {
				if ( !strcmp(argv[iii+1], "incore") )
				        syncMethod = InCore;
				else if ( !strcmp(argv[iii+1], "outcore") )
    				        syncMethod = OutOfCore;
				else{
					std::cerr << "\n Un-recognized sync parameter value \n\n";
					return( EXIT_FAILURE );
				}  

			}
			else if ( !strcmp(argv[iii], "--usesmem") && iii != argc-1 ) {
				if ( !strcmp(argv[iii+1], "yes") )
				        smemMethod = UseSmem;
				else if ( !strcmp(argv[iii+1], "no") )
    				        smemMethod = UseNoSmem;
        			else{
					std::cerr << "\n Un-recognized usesmem parameter value \n\n";
					return( EXIT_FAILURE );
				}  
			}
			else if( !strcmp(argv[iii], "--sort") && iii != argc-1 ) {
				if ( !strcmp(argv[iii+1], "dest") )
					sortMethod = sortDest;
				else if(!strcmp(argv[iii+1], "src") )
					sortMethod = sortSrc;
				else{
					std::cerr << "\n Un-recognized sort parameter value \n\n";
					return( EXIT_FAILURE );
				}
			}
			else if( !strcmp( argv[iii], "--input" ) && iii != argc-1 /*is not the last one*/)
				openFileToAccess< std::ifstream >( inputFile, std::string( argv[iii+1] ) );
			else if( !strcmp( argv[iii], "--output" ) && iii != argc-1 /*is not the last one*/)
				openFileToAccess< std::ofstream >( outputFile, std::string( argv[iii+1] ) );
			else if( !strcmp( argv[iii], "--bsize" ) && iii != argc-1 /*is not the last one*/)
				bsize = std::atoi( argv[iii+1] );
			else if( !strcmp( argv[iii], "--bcount" ) && iii != argc-1 /*is not the last one*/)
				bcount = std::atoi( argv[iii+1] );

		if(bsize <= 0 || bcount <= 0){
			std::cerr << "Usage: " << usage;
			throw std::runtime_error("\nAn initialization error happened.\nExiting.");
		}
		if( !inputFile.is_open() || processingMethod == ProcessingType::Unknown ) {
			std::cerr << "Usage: " << usage;
			throw std::runtime_error( "\nAn initialization error happened.\nExiting." );
		}
		if( !outputFile.is_open() )
			openFileToAccess< std::ofstream >( outputFile, "out.txt" );
		CUDAErrorCheck( cudaSetDevice( selectedDevice ) );
		std::cout << "Device with ID " << selectedDevice << " is selected to process the graph.\n";


		/********************************
		 * Read the input graph file.
		 ********************************/

		std::cout << "Collecting the input graph ...\n";
		std::vector<initial_vertex> parsedGraph( 0 );
		uint nEdges = parse_graph(
				inputFile,		// Input file.
				parsedGraph,	// The parsed graph.
				arbparam,
				nonDirectedGraph );		// Arbitrary user-provided parameter.
		std::cout << "Input graph collected with " << parsedGraph.size() << " vertices and " << nEdges << " edges.\n";

		edge* edges = new edge[nEdges];
		sort_by_dest(edges,nEdges,&parsedGraph);
		int nNodes = parsedGraph.size();
		int* results = new int[nNodes];
		std::fill_n(results,nNodes,INF);
		results[0] = 0;

		if(sortMethod == sortSrc){
			sort_by_src(edges,nEdges);
		}

		/********************************
		 * Process the graph.
		 ********************************/
		
		switch(processingMethod){
		case ProcessingType::Push:
		    impl1(results,edges,nEdges,nNodes, bsize, bcount);
		    break;
		case ProcessingType::Neighbor:
		    impl2(results,edges,nEdges,nNodes, bsize, bcount);
		    break;
		default:
		    break;
		}

		testCorrectness(edges, results,nEdges, nNodes);
		saveResults(outputFile,results,nNodes);
		delete[] edges, results;

		/********************************
		 * It's done here.
		 ********************************/

		CUDAErrorCheck( cudaDeviceReset() );
		std::cout << "Done.\n";
		return( EXIT_SUCCESS );

	}
	catch( const std::exception& strException ) {
		std::cerr << strException.what() << "\n";
		return( EXIT_FAILURE );
	}
	catch(...) {
		std::cerr << "An exception has occurred." << std::endl;
		return( EXIT_FAILURE );
	}

}

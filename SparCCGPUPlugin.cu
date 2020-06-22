/*
GPU SPARCC

Authors: Brandom Suarez, Cristina Alonso, Robert Panoff, Yadelis Escobar

Program Description: Find the correlation between sets (cols) across samples (rows).
Uses sparCC in GPU parallel. Takes the original data set, randomizes the data in the across 
rows in the same columns a numPerturbs amount of times. Runs sparCC on this new random data 
as well as the original data. Then compares the perturbed networks to the original sparCC
results, and 0s out any correlations that also appear more than limit amout of times. 

Rows: Number of samples
Cols: Number of sets to compare
numPerturbs: Number of times to perturb the original dataset
limit: Number of times the original correlation is allowed to appear across the permutations

Returns an array of size cols*cols, filled with the matrix of correlations

*/

#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <sys/time.h>
#include <curand.h>
#include <curand_kernel.h>
#include <getopt.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include "csv.h"
#include "libcsv.c"
#include "SparCCGPUPlugin.h"

#include <fcntl.h>
#include <iostream>
#include <fstream>
#include <vector>
using std::ofstream;
using std::ios;
using std::cout;
using std::endl;
using std::vector;

int doPrint = 1;
int r = 0;
int columns = 0;
int rowCSV = 0;
int colCSV = 0;
double * buffer1;
double * buffer2;
double * dataMatrix;
int currCol = 0;
int counter = 0;
//static char* fileName[] = "normFileNever.csv";


///////////////////////////////////////////////////////////////////////////////////////////////////////////
// HELPER CODE TO INITIALIZE, PRINT AND TIME
struct timeval start, end;
void initialize(double *a, int N) {
  int i;
  for (i = 0; i < N; ++i) { 
    a[i] = pow(rand() % 10, 2); 
  }                                                                                                                                                                                       
}

void writeCSV(std::string outputfile, std::vector<std::string>& taxa, double* a, int N) {
   std::ofstream outfile(outputfile.c_str(), ios::out);
   int M = taxa.size()-1;
   for (int i = 0; i < taxa.size(); i++) {
      outfile << taxa[i];
      if (i != M)
         outfile << ",";
         //outfile << "\n";
      //else
   }
   int i;
   for (i = 0; i < N; ++i){
      //printf("%.14G ", a[i]);
   if((i % M) == 0) {
        outfile << "\n";
        outfile << taxa[(i/M)+1];
        outfile << ",";
   }
   else
        outfile << ",";

   outfile << a[i];
        //printf("\n\n  %d \n\n", (i/126)+2);
        //printf("\n");
   }
   //else
   //     outfile << ",";
        //printf(",");
    //}
      //printf("\n");
}
/*
void print(double* a, int N) {
   if (doPrint) {
   int i;
   for (i = 0; i < N; ++i){
   if((i % 126) == 0)
        printf("\n\n  %d \n\n", (i/126)+2);
        
      printf("%.14G ", a[i]);
    }
      printf("\n");
   }
} */ 

void starttime() {
  gettimeofday( &start, 0 );
}

void endtime(const char* c) {
   gettimeofday( &end, 0 );
   float elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
   printf("%s: %f ms\n", c, elapsed); 
}

void init(double* a, int N, const char* c) {
  printf("\n******************************* %s *******************************\n", c);
  //initialize(a, N); 
  //print(a, N);
  starttime();
}

void finish(std::string file, std::vector<std::string>& taxa, double* a, int N, const char* c) {
  writeCSV(file.c_str(), taxa, a, N);
  //print(a, N);
  printf("Number of rows: %d | Number of cols: %d | ", r, columns);
  endtime(c);
  printf("*******************************************************************\n\n");
}

////////////////////////////////////////////////////////


/*
Parameters: 
    matrix: Matrix to be logged
    n: Size of the matrix
    
    Logs the entire matrix using gpu methods. If the matrix holds a 0, first adds a
    extremely small amount to make sure all logs are done to non-zero numbers.
*/
__global__ void gpu_logMatrix(double * matrix, int n){   
    int element = blockIdx.x*blockDim.x + threadIdx.x;
    if(element < n){
        if(matrix[element] == 0)
            matrix[element] = matrix[element]+ 0.0000000000001;
        matrix[element] = log(matrix[element]);    
    }
}


/*
Parameters:
    states: The curand generator
    matrix: Original matrix to be perturbed
    newMatrix: Location to place the new matrix
    rows: Number of rows in the original matrix
    cols: Number of columns in the original matrix
    
    Takes a matrix and randomizes the data across rows using GPU methods. Takes a random number from a random row but the same
    column from the original matrix. Samples with replacement.
*/
__global__ void gpu_perturb(curandState_t* states, double * matrix, double * newMatrix, int rows, int cols){
    
    int element = blockIdx.x*blockDim.x + threadIdx.x;
    if(element < (rows*cols)){        
        int currentCol = element%cols;       
        newMatrix[element] =   matrix[((curand(&states[element])%rows)*cols) + currentCol ];      
    } 
}

/*
Parameters:
    matrix: Matrix where the data is located
    col1: The first set to be used (x_i)
    col2: The second set to be used (x_j)
    rows: Numbers of rows in the matrix
    cols: Numbers of columns in the matrix
    
    Finds the variance of the log transformed ratio of the set across col1 and col2. 
    
Returns:
    variance: The variance of the log transformed ratio (Assuming log calculations performed earlier)
*/
__device__  double varianceOfRatio(double * matrix, int col1, int col2, int rows, int cols){
    
    double* set1 = matrix + col1;
    double* set2 = matrix + col2;
    
    int ii, jj;
    
    double mean = 0; 
    double variance = 0;

//Calculate average for the ratio set. Assumes log transformations previous
    for(ii = 0; ii < rows; ii++){
        mean = mean + ((*(set1+(cols*ii))) - (*(set2+(cols*ii))));   //division of a number in log is subracting the log of the two             
    }
    mean = mean/rows;
    double x;
    
//using the mean and log transformed sets, calculate variance of ratio
    for(jj = 0; jj < rows; jj++){
        x = ((*(set1+(cols*jj))) - (*(set2+(cols*jj))));    //division of a number in log is subracting the log of the two 
        variance = variance + ((x - mean) * (x - mean));                
    }
    
    variance = variance/(rows-1);
    
    return variance;
}

/*
Parameters:
    matrix: Matrix where the data is located
    col1: The set to find the variance of
    rows: Numbers of rows in the matrix
    cols: Numbers of columns in the matrix
    
    Finds the variance of the log transformed set across col1. 
    
Returns:
    variance: The variance of the log transformed set (Assuming log calculations performed earlier)
*/
__device__ double variance(double * matrix, int col1, int cols, int rows){
    
    double* set1 = matrix + col1;
    
    int ii, jj;
    
    double mean = 0; 
    double variance = 0;
    
//Calculate average for the set
    for(ii = 0; ii < rows; ii++){
        mean = mean + (*(set1+(cols*ii)));                
    }
    mean = mean/rows;
    double x;
    
//using the mean and log transformed set, calculate variance
    for(jj = 0; jj < rows; jj++){
        x = (*(set1+(cols*jj)));
        variance = variance + ((x - mean) * (x - mean));                
    }
    
    variance = variance/(rows-1);
    
    return variance;
}

/*
Parameters:
    matrix: The matrix to find correlations of
    corMatrix: Matrix to place the newly found correlations in
    rows: Number of rows of matrix
    cols: Number of cols of matrix (cols*cols is the size of corMatrix)
    
    Runs the sparCC algorithm on matrix, places the correlations into corMatrix using GPU methods. 
    sparCC finds the correlation based on variances between sets. Numbers must be log transformed prior.
*/
__global__ void gpu_sparCC(double * matrix, double * corMatrix, int rows, int cols){
    
    int element = blockIdx.x*blockDim.x + threadIdx.x;

    if(element < (cols*cols)){      
       
        int ii = element/cols;  //First column set to use
        int jj = element%cols;  //Second column set to use
        double w1, w2, covariance, correlation;

//t_ij of original algorithm, calculated by finding the variance of the log transformed ratio  
        covariance = varianceOfRatio(matrix, ii, jj, rows, cols);

//The variances of the two individual log transformed sets        
        w1 = variance(matrix, ii, cols, rows);
        w2 = variance(matrix, jj, cols, rows);
                
        
        correlation = (w1 + w2 - covariance)/(2 * sqrt(w1) * sqrt(w2));
              
            
        *(corMatrix + (cols * ii) + jj) = correlation;
        
    }
}

/* this GPU kernel function is used to initialize the random states
-Taken from online source (Nvidia)
 */
__global__ void init(unsigned int seed, curandState_t* states) {
int element = blockIdx.x*blockDim.x + threadIdx.x;
  /* we have to initialize the state */
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              element, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[element]);
}

/*
Parameters:
    matrix: Matrix containing the actual correlations
    compareMatrix: Matrix containing the correlation of an ith perturbation
    pValueMatrix: Matrix containing the number of times compareMatrix correlations have been greater than matrix correlations
    index: Index to check if pValueMatrix must be initialized to 0
    cols: Number of cols for matrix (size of matrix is cols*cols)
    
    Calculates whether the correlation values in compareMatrix have a greater magnitude than the correlations in matrix 
    using GPU methods. If so, increment the number of the corresponding location in pValueMatrix.
*/
__global__ void gpu_pValueCalc(double * matrix, double * compareMatrix, double * pValueMatrix, int index, int cols){
    int element = blockIdx.x*blockDim.x + threadIdx.x;
    
    
    if(element < (cols*cols)){
        if(index == 0) pValueMatrix[element] = 0;
        
        if(matrix[element] > 0){
            if(compareMatrix[element] > (matrix[element]+0.000000000000001)){
                pValueMatrix[element] = pValueMatrix[element]+1;
            }
        }                
        if(matrix[element] < 0){
            if(compareMatrix[element] < (matrix[element]-0.000000000000001)){
                pValueMatrix[element] = pValueMatrix[element]+1;
            }
        }
    }            
}

/*
Parameters:
    matrix: Matrix containing the actual correlations
    pValueMatrix: Matrix containing the p values calculated earlier
    limit: The number of allowed p values
    rows: ----
    cols: Number of cols in matrix and pValueMatrix (size of both is cols*cols)
    
    Check the pValueMatrix against the limit using GPU methods. If greater than limit, 0 out
    that location in matrix.
*/
__global__ void gpu_pValueCompare(double * matrix, double * pValueMatrix, int limit, int rows, int cols){
    int element = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(element < (cols*cols)){
        if(pValueMatrix[element] > limit)
            matrix[element] = 0;
    }
}


void cb1 (void *s, size_t i, void *p) { 
  if(rowCSV == 0){
   // csv_fwrite(stdout, s, i);
  }
  else if(currCol != 0){
    char c1[25];
    char * c = c1;
    csv_write(c, 50, s, i);
    c = c+1;
    *strstr(c,"\"") = '\0';
    
    *buffer1 = strtod(c, NULL);
    buffer1++;
  } 
  currCol++;
}

void cb2 (int c, void *p) {
 // put_comma = 0;
  //putc('\n', stdout);
  colCSV = currCol-1;
  
  if(rowCSV == 0)
    dataMatrix = (double *) malloc(sizeof(double) * colCSV);
  else{ 
    dataMatrix = (double *) realloc(dataMatrix, (sizeof(double) * colCSV * rowCSV));
    while(buffer1 != buffer2){
        dataMatrix[counter] = *buffer2;
        buffer2++;       
        counter++;
    }
  }

  buffer1 = (double *) malloc(sizeof(double) * colCSV);
  buffer2 = buffer1;
  
  currCol = 0;
  rowCSV++;
}

//int main(int argc, char **argv){

void SparCCGPUPlugin::input(std::string file) {
    inputfile = file;
    extern char *optarg;
    extern int optind;
    //int c;
    //int rows = 100;
    //int cols = 100;

    // ./sparGPU -f [filename] -p
  /*  while ((c = getopt(argc, argv, "p")) != -1)
        switch (c) {
       // case 'r':
         //   rows = atoi(optarg);
           // break;
        //case 'c':
          //  cols = atoi(optarg);
            //break;
        //case 'f':
            //fileName = optarg;
        case 'p':
            doPrint = 1;
            break;         
        case '?':
            break;
        }
    */    
        
        
        
        
    struct csv_parser p;
    int i;
    char ch;
        
    csv_init(&p, 0);
    
    int inFileId;
        //inFileId = open(fileName, O_RDWR);
        inFileId = open(inputfile.c_str(), O_RDWR);
        dup2( inFileId, fileno(stdin) );
        close(inFileId);
        
    while ((i=getc(stdin)) != EOF) {
        ch = i;
        if (csv_parse(&p, &ch, 1, cb1, cb2, NULL) != 1) {
            fprintf(stderr, "Error: %s\n", csv_strerror(csv_error(&p)));
            exit(EXIT_FAILURE);
        }
    }
    rows = rowCSV - 1;
    cols = colCSV;
        
    csv_fini(&p, cb1, cb2, NULL);
    csv_free(&p);
  
    free(buffer1);    
     
        
    /*    
        
     counter = 0;
  while(counter < (rows*cols)){
    printf("%.15G\n", dataMatrix[counter]);
    counter++;  
  }
  printf("\n");   
      */  
        
          printf("Testing if it gets the right number of rows/cols %d , %d \n", rows, cols);

    // Once more, get the taxa
    taxa.resize(cols+1);
    std::ifstream infile(inputfile.c_str(), ios::in);
    std::string firstline;
    getline(infile, firstline);
    //std::cout << "FIRSTLINE: " << firstline << std::endl;
    //int startpos = 0;
    for (int i = 0; i < cols+1; i++) {
       //std::cout << firstline.substr(0, firstline.find_first_of(',')) << std::endl;
       taxa[i] = firstline.substr(0, firstline.find_first_of(','));
       firstline = firstline.substr(firstline.find_first_of(',')+1);  
    }     
    }   
        
void SparCCGPUPlugin::run() {    
        
    r = rows;   //set for printing
    columns = cols;   //set for printing
    
    
    
    
    int n = rows * cols;            
    int numThreads = 512;           //number of threads to use
    int numCores = n / 512 + 1;     //number of cores to use
    int numPerturbs = 100;          //number of times to perturb the original data
    int jj, kk;                 //indexes to be used.
    //int ii;
//Points to one of the temp arrays used. Used to hold original data
    //double* a = (double *) malloc(sizeof(double) * n);    
    //double * a = dataMatrix;
    a = dataMatrix;
//initialize random test array    
   // for(ii = 0; ii < n; ii++){
     //   a[ii] = (rand() % 1000); 
    //}

   
     
    
    
//Pointer to one of the temp arrays. Used to hold correlation values
    //double* b = (double *) malloc(sizeof(double) * cols * cols); 
    b = (double *) malloc(sizeof(double) * cols * cols);
  // Test 2: Vectorization
  init(a, n, "GPU");
  
    curandState_t* states;

  /* allocate space on the GPU for the random states */
  cudaMalloc((void**) &states, n * sizeof(curandState_t));

  /* invoke the GPU to initialize all of the random states */
  init<<<numCores, numThreads>>>(time(0), states);

    double* initialMatrix;          //Matrix of initial data inside GPU
    double* perturbMatrix;          //Matrix of the current perturbed data inside the GPU
    double* corrMatrix;             //Matix of the original correlations inside the GPU
    double* perCorMatrix;           //Matrix of the perturbed correlations inside the GPU
    double* pValueMatrix;           //Matrix of the p values inside the GPU

    
//Data allocations inside the GPU
    cudaMalloc(&initialMatrix, n*sizeof(double));
    cudaMalloc(&perturbMatrix, n*sizeof(double));
    cudaMalloc(&corrMatrix, cols*cols*sizeof(double));
    cudaMalloc(&perCorMatrix, numPerturbs*cols*cols*sizeof(double));
    cudaMalloc(&pValueMatrix, cols*cols*sizeof(double));

//Copying the array from CPU to GPU
    cudaMemcpy(initialMatrix, a, n*sizeof(double), cudaMemcpyHostToDevice);
    
//Log this matrix for later use
    gpu_logMatrix<<<numCores, numThreads>>>(initialMatrix, n);
    
//Perturb this log transformed data numPerturb amount of times
    for(jj = 0; jj < numPerturbs; jj++){
        numCores = n/numThreads + 1;
        gpu_perturb<<<numCores, numThreads>>>(states, initialMatrix, perturbMatrix, rows, cols);
               
        numCores = (cols*cols)/numThreads + 1;         //Edit the numCores based on the new matrix
        
        //runs the sparCC algorithm on the newly perturbed matrix
        gpu_sparCC<<<numCores, numThreads>>>(perturbMatrix, (perCorMatrix+(jj*cols*cols)), rows, cols);       
    }
    
//Run sparCC algorithm on original data  
    gpu_sparCC<<<numCores, numThreads>>>(initialMatrix, corrMatrix, rows, cols);
        
//Calcluate the pvalues across all perturbations       
    for(kk = 0; kk < numPerturbs; kk++)
        gpu_pValueCalc<<<numCores, numThreads>>>(corrMatrix, (perCorMatrix+(kk*cols*cols)), pValueMatrix, kk, cols);
    
        
    int limit = 101;      //hard coded threshold

//0s outs any correlation above threshold p value        
    gpu_pValueCompare<<<numCores, numThreads>>>(corrMatrix, pValueMatrix, limit, rows, cols);
             
//Copy the correlations back to the CPU       
    cudaMemcpy(b, corrMatrix, cols*cols*sizeof(double), cudaMemcpyDeviceToHost);
    
//Free memory used inside of GPU
    cudaFree(&initialMatrix); 
    cudaFree(&perturbMatrix); 
    cudaFree(&corrMatrix); 
    cudaFree(&perCorMatrix); 
    cudaFree(&pValueMatrix); 
}

void SparCCGPUPlugin::output(std::string file) {
    outputfile = file;
    finish(outputfile, taxa, b, cols*cols, "GPU");
    
//Free memory used by the CPU
    free(a);
    free(b);
    
}

PluginProxy<SparCCGPUPlugin> SparCCGPUPluginProxy = PluginProxy<SparCCGPUPlugin>("SparCCGPU", PluginManager::getInstance());

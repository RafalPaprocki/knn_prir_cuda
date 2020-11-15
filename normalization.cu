#include "CsvFunctions.h"

void normalize(string outFile, vector< pair<string, vector<double>> >& data);
double findMin(vector<double> data);
double findMax(vector<double> data);

__global__ void normalizeCuda(double *data, int *size, double *min, double *max) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    while (idx < *size) {
        if (idx < *size){
            data[idx] = (data[idx] - *min) / (*max - *min);
        }
        idx += blockDim.x * gridDim.x;
    }
}

int main(int argc, char* argv[])
{	
    vector<pair<string, vector<double>> > out;
    out = readFromCsv("dataset/bigheartdata.csv");
    normalize("normalized_cuda.csv", out);
}


void normalize(string outFile, vector< pair<string, vector<double>> >& data) {
    double min;
    double max;	
    int size = data[0].second.size();
    double * tempData = new double[size];
    for (int i = 0; i < data.size(); ++i) {
        if (data[i].first != "target") {
            min = findMin(data[i].second);
            max = findMax(data[i].second);
            
            //copy vector to array
            for (int j = 0; j < size; ++j){
                tempData[j] = data[i].second[j];
            }

            //creating cuda variables and memory allocation
            double * cudaData;
		    double * cudaMin;
            double * cudaMax;
		    int * dataSize;
            cudaMalloc( (void**)&cudaData, sizeof(double) * size ) ;    
    	    cudaMalloc( (void**)&cudaMin, sizeof(double));
    	    cudaMalloc( (void**)&cudaMax, sizeof(double));
		    cudaMalloc( (void**)&dataSize, sizeof(int));

            cudaMemcpy( dataSize, &size, sizeof(int), cudaMemcpyHostToDevice);
  	        cudaMemcpy( cudaData, tempData, sizeof(double) * size, cudaMemcpyHostToDevice );
    	    cudaMemcpy( cudaMin, &min, sizeof(double), cudaMemcpyHostToDevice );
    	    cudaMemcpy( cudaMax, &max, sizeof(double), cudaMemcpyHostToDevice );
            //normalizowanie
            int num_blocks = ceil(size / 1000) + 1;
            normalizeCuda<<<num_blocks, 1000>>>(cudaData, dataSize, cudaMin, cudaMax);
            
            //copy arra preprocessed by cuda
            cudaMemcpy( tempData, cudaData, sizeof(double) * size, cudaMemcpyDeviceToHost );

            //copy table to vector 
            for (int j = 0; j < size; ++j){
                data[i].second[j] = tempData[j];
            }

            cudaFree( cudaData );
            cudaFree( cudaMin );
            cudaFree( cudaMax );
            cudaFree( dataSize );
            cudaDeviceSynchronize();
        }
    }
    delete tempData;
	
    // zapisywanie do pliku
    //writeToCsv(outFile, data);
}

double findMin(vector<double> data) {
    double min = 10000000;
    for (int i = 0; i < data.size(); ++i) {
	
        if (data[i] < min) {
            min = data[i];
        }
    }
    return min;
}

double findMax(vector<double> data) {
    double max = -10000000;
    for (int i = 0; i < data.size(); ++i) {
	
        if (data[i] > max) {
            max = data[i];
        }
    }
    return max;
}


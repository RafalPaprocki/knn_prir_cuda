#include "CsvFunctions.h"

double findAverage(vector<double> data);
double findDeviation(vector<double> data);
void standardize(string outFile, vector< pair<string, vector<double>> >& data);

__global__ void standarizeCuda(double *data, double *deviation, double *avg, int *size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    while (idx < *size) {
        if (idx < *size){
            int tempIdx = idx % 13;
            data[idx] = (data[tempIdx] - *avg) / *deviation;
        }
        idx += blockDim.x * gridDim.x;
    }
}

int main(int argc, char* argv[])
{
    vector<pair<string, vector<double>> > out;
    out = readFromCsv("dataset/bigheartdata.csv");
    standardize("standarized_cuda.csv" , out);
}


void standardize(string outFile, vector< pair<string, vector<double>> >& data) {
    int i;
    int size = data[0].second.size();
    double * tempData = new double[size];
    for ( i = 0; i < data.size(); ++i) {
        if (data[i].first != "target") {
            double avg = findAverage(data[i].second);
            double deviation = findDeviation(data[i].second);
            //copy vector to array
            for (int j = 0; j < size; ++j){
                tempData[j] = data[i].second[j];
            }

            //creating cuda variables and memory allocation
            double * cudaData;
		    double * cudaAvg;
            double * cudaDeviation;
		    int * dataSize;
            cudaMalloc( (void**)&cudaData, sizeof(double) * size ) ;    
    	    cudaMalloc( (void**)&cudaAvg, sizeof(double));
    	    cudaMalloc( (void**)&cudaDeviation, sizeof(double));
		    cudaMalloc( (void**)&dataSize, sizeof(int));
           
            cudaMemcpy( dataSize, &size, sizeof(int), cudaMemcpyHostToDevice);
  	        cudaMemcpy( cudaData, tempData, sizeof(double) * size, cudaMemcpyHostToDevice );
    	    cudaMemcpy( cudaDeviation, &deviation, sizeof(double), cudaMemcpyHostToDevice );
    	    cudaMemcpy( cudaAvg, &avg, sizeof(double), cudaMemcpyHostToDevice );
		
            //standardization
            int num_blocks = ceil(size /1000) + 1;
            standarizeCuda<<<num_blocks, 1000>>>(cudaData, cudaDeviation, cudaAvg, dataSize);

            cudaMemcpy( tempData, cudaData, sizeof(double) * size, cudaMemcpyDeviceToHost );

            //copy table to vector 
            for (int j = 0; j < size; ++j){
                data[i].second[j] = tempData[j];
            }
           
        }
    }

    // zapisywanie do pliku
    // writeToCsv(outFile, data);
}

double findAverage(vector<double> data) {
    double average = accumulate(data.begin(), data.end(), 0.0) / data.size();
    return average;
}

// https://www.statisticshowto.com/probability-and-statistics/standard-deviation/#HFSSD - pattern
double findDeviation(vector<double> data) {
    double sum = accumulate(data.begin(), data.end(), 0.0);
	
    double square = sum * sum / data.size();
    double squareSum = 0;

    for (int i = 0; i < data.size(); ++i) {
        squareSum += data[i] * data[i];
    }

    double diff = squareSum - square;
    double variance = diff / (data.size() - 1);
    double deviation = sqrt(variance);

    return deviation;
}

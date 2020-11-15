#include "CsvFunctions.h"
#include <algorithm> 

__global__ void distanceForEuclidean(double *data, double *test, int *size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    while (idx < *size) {
        if (idx < *size){
            int tempIdx = idx % 13;
			double diff = data[idx] - test[tempIdx];
            data[idx] = diff * diff;
        }
        idx += blockDim.x * gridDim.x;
    }
}

class Knn {
private:
	int k_numbers;
	int metric;
	int targetColumn;
	int threadNum;
public:
	vector<vector<double>> trainData;
	vector<vector<double>> learningData;
	Knn(int k = 1, int m = 1, int thrd = 1) {
		k_numbers = k;
		metric = m;
		targetColumn = 0;
		threadNum = thrd;
	}

	void setMetric(int number) {
		metric = number;
	}

	void setK(int k) {
		k_numbers = k;
	}

	void loadData(string file, int targetColumnNumber, int trainingPercent = 30) {
		targetColumn = targetColumnNumber - 1;
		vector<vector<double>> data = readFromCsvWithoutLabels(file);
		std::random_shuffle(data.begin(), data.end());
		int startIndex = (trainingPercent / 100.0) * data.size();
		vector<vector<double>> train(data.end() - startIndex, data.begin() + data.size());
		data.erase(data.end() - startIndex, data.begin() + data.size());
		learningData = data;
		trainData = train;
	}

	int predict(vector<double> features) {
		vector<pair<double, int>> distancesAndLabels = {};
		int size = learningData.size() * (learningData[0].size() - 1);
		double * tempData = new double[size];
		double * tempLabels = new double[learningData.size()];
		double * tempTestData = new double[learningData[0].size() - 1];
		for (int i = 0; i < learningData.size(); ++i){
			for( int j = 0; j < learningData[0].size(); ++j){
				if (j != 13) {
					tempData[i*13+j] = learningData[i][j];
				} else {
					tempLabels[i] = learningData[i][j];
				}
			}
		}

		for (int i = 0; i < features.size(); ++i) {
			tempTestData[i] = features[i];
		}
		
		double * cudaData;
		double * cudaTestValues;
		int * dataSize;
		cudaMalloc( (void**)&cudaData, sizeof(double) * size ) ;
    	cudaMalloc( (void**)&cudaTestValues, sizeof(double) * 13);
		cudaMalloc( (void**)&dataSize, sizeof(int));
		cudaMemcpy( dataSize, &size, sizeof(int), cudaMemcpyHostToDevice);
  	    cudaMemcpy( cudaData, tempData, sizeof(double) * size, cudaMemcpyHostToDevice );
    	cudaMemcpy( cudaTestValues, tempTestData, sizeof(double) * 13, cudaMemcpyHostToDevice );
		int num_blocks = ceil(size /1000) + 1;
		distanceForEuclidean<<<num_blocks, 1000>>>(cudaData, cudaTestValues, dataSize);
    	cudaMemcpy( tempData, cudaData, sizeof(double) * size, cudaMemcpyDeviceToHost );
 		cudaFree( cudaData );
    	cudaFree( cudaTestValues );
		cudaDeviceSynchronize();
		
		for (int i = 0; i < learningData.size(); ++i) {
		 	double sum = 0;
		  	for( int j = 0; j < learningData[0].size() - 1; ++j){
		 		 sum += tempData[i*13 + j];
		 	 }
		 	double euclidean = sqrt(sum);
		 	distancesAndLabels.push_back({ sum, tempLabels[i] });
	    }

	    sort(distancesAndLabels.begin(), distancesAndLabels.end());
	    vector<int> nearestResults = {0, 0};

		for (int i = 0; i < k_numbers; ++i) {
		  	nearestResults[(int)distancesAndLabels[i].second]++;
		}

		delete tempData;
		delete tempLabels;
		delete tempTestData;

		if (nearestResults[0] > nearestResults[1]) {
		    return 0;
		}
		else {
		    return 1;
		}

	}

	double checkAccuracy() {
		int good = 0;
		int bad = 0;
	
		for (int i = 0; i < 10; ++i) {
			int predictedTarget = predict(trainData[i]);
			if (predictedTarget == trainData[i][targetColumn]) {
				++good;
			}
			else {
				++bad;
			}
		}
		return good / (double)(good + bad);
	}

	double euclideanDistance(vector<double> learning, vector<double> target) {
		vector<double> distanceSquares = {};
		double euclideanDistance = 0;
		for (int i = 0; i < learning.size(); ++i) {
			if (i != targetColumn) {
				double diff = learning[i] - target[i];
				distanceSquares.push_back(diff * diff);
			}
		}
		
		for (int i = 0; i < distanceSquares.size(); ++i) {
			euclideanDistance += distanceSquares[i];
		}

		euclideanDistance = sqrt(euclideanDistance);
		return euclideanDistance;
	}
};


int main(int argc, char* argv[]) {
	Knn* knn = new Knn(5,0);
	knn->loadData("dataset/bigheartdata.csv", 14, 30);
	double accuracy = knn->checkAccuracy();
	cout << endl << accuracy;
	delete knn;
}

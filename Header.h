#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
 
#include <cstdio>
#include <vector>
//#include "spline.h"

#include<opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core.hpp>
#include<type_traits>
#include <list>
#include "main_shortestPath.h"
#include "bellman.cuh"
#include "kernels.cuh"
#include "spline.h"
#include "SGSmooth.hpp"
///#include "SGSmooth.hpp"
//#include "dijkstra_kernel.h"
//#include "Header_Dijkstra.h"
using namespace std;
using namespace cv;
 
#define V 14544
//#include <helper_cuda.h>

//Parameters
int numFrames = 1;
int imageSize = 867 * 500;

float X_RESOLUTION = 14.4;
float Y_RESOLUTION = 9.7;
float LATERAL_RESOLUTION = 2.0;
float AXIAL_RESOLUTION = 2.0;

//Get BWImage Parameters
float X_FilterSize = 147.4;
float Y_FilterSize = 73.3;
float MIN_CLUSTER_SIZE = 18000;

typedef struct 
{
    vector<int>Index, Edge, Vertex;
    vector<float>Weight;
}WeightingMatrix;

//WeightingMatrix inputW, outputW;
WeightingMatrix W;

vector<WeightingMatrix>inputW,outputW;

Mat normal_getBWImage(Mat resizedImage);

template <typename T>
void writeMatToFile(cv::Mat m, const char* filename);

Mat C_fspecial_LOG(double* kernel_size, double sigma);
Mat diff(Mat Image);
Mat GetBorder_Image(Mat FinalBWImage);
vector<WeightingMatrix> bwWeightingMatrix(Mat FinalBWImage);

//void ConvertDataFormatToGraph(Mat totalEdge, Mat totalWeights, vector<int>Index, vector<int>Edge, vector<int>Vertex, vector<float>Weight);
WeightingMatrix ConvertDataFormatToGraph(Mat totalEdge, Mat totalWeights);
vector<WeightingMatrix> GenerateWeightingMatrices(Size MatrixSize, Mat ImageEdges, Mat Weights,Mat ColumnEdges, float MIN_WEIGHT);
Mat normalizeValues(Mat valuesRowWise);
 
vector<Point> getRegion(Size ImageSize, Mat topLayer, Mat bottomLayer, vector<int>invalidIndices);

vector<int> cutRegion(Size ImageSize,vector<Point>regionIndices, WeightingMatrix weightEdgeIndexstruct);
void ind2sub(const int sub, const int cols, const int rows, int& row, int& col);
//float modulusValue(float a, float b);
int sub2ind(const int row, const int col, const int cols, const int rows);
//void ind2sub(const int sub, const int cols, const int rows, int& row, int& col);
 
//Mat resampleLayers(Mat Borders, Size ResizedSize, Size OriginalSize);

void Text_FileRead(const char* TextFileWithPath, unsigned short* preprocessing);
//Mat Dijkstra_ShortestPath(Mat totalEdges, Mat totalWeights);
//void Text_FileRead1(const char* TextFileWithPath, int* preprocessing);
//Mat drawLayers(Mat InputImage1, Mat layers);

void addEdge(vector<pair<unsigned short, float>> graph[], unsigned short u, unsigned short v, float weight);

int* runBellmanFordOnGPU(vector<int>V1, vector<int>I, vector<int>E, vector<float>W);
vector<int> Pathshort(int* out_pred, int start, int destination);
float modulusValue(float a, float b);
Mat resampleLayers(Mat layers, Size originalSize, Size newSize);
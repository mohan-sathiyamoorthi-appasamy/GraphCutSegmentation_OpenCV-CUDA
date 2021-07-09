
#include "Header.h"

//Read Registered BM3 Image
int main(int argc, char** argv)
{
    //1.Read Retina Tiff Image 
    double t = (double)getTickCount();
	Mat InputImage = imread("AvgImg.tif", IMREAD_GRAYSCALE);

    //2.Type Conversion-Integer to Float
	InputImage.convertTo(InputImage, CV_32FC1);

	//3.Resized the image to Down-Sample
	Mat resizedImage;
	resize(InputImage, resizedImage, Size(70,202));

	//4.Normalization 
	normalize(resizedImage,resizedImage,0,255,CV_MINMAX);

	//5.Get Black and White Image of Hyper Reflective Bands
	Mat FinalBWImage;
	FinalBWImage = normal_getBWImage(resizedImage);

    //6.Get Borders of the two hyper reflective layers
    Mat Borders = GetBorder_Image(FinalBWImage);
    //cout << Borders.row(0) << endl;
    //7.Resample & Smooth layers - Convert Down sample Image to Original Image Size
    Size resizedSize = { 70,201 };
    Size originalSize = { 500,974 };
    Mat layers = resampleLayers(Borders, resizedSize, originalSize);

   
    //8.Finally, Overlay the layers on retina image
    t = ((double)getTickCount() - t) / getTickFrequency();
    std::cout << "Times passed in seconds: " << t << std::endl;
    namedWindow("Image");
    
    imshow("Image", FinalBWImage);
    waitKey(0);
}

//*************************Main Function is Over**********************************//

//5.Get Black and White Image of Hyper Reflective Bands
Mat normal_getBWImage(Mat resizedImage)
{ 
    cout << "Normal Get BW Image" << endl;
   //1.Smoothning Image using Gaussian Filter - Reduce Noise
    Mat blurredImage;
    double kernel_size[2] = { 10,8};    // kernel size set to 4x4
    double sigma = 11;
    Mat kernel = C_fspecial_LOG(kernel_size, sigma);
    filter2D(resizedImage, blurredImage, 5, kernel);

    //2.Find the edges of the Image
    Mat derivative;
    derivative = diff(blurredImage);

    //3.Padding the Borders
    cv::Mat derivative_Pad;
    int padding = 1;
    derivative_Pad.create(derivative.rows + 1 * padding, derivative.cols + 0, derivative.type());
    derivative_Pad.setTo(cv::Scalar::all(0));
    derivative.copyTo(derivative_Pad(Rect(0, padding, derivative.cols, derivative.rows)));

    //4.Threshold based Binary Image Creation
    Mat bwImage;
    threshold(derivative_Pad, bwImage, 0.8, 1, 0);

    //5.Open any gaps in the Clusters
    Mat structuringElement;
    structuringElement = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(bwImage, bwImage, MORPH_OPEN, structuringElement);

    //6.Remove All Clusters smallar than a Certain Size
    Mat labels;
    bwImage.convertTo(bwImage, CV_8UC1);
    Mat stats, centroid;
    int nncomps = connectedComponentsWithStats(bwImage, labels, stats, centroid, 8);

    Mat mask(labels.size(), CV_8UC1, Scalar(0));
    Mat surfSup = stats.col(4) > 2000;

    for (int i = 1; i < nncomps; i++)
    {
        if (surfSup.at<uchar>(i, 0))
        {
            mask = mask | (labels == i);
        }
    }

    bwImage.copyTo(bwImage, mask);
    bwImage.convertTo(bwImage, CV_32FC1);

    //7.Close any gaps in the Clusters
    Mat FinalBW;
    morphologyEx(bwImage, FinalBW, MORPH_CLOSE, structuringElement);

    return FinalBW;
}

//5.1 Smoothning the image using Gaussian Filer
Mat C_fspecial_LOG(double* kernel_size, double sigma)
{
    cout << "C_fspecial_LOG" << endl;
    double size[2] = { (kernel_size[0] - 1) / 2   , (kernel_size[1] - 1) / 2 };
    double std = sigma;
    const double eps = 2.2204e-16;
    cv::Mat kernel(kernel_size[0], kernel_size[1], CV_64FC1, 0.0);
    int row = 0, col = 0;
    for (double y = -size[0]; y <= size[0]; ++y, ++row)
    {
        col = 0;
        for (double x = -size[1]; x <= size[1]; ++x, ++col)
        {
            kernel.at<double>(row, col) = exp(-(pow(x, 2) + pow(y, 2)) / (2 * pow(std, 2)));
        }
    }

    double MaxValue;
    cv::minMaxLoc(kernel, nullptr, &MaxValue, nullptr, nullptr);
    Mat condition = ~(kernel < eps* MaxValue) / 255;
    condition.convertTo(condition, CV_64FC1);
    kernel = kernel.mul(condition);

    cv::Scalar SUM = cv::sum(kernel);
    if (SUM[0] != 0)
    {
        kernel /= SUM[0];
    }

    return kernel;
}
//********************Get B/W Retina Image Function is Over*************************************//

// 6. Get Borders of the tow hyper reflective bands
Mat GetBorder_Image(Mat FinalBWImage)
{
    cout << "GetBorder_Image" << endl;
    vector<int>line;
    //1.Create borders 
    copyMakeBorder(FinalBWImage, FinalBWImage, 0, 0, 1, 1, BORDER_CONSTANT, Scalar(0));


    Size newImageSize = size(FinalBWImage);
    int newImageWidth = newImageSize.width;
    int imageHeight = newImageSize.height;

    //2.Get the Weighting Matrics
    vector<WeightingMatrix> WeightEdgeVertex = bwWeightingMatrix(FinalBWImage);
     
    //3.Cut each border
    int numBorders = 4;
    Mat lines(numBorders,newImageWidth, CV_32S, Scalar(0));

    vector<int>xval, yval;
    vector<int>invalidIndices{};    
    int y;
    Mat prevBorder(1, newImageWidth, CV_32S);
    int removedInd;
   
    for (int iBorder = 0; iBorder < numBorders; iBorder++)
    {
       //Exclude the previous borders from the region to cut
        if (iBorder > 0)
        {
            //xal 
            for (int i = 0; i < newImageWidth - 2; i++)
            {
                xval.push_back(i + 1);
            }  
            //yval
            prevBorder = lines.row(iBorder-1);
           // cout << "PrevBorder" << prevBorder << endl;
            for (int i = 0; i < newImageWidth - 2; i++)
            {
                yval.push_back(prevBorder.at<int>(xval[i]));    
            }

            for (int i = 0; i < newImageWidth - 2; i++)
            {
                if ((yval[i] == 0) || (yval[i] == imageHeight - 1))
                {
                    xval.erase(xval.begin() + yval[i]);
                    yval.erase(yval.begin() + yval[i]);
                }
                invalidIndices.push_back(sub2ind(202, yval[i], xval[i], 72));
                //cout << "InvalidIndex length: " << invalidIndices.size() << "," << sub2ind(202, yval[i], xval[i], 72) << endl;
                //cout << invalidIndices[i] << endl;
            }
            yval.clear();

        }

        Mat yBottom(1, 72, CV_32S, Scalar(0));
        if (iBorder < numBorders)
        {
            yBottom = imageHeight - 1 * Mat(1, newImageWidth, CV_32S, Scalar(1));
        }
        else
        {
            double min, max;
            for (int i = 0; i < 72; i++)
            {
                Mat lines_col = lines.col(i);
                minMaxIdx(lines_col, &min, &max);
                yBottom.at<int>(i) = max;
            }
            
        }
       
        // 4.Get the Valid region to cut
        Size newImageSize = FinalBWImage.size();
        Mat oneceMx = Mat::zeros(1, newImageWidth, CV_32S);
       
        //vector<int>invalidIndices(14544,0);
        vector<Point>regionIndices;
       
        regionIndices = getRegion(newImageSize, oneceMx, yBottom, invalidIndices);
         
        cout << "iBorder" << (((iBorder+1) % 2)+1)-1 << endl;
        //5.Cut the region to get the border
        line = cutRegion(newImageSize, regionIndices, WeightEdgeVertex.at((((iBorder + 1) % 2) + 1) - 1));
        
        memcpy(lines.row(iBorder).data, line.data(), line.size() * sizeof(int));
        cout << lines.row(iBorder) << endl;
    }
       //Remove the added columns
       Mat bwImage_removedCols = FinalBWImage.colRange(1, 71);
       Mat lines_removedCols = lines.colRange(1, 71);
    
       //Odd Mean Value
       Scalar Mean_Odd1 = mean(lines_removedCols.row(0));
       Scalar Mean_Odd2 = mean(lines_removedCols.row(2));

       //Even Mean Value
       Scalar Mean_Even1 = mean(lines_removedCols.row(1));
       Scalar Mean_Even2 = mean(lines_removedCols.row(3));

       //Sort the lines in Ascending Order
       Mat Mean_Odd;
       Mean_Odd.push_back(Mean_Odd1);
       Mean_Odd.push_back(Mean_Odd2);
       int oddIndices[2];
       if (Mean_Odd.at<float>(0, 0) > Mean_Odd.at<float>(0, 1))
       {
           oddIndices[0] = 2;
           oddIndices[1] = 0;
       }
       else
       {
           oddIndices[0] = 0;
           oddIndices[1] = 2;
       }

       Mat Mean_Even;
       Mean_Even.push_back(Mean_Even1);
       Mean_Even.push_back(Mean_Even2);
       int evenIndices[2];
       if (Mean_Even.at<float>(0, 0) > Mean_Even.at<float>(0, 1))
       {
           evenIndices[0] = 3;
           evenIndices[1] = 1;
       }
       else
       {
           evenIndices[0] = 1;
           evenIndices[1] = 3;
       }

       Mat bottomBorders(2, 70, CV_32F, Scalar(0));
       lines_removedCols.row(oddIndices[0]).copyTo(bottomBorders.row(0));
       lines_removedCols.row(oddIndices[1]).copyTo(bottomBorders.row(1));

       Mat topBorders(2, 70, CV_32F, Scalar(0));
       lines_removedCols.row(evenIndices[0]).copyTo(topBorders.row(1));
       lines_removedCols.row(evenIndices[1]).copyTo(topBorders.row(0));

       Mat borders(4, 70, CV_32F, Scalar(0));

       topBorders.row(0).copyTo(borders.row(0));
       topBorders.row(1).copyTo(borders.row(2));
       bottomBorders.row(0).copyTo(borders.row(1));
       bottomBorders.row(1).copyTo(borders.row(3));

       //Replace Extrapolated Points (those that do not lie along a hyper-reflective band)
       int NUM_BORDER = 4;
       Mat Borders(4, 70, CV_32F, Scalar(0));
       for (int iBorder = 0; iBorder < NUM_BORDER; iBorder++)
       {
           Mat border = borders.row(iBorder);
           if ((modulusValue((iBorder + 1), 2)) == 0)
           {
               border = border - 1;
               Mat mask = border < 1;
               border.setTo(1, mask);
           }

           Mat ind(1, 70, CV_32F);
           for (int i = 0; i < 70; i++)
           {
               ind.at<float>(i) = sub2ind(202, border.at<float>(i), i, 202);
           }

           int* row = (int*)malloc(70 * sizeof(int));
           int* col = (int*)malloc(70 * sizeof(int));

           Mat FinalBwImage_Ind(1, 70, CV_32F, Scalar(0));
           for (int i = 0; i < 70; i++)
           {
               ind2sub(ind.at<float>(i), 202, 70, row[i], col[i]);

           }

           for (int i = 0; i < 70; i++)
           {
               FinalBwImage_Ind.at<float>(0, i) = bwImage_removedCols.at<float>(col[i], row[i]);
           }

           FinalBwImage_Ind.convertTo(FinalBwImage_Ind, CV_8UC1);
           Mat nonZeroCoord;
           findNonZero(FinalBwImage_Ind, nonZeroCoord);
           Size nzcord = nonZeroCoord.size();

           int x_ind = ind.at<float>(nonZeroCoord.at<Point>(0).y, nonZeroCoord.at<Point>(0).x);
           int y_ind = ind.at<float>(nonZeroCoord.at<Point>(nonZeroCoord.total() - 1).y, nonZeroCoord.at<Point>(nonZeroCoord.total() - 1).x);

           int* xStart, * yStart;
           ind2sub(x_ind, 202, 70, *xStart, *yStart);

           int* yEnd, * xEnd;
           ind2sub(y_ind, 202, 70, *xEnd, *yEnd);

           borders.row(iBorder).colRange(*xStart, *xEnd + 1).copyTo(Borders.row(iBorder).colRange(*xStart, *xEnd + 1));

       }
       cout << "Border Function is over" << endl;
       return borders;
   
}

//6.2. Get the Weighting Matrix
vector<WeightingMatrix> bwWeightingMatrix(Mat FinalBWImage)
{
    cout << "bwWeightingMatrix" << endl;
    //1. Lattice Creation for given image size
    //Find edges of column & Image(export from Matlab)
    //(This is same for all image with size of 202x70)
    Mat OriginalImage = FinalBWImage;

    Size ImageSize = size(FinalBWImage);
    int imageHeight = ImageSize.height;
    int imageWidth = ImageSize.width;

    char imagePath_ColumnEdges[100] = "columnEdges.txt";
    unsigned short* columnEdges_Data = (unsigned short*)malloc(1350 * sizeof(unsigned short));
    Text_FileRead(imagePath_ColumnEdges, columnEdges_Data);

    char imagePath_ImageEdges[100] = "ImageEdges.txt";
    unsigned short* imageEdges_Data = (unsigned short*)malloc(113362 * sizeof(unsigned short));
    Text_FileRead(imagePath_ImageEdges, imageEdges_Data);

    //Generate Column Edges
    Mat columnEdges(2, 675, CV_16U, columnEdges_Data);
    //Generate Image Edges
    Mat imageEdges(2, 56681, CV_16U, imageEdges_Data);

    // ----------------------------------------------------------------------
    // Calculate the weights based on the image gradient.
    // Lower weights are assigned to areas with a higher gradient
    // ----------------------------------------------------------------------
    // Create two edge maps(one for edges that transition from dark->light
    // in the vertical direction, and one for edges transitioning from light->dark).

    //2.Create border at top
    copyMakeBorder(FinalBWImage, FinalBWImage, 1, 0, 0, 0, BORDER_CONSTANT, Scalar(0));


    //3.Find Derivative Image 
    Mat diffImage = diff(FinalBWImage);
  

    //4.light->Dark & Dark->light Image Creation
    Mat lightDarkEdgeImage(imageHeight, imageWidth, CV_32F, Scalar(0));
    lightDarkEdgeImage.setTo(1, (diffImage > 0));

    Mat darkLightEdgeImage(imageHeight, imageWidth, CV_32F, Scalar(0));
    darkLightEdgeImage.setTo(1, abs(diffImage < 0));

    Mat mask1 = Mat::zeros(lightDarkEdgeImage.size(), CV_8U);
    bitwise_and((lightDarkEdgeImage == 0), (OriginalImage == 1), mask1);
    lightDarkEdgeImage.setTo(-1, mask1);


    Mat mask2 = Mat::zeros(darkLightEdgeImage.size(), CV_8U);
    bitwise_and((darkLightEdgeImage == 0), (OriginalImage == 1), mask2);
    darkLightEdgeImage.setTo(-1, mask2);

    //5.Calculate Gradient Weights of edge image - light->Dark Edge Image
    lightDarkEdgeImage = lightDarkEdgeImage.t();
    std::vector<float> lightDarkEdgeImage1D;
    if (lightDarkEdgeImage.isContinuous()) {

        lightDarkEdgeImage1D.assign((float*)lightDarkEdgeImage.datastart, (float*)lightDarkEdgeImage.dataend);
    }
    else {
        for (int i = 0; i < lightDarkEdgeImage.rows; ++i) {
            lightDarkEdgeImage1D.insert(lightDarkEdgeImage1D.end(), lightDarkEdgeImage.ptr<float>(i), lightDarkEdgeImage.ptr<float>(i) + lightDarkEdgeImage.cols);

        }
    }

    Mat imageEdgesIndexRow1 = imageEdges.row(0);
    Mat imageEdgesIndexRow2 = imageEdges.row(1);

    Mat lightDarkEdgeImageRow1Result(imageEdgesIndexRow1.size(), CV_32F, Scalar(0));
    Mat lightDarkEdgeImageRow2Result(imageEdgesIndexRow2.size(), CV_32F, Scalar(0));

    for (int i = 0; i < imageEdgesIndexRow1.cols; i++)
    {
        lightDarkEdgeImageRow1Result.at<float>(0, i) = lightDarkEdgeImage1D[imageEdgesIndexRow1.at<unsigned short>(0, i)];

    }

    for (int i = 0; i < imageEdgesIndexRow2.cols; i++)
    {
        lightDarkEdgeImageRow2Result.at<float>(0, i) = lightDarkEdgeImage1D[imageEdgesIndexRow2.at<unsigned short>(0, i)];
       
    }

    Mat lightDarkGradientWeights = 2 - lightDarkEdgeImageRow1Result - lightDarkEdgeImageRow2Result;

    //6.Calculate Gradient Weights image - Dark->light Edge Image
    darkLightEdgeImage = darkLightEdgeImage.t();
    std::vector<float> darkLightEdgeImage1D;
    if (darkLightEdgeImage.isContinuous()) {

        darkLightEdgeImage1D.assign((float*)darkLightEdgeImage.datastart, (float*)darkLightEdgeImage.dataend);
    }
    else {
        for (int i = 0; i < darkLightEdgeImage.rows; ++i) {
            darkLightEdgeImage1D.insert(darkLightEdgeImage1D.end(), darkLightEdgeImage.ptr<float>(i), darkLightEdgeImage.ptr<float>(i) + darkLightEdgeImage.cols);

        }
    }

    Mat darkLightEdgeImageRow1Result(imageEdgesIndexRow1.size(), CV_32F, Scalar(0));
    Mat darkLightEdgeImageRow2Result(imageEdgesIndexRow2.size(), CV_32F, Scalar(0));

    for (int i = 0; i < imageEdgesIndexRow1.cols; i++)
    {
        darkLightEdgeImageRow1Result.at<float>(0, i) = darkLightEdgeImage1D[imageEdgesIndexRow1.at<unsigned short>(i)];
        // cout << darkLightEdgeImage1D[imageEdgesIndexRow1.at<unsigned short>(0, i)] << endl;
    }
    
    for (int i = 0; i < imageEdgesIndexRow2.cols; i++)
    {
        darkLightEdgeImageRow2Result.at<float>(0, i) = darkLightEdgeImage1D[imageEdgesIndexRow2.at<unsigned short>(i)];
        //cout << lightDarkEdgeImageRow2Result.at<float>(0, i) << endl;
    }

    Mat darkLightGradientWeights = 2 - darkLightEdgeImageRow1Result - darkLightEdgeImageRow2Result;

    // Combine Two Weights
    Mat weights;
    weights.push_back(lightDarkGradientWeights);
    weights.push_back(darkLightGradientWeights);

    //Generate Weighting Matrix
    int matrixSize = imageWidth * imageHeight;
    float MIN_WEIGHT = 0.00001;
    
    outputW = GenerateWeightingMatrices(ImageSize, imageEdges, weights, columnEdges, MIN_WEIGHT);
    return outputW;
}

//6.2.1 Generate the Weighting Matrix
vector<WeightingMatrix> GenerateWeightingMatrices(Size MatrixSize, Mat ImageEdges, Mat weights, Mat ColumnEdges, float column_weight)
{
    cout << "GenerateWeightingMatrices" << endl;
    //Set the Column Weights
    int columnEdgesLength = ColumnEdges.cols;
    
    Mat onesMx = Mat::ones(1, columnEdgesLength, CV_32FC1);
    Mat column_Weights(1, columnEdgesLength, CV_32FC1);
    column_Weights = onesMx * column_weight;

    WeightingMatrix out;  
    //Normalize the Image Weights and combine all weights together
    for (int i = 0; i <2; i++)
    {
        Mat valuesRow = weights.row(i);

        // Normalize Weight Values
        Mat values = normalizeValues(valuesRow);
        Mat imageWeights = column_weight + values;
        
        //Combine the Image and Column Weights
        //Adding Node Pair Duplicates, Such that the Paths are Bidirectional
        Mat totalEdges;
        hconcat(ImageEdges, ColumnEdges, totalEdges);
       
        Mat totalWeights;
        hconcat(imageWeights, column_Weights, totalWeights);  

        //Conversion to Adjacency List Form
        out = ConvertDataFormatToGraph(totalEdges,totalWeights);       
        outputW.push_back(out);
    }
   
    return outputW;

}

//6.2.1.1 Normalization
Mat normalizeValues(Mat valuesRowWise)
{
    cout << "normalizeValues" << endl;
    int minValue = 0;
    int maxValue = 1;
    //Find old Minima and Maxima 
    double oldMinVal, oldMaxVal;
    minMaxLoc(valuesRowWise, &oldMinVal, &oldMaxVal);
    Mat values_Result = ((valuesRowWise - oldMinVal) / (oldMaxVal - oldMinVal) * (maxValue - minValue)) + minValue;
    return values_Result;
}


//6.2.1.2 Convert Edges & Weights to structure type that contain(Vertex,Index offset,Edges and Weights)
//WeightingMatrix ConvertDataFormatToGraph(Mat totalEdge, Mat totalWeights,WeightingMatrix W)
WeightingMatrix ConvertDataFormatToGraph(Mat totalEdge, Mat totalWeights)
{
    cout << "ConvertDataFormatToGraph" << endl;
    WeightingMatrix W1;
    Mat u = totalEdge.row(0);
    Mat v = totalEdge.row(1);
    
    //Mat to ushort array 
    vector<unsigned short>src(u.total() * u.channels());
    vector<unsigned short>dst(v.total() * v.channels());

    //Mat To Vector
    Mat flat = u.reshape(1, u.total() * u.channels());
    src = u.isContinuous() ? flat : flat.clone();

    Mat flat1 = v.reshape(1, v.total() * v.channels());
    dst = v.isContinuous() ? flat1 : flat1.clone();

    //Mat to float array 
    vector<float>weights(totalWeights.rows * totalWeights.cols * totalWeights.channels());

    if (totalWeights.isContinuous())
        weights.assign((float*)totalWeights.datastart, (float*)totalWeights.dataend);

    // Array of vectors, every vector represents
    // adjacency list of a vertex
    vector<pair<unsigned short, float>> graph[V];

    //Add Edge To Each Vertex
    for (int i = 0; i < 57356; i++)
    {
        addEdge(graph, src[i], dst[i], weights[i]);
    }

    int count = 0;
    for (int i = 0; i < V; i++)
    {

        //VertexArray[i] = count;
        //cout << i;
        for (int j = 0; j < graph[i].size(); j++)
        {
            W1.Edge.push_back(graph[i][j].first);
            W1.Weight.push_back(graph[i][j].second);
            count++;
            
        }
       
        W1.Index.push_back(count);
    }
    if ((W1.Index.at(0)) != 0)
    {
        W1.Index.insert(W1.Index.begin(), 0);
    }
   
   int IndexSize = W1.Index.size();
   //cout << "Index Size" << IndexSize << endl;
   for (int i = 0; i < IndexSize; i++)
   {
       W1.Vertex.push_back(i);
   }
   
   return W1;
}

//6.2.1.2.1 Create Edge and Weight - Pair wise
void addEdge(vector<pair<unsigned short, float>> graph[], unsigned short u, unsigned short v, float weight)
{
    //cout << "addEdge" << endl;
    graph[u].emplace_back(make_pair(v, weight));
}


//6.4 Get Region
vector<Point> getRegion(Size ImageSize, Mat topLayer, Mat bottomLayer, vector<int>invalidIndices)
{
    cout << "getRegion" << endl;
    //Replace all layer values outside the image range 
    topLayer.setTo(0, topLayer < 0);
    topLayer.setTo(ImageSize.height-1, topLayer > ImageSize.height-1);
    bottomLayer.setTo(0, bottomLayer < 0);
    bottomLayer.setTo(ImageSize.height - 1,bottomLayer > ImageSize.height-1);

    //vector<int>invalidImage(72 * 201, 0);
    Mat invalidImage(1, 202* 72, CV_32SC1, Scalar(0));

    if (!invalidIndices.empty())
    {
        for (int i = 0; i < invalidIndices.size(); i++)
        {
            invalidImage.at<int>(invalidIndices[i]) = 1;
        }                                 
    }

    //1D Vector to 2 Mat Array Conversion
    uchar* arr = (invalidImage.isContinuous()) ? invalidImage.data : invalidImage.clone().data;
    //uchar length = invalidImage.total() * invalidImage.channels();
    Mat invalidImage_Mat(72, 202, CV_32SC1, arr);

    //Limit the Layers by the invalid region
    for (int i = 0; i < 72; i++)
    {
        vector<int>nonZero;
        //Access Row wise
        Mat Image = invalidImage_Mat.row(i);
        //Invert it
        Image = 1 - Image;
        Image.convertTo(Image, CV_8U);
        //Find locations 
        vector<Point>locations;
        findNonZero(Image, locations);
        int topIndex = locations[0].x;
        //Access first element 
        int end = locations.size();
        int bottomIndex = locations[end - 1].x;

        if (topIndex != NULL)
        {
            topLayer.at<int>(i) = ((topLayer.at<int>(i) > topIndex) ? topLayer.at<int>(i) : topIndex);
            bottomLayer.at<int>(i) = ((bottomLayer.at<int>(i) > bottomIndex) ? bottomLayer.at<int>(i) : bottomIndex);
        }
    }

    //Get the indices of all pixels in between the two regions
    Mat regionImage(1, 202 * 72, CV_32S, Scalar(0));
    vector<int>yRegion;
    for (int iCol = 0; iCol < 72; iCol++)
    {
        //Close any vertical gaps that there may be in the region
        if (iCol < 72)
        {
            if (topLayer.at<int>(iCol) > bottomLayer.at<int>(iCol + 1))
            {
                topLayer.at<int>(iCol) = topLayer.at<int>(iCol + 1);
                bottomLayer.at<int>(iCol + 1) = bottomLayer.at<int>(iCol);
            }
            else if (bottomLayer.at<int>(iCol) < topLayer.at<int>(iCol + 1))
            {
                bottomLayer.at<int>(iCol) = bottomLayer.at<int>(iCol + 1);
                topLayer.at<int>(iCol + 1) = topLayer.at<int>(iCol);
            }
        }
        //Get the indices in the region
        for (int i = topLayer.at<int>(iCol); i < bottomLayer.at<int>(iCol) + 1; i++)
        {
            yRegion.push_back(i);
            //cout <<"yRegion:"<< yRegion[i] << endl;
        }


        Mat ones(1, yRegion.size(), CV_32S, Scalar(1));
        Mat value = iCol * ones;

        float indices;
        for (int i = 0; i < 202; i++)
        {
            indices = sub2ind(202, yRegion[i], value.at<int>(i), 72);
            //cout << indices << endl;
            if (isfinite(indices))
            {
                regionImage.at<int>(indices) = 1;
                //cout << regionImage.at<int>(indices)  << endl;
            }
            else
            {
                regionImage.at<int>(iCol) = 1;
            }
        }
    }

    //Take out any region indices that were specified as invalid
    //Remove Invalid indices from the region
    for (int i = 0; i < invalidIndices.size(); i++)
    {
        regionImage.at<int>(invalidIndices[i]) = 0;
    }
    //Get the Indices
    regionImage.convertTo(regionImage, CV_8U);
    vector<Point>regionIndices;
    findNonZero(regionImage, regionIndices);   
   /// for (int i = 0; i < regionIndices.size(); i++)
   // {
        cout << regionIndices.size() << endl;
    //}
   
    return regionIndices;
}

vector<int> cutRegion(Size ImageSize, vector<Point>regionIndices, WeightingMatrix weightEdgeIndexstruct) {
    cout <<"Cut Region" << endl;

    vector<int>cut(ImageSize.width, 0);
    vector<int>regionInd;
    int temp;

    for (int i = 0; i < regionIndices.size(); i++)
    {
        temp = regionIndices[i].x;
        regionInd.push_back(temp);
       
    }
    sort(regionInd.begin(), regionInd.end());
   
   //Make sure the region spans the entire width of the image
    int* x = (int*)malloc(V * sizeof(int)); 
    int* y = (int*)malloc(V * sizeof(int));
   
    for (int i = 0; i < 202 * 72; i++) {
        ind2sub(regionInd[i],202,72, x[i], y[i]);    
    }
    
    int startIndex = regionInd[0];
    cout << "start Index" << startIndex << endl;
    int endIndex;
  
    Mat xMat(1, 14544, CV_32S,x);
    Mat xImage(1, 14544, CV_8U, Scalar(0));
    xImage.setTo(Scalar(1),xMat==71);

    Mat xlocations;
    findNonZero(xImage, xlocations);
    endIndex = xlocations.at<Point>(0).x;
    cout << "endIndex" << endIndex << endl;
    list<int>coordinateIndices = {};

    //make sure the cooordinate indices span the entire width of the image
    if ((coordinateIndices.empty()) || coordinateIndices.front() > ImageSize.height-1)
    {
        coordinateIndices.push_back(startIndex);
    }

   int x1, y1;
   ind2sub(coordinateIndices.back(), 202, 72, x1, y1);
  
   if (coordinateIndices.empty() || x1 < ImageSize.width)
   {
       //endIndex = regionInd[endIndex];
       coordinateIndices.push_back(endIndex);
       
    }
   
   list <int> ::iterator it;
   unsigned int nCoordindates = coordinateIndices.size();
   
   //********************************************************************//
   // Restrict the graph cut region to the regionIndices Input
   //********************************************************************//
   //********************************************************************//
   // Calculate the best path that will connect the input coordinates
   //********************************************************************//

   int* pred = (int*)malloc(14544*sizeof(int));
   pred = runBellmanFordOnGPU(weightEdgeIndexstruct.Vertex, weightEdgeIndexstruct.Index, weightEdgeIndexstruct.Edge, weightEdgeIndexstruct.Weight);

  //**Find Shortest Path from Predecessor**//
   vector<int>Path;
   int start = coordinateIndices.front();
   int destination = coordinateIndices.back();
  
   Path = Pathshort(pred, start, destination);

  // reverset the Path & Insert the Destination at last 
   reverse(Path.begin(), Path.end());
   Path.insert(Path.end(), destination);

   //Index to Subscript
   for (int i = 0; i < Path.size(); i++)
   {
       ind2sub(regionInd[Path[i]], 202, 72,x[i],y[i]);    
   }

   Mat x_Mat(1, Path.size(), CV_32S, x);
  //Since the path contains muliple points per column, take the first point in every column as the path
   vector<Point>x_locations;  
   int index;

   for (int column = 0; column < ImageSize.width; column++)
   {
       if (column == 0)
       {
           Mat x_Image(1, Path.size(), CV_8U, Scalar(0));
           x_Image.setTo(Scalar(1), x_Mat == column);
           findNonZero(x_Image, x_locations);
           index = x_locations[x_locations.size()-1].x;
       }
       else
       {
           Mat x_Image1(1, Path.size(), CV_8U, Scalar(0));
           x_Image1.setTo(Scalar(1), x_Mat == column);
           findNonZero(x_Image1, x_locations);
           index = x_locations[0].x;
            
       }
       cut[column] = y[index];    
   }
  
   free(x);
   free(y);
   
  return cut;
  
}

vector<int> Pathshort(int* out_pred, int start, int destination)
{ vector<int>path;
   
    int j=0;

        for (int i = destination-1; i < destination; i++)
        {
            if (i != start)
                // printf("The Path is %d", i);
                j = i;
            do {
                j = out_pred[j];
                path.push_back(j);
                //printf("<-%d", j);
            } while (j != start);
            //printf("\n");
        }
        return path;
}


Mat diff(Mat Image)
{
    cv::Mat Kernely = (cv::Mat_<float>(2, 1) << -1, 1);
    cv::Mat dy;
    cv::filter2D(Image, dy, -1, Kernely, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
    // Remove padding and take the abs of the output
    dy = cv::Mat(dy, cv::Rect(0, 1, dy.cols, dy.rows - 1));
    //dy = cv::abs(dy);
    return dy;
}

void ind2sub(const int sub, const int cols, const int rows, int& row, int& col)
{
    row = sub / cols;
    col = sub % cols;
}


int sub2ind(const int row, const int col, const int cols, const int rows)
{
    return row * cols + col;
}

void Text_FileRead(const char* TextFileWithPath, unsigned short* preprocessing)
{

    ifstream dechirpFile;
    dechirpFile.open(TextFileWithPath);
    char buff[2] = { 0 };
    int cnt = 0;
    while (dechirpFile >> buff)
    {
        preprocessing[cnt] = atoi(buff);
        cnt++;
    }
    dechirpFile.close();

}

template <typename T>
void writeMatToFile(cv::Mat m, const char* filename)
{
    ofstream fout(filename);

    if (!fout)
    {
        cout << "File Not Opened" << endl;  return;
    }

    for (int i = 0; i < m.rows; i++)
    {
        for (int j = 0; j < m.cols; j++)
        {
            fout << m.at<T>(i, j) << "\t";
        }
        fout << endl;
    }

    fout.close();
}


float modulusValue(float a, float b)
{
    float mod = a - floor(a / b) * b;
    return mod;
}

Mat resampleLayers(Mat layers, Size originalSize, Size newSize)
{
    Mat SmoothResult(4, 500, CV_64F);

    Mat Final_Index(4, 500, CV_64F);

    //Calculate the Scaling needed to upsample
    float width, height;
    width = (float)newSize.width / (float)originalSize.width;
    height = (float)newSize.height / (float)originalSize.height;

    // cout << height << endl;
     //Upsample in Y Direction
    layers = layers * height;


    Mat layers1;
    layers.convertTo(layers1, CV_32S);


    //UpSample Each Layer in the X-Direction Using Interpolation
    int nLayers = layers1.rows;
    if (width < 1)
    {

    }
    else
    {
        int newWidth = newSize.width;

        Mat x(1, newWidth, CV_32F);
        for (int i = 0; i < newWidth; i++)
            x.at<float>(0, i) = i; // set the column in row 0

        Mat y = layers1.clone();


        Mat layers_Rescaled(nLayers, newWidth, CV_32F, Scalar(0));

        Mat ind(1, originalSize.width, CV_32F, Scalar(0));

        int itr = 0, value = 0;
        while (itr < originalSize.width)
        {
            ind.at<float>(0, itr) = value;
            value = value + round(width);
            itr = itr + 1;
        }
        ind.at<float>(0, originalSize.width - 1) = newWidth - 1;

        
        y.convertTo(y, CV_32F);
        for (int i = 0; i < nLayers; i++)
        {
            for (int j = 0; j < originalSize.width; j++)
            {
                layers_Rescaled.at<float>(i, ind.at<float>(0, j)) = y.at<float>(i, j);

            }
        }

        //Loop Through Each Layer
        for (int i = 0; i < 4; i++)
        {
            //Mask - Process
            Mat layer = layers_Rescaled.row(i);
           // cout << "layers_Rescaled" << layers_Rescaled.row(i) << endl;
            layer.setTo(1, layer > 0);
            layer.convertTo(layer, CV_8UC1);

            //Valid Index
            Mat validInd;
            findNonZero(layer, validInd);

            //InValid Index
            Mat invalidInd;
            layer = 1 - layer;
            layer.setTo(1, layer > 0);
            findNonZero(layer, invalidInd);
           
            //X - Known Points
            Mat Index_valid[2];
            split(validInd, Index_valid);

            Mat Index_unKnown[2];
            split(invalidInd, Index_unKnown);

            Index_valid[0].convertTo(Index_valid[0], CV_64F);
            transpose(Index_valid[0], Index_valid[0]);

            cv::Mat flat = Index_valid[0].reshape(1, Index_valid[0].total() * Index_valid[0].channels());
            std::vector<double> X_IndexValid = Index_valid[0].isContinuous() ? flat : flat.clone();

            cv::Mat flat1 = y.row(i).reshape(1, y.row(i).total() * y.row(i).channels());
            std::vector<double> Y_Known = y.row(i).isContinuous() ? flat1 : flat1.clone();

            //std::memcpy(Test.data, Y_Known, 70 * sizeof(double));
            Index_unKnown[0].convertTo(Index_unKnown[0], CV_64F);

            transpose(Index_unKnown[0], Index_unKnown[0]);

            cv::Mat flat2 = Index_unKnown[0].row(0).reshape(1, Index_unKnown[0].row(0).total() * Index_unKnown[0].row(0).channels());
            std::vector<double> X_unKnown = Index_unKnown[0].row(0).isContinuous() ? flat2 : flat2.clone();

            // Mat Test(1, 500, CV_64F);
             //memcpy(Test.data, X_unKnown.data(), X_unKnown.size() * sizeof(double));

             //Spline Interpolation

            tk::spline s(X_IndexValid, Y_Known);
            //double *result = (double*)malloc(500*sizeof(double));
            vector<double>result;
            vector<double>out;

           for (int i = 0; i < 500; i++)
            {
                double x1 = x.at<float>(0, i);
                double output = s(x1);
                result.push_back(round(output));
            }

           std::memcpy(Final_Index.row(i).data, result.data(), 500 * sizeof(double));
          
            //Layer Smoothing Function
            out = sg_smooth(result, 50, 3);
  
            memcpy(SmoothResult.row(i).data, out.data(), out.size() * sizeof(double));
        }

    }
    return SmoothResult;
}



   // cout << weights.type() << endl;
//ofstream file;
//file.open("TextData.txt");
//if (file.is_open())
//{
//    for (int i = 0; i < 202 * 72; i++)
//    {
//        file << pred[i] << " ";
//    }
//    file.close();
//}
//const char* fileName = "C:\\Users\\Mohan\\source\\repos\\GraphCutSegmentationProject\\TEST.txt";
//writeMatToFile<unsigned short>(totalEdges, fileName);
//Mat check(1, outputW.at(0).Weight.size(), CV_32FC1);
//memcpy(check.data, outputW.at(0).Weight.data(), outputW.at(0).Weight.size() * sizeof(float));
//const char* fileName = "C:\\Users\\Mohan\\source\\repos\\CUDA_GraphCutSegmentation\\TEST.txt";
//writeMatToFile<float>(check, fileName);
//
//Mat check1(1, outputW.at(1).Weight.size(), CV_32FC1);
//memcpy(check1.data, outputW.at(1).Weight.data(), outputW.at(1).Weight.size() * sizeof(float));
//const char* fileName1 = "C:\\Users\\Mohan\\source\\repos\\CUDA_GraphCutSegmentation\\TEST1.txt";
//writeMatToFile<float>(check1, fileName1);
//
//cout << "finished Iteration" << endl;

//vector<double>Path1;
//ifstream file;
//double x11;
//file.open("path.txt");
//if (file.is_open())
//{
//   while(file >> x11)
//    {
//      Path1.push_back(x11);
//    }  
//}
//else
//{
//    cout << "File is not open" << endl;
//}
//file.close();
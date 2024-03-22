#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <cmath>
using namespace cv;
using namespace std;

extern "C" 
{
    vector<int> DCTfunc(cv::Mat img, vector<vector<vector<vector<float>>>> mask, int size=8)
    {
        // printf("We good here!\n");
        vector<int> staticVector(10, 0);
        cv::Size sz = img.size();
        int col = sz.width;
        int row = sz.height;
        int colFactor = col/size;
        int rowFactor = row/size;
        int ampFactor = pow(10, 8);

        for(int c = 0; c < colFactor; c++)
        {
            for(int r = 0; r < rowFactor; r++)
            {
                float tmp = 0;
                for(int i=0; i < size; i++)
                {
                    for(int j=0; j < size; j++)
                    {
                        tmp = 0;
                        for(int x=0; x < size; x++)
                        {
                            for(int y=0; y < size; y++)
                            {
                                int xIndex = size*c + x;
                                int yIndex = size*r + y;
                                if(xIndex > sz.width || yIndex > sz.height)
                                {
                                    printf("img size: %d x %d\n", sz.width, sz.height);
                                    printf("xIndex: %d, yIndex: %d\n", xIndex, yIndex);
                                    printf("colFactor: %d, rowFactor: %d\n", colFactor, rowFactor);
                                    printf("c: %d, r: %d\n", c, r);
                                    printf("Broken!\n");
                                    return staticVector;
                                }
                                tmp += (mask[i][j][x][y]) * (img.at<uchar>(xIndex, yIndex)) * ampFactor;
                            }
                        }

                        tmp = abs(tmp);
                        std::string str = std::to_string(tmp);
                        
                        if (!str.empty()) {
                            char firstDigit = str[0];
                            int firstInt = firstDigit - '0';
                            staticVector[firstInt] += 1;
                            // std::cout << "First digit of " << tmp << " is: " << firstDigit << std::endl;
                        } else {
                            std::cerr << "Error: Unable to get the first digit of the float number." << std::endl;
                            continue;
                        }
                    }
                }

            }
        }
        return staticVector;
    }
    vector<vector<vector<vector<float>>>> initMask(const int size)
    {
        float res = 0.0f;
        vector<vector<vector<vector<float>>>> mask(size, vector<vector<vector<float>>>(size, vector<vector<float>>(size, vector<float>(size, 1.0))));
        float para = 1/(2*sqrt(2*size));
        float OrthogonalValue = 1/sqrt(2);
        float Ci = OrthogonalValue;
        float Cj = OrthogonalValue;
        for(int i = 0; i < size; i++)
        {
            if(i==0)
            {
                Ci = OrthogonalValue;
            }
            else
            {
                Ci = 1;
            }
            for(int j = 0; j < size; j++)
                    {
                        if(j==0)
                        {
                            Cj = OrthogonalValue;
                        }
                        else
                        {
                            Cj = 1;
                        }
                        for(int x = 0; x < size; x++)
                        {
                            for(int y = 0; y < size; y++)
                            {
                                mask[i][j][x][y] = (para * Ci * Cj * cos((2*x+1)*i*M_PI / (2*size)) * cos((2*y+1)*j*M_PI / (2*size)));
                                res += mask[i][j][x][y];
                            }
                        }
                    }  
        }
        return mask;

    }

    // 宣告 DetectC 函數
    float DetectC(const char *imgPath) 
    {
        printf("imgPath: %s\n",imgPath);
        cv::Mat img = cv::imread(imgPath);
        if (img.empty())  
        {
            fprintf(stderr, "Could not open or find the image\n");
        }
        cv::Size sz = img.size();
        int windowSize = 8;
        int imgWidth = sz.width;
        int imgHeight = sz.height;
        int reWidth = imgWidth % windowSize;
        int reHeight = imgHeight % windowSize;
        imgWidth = imgWidth - reWidth;
        imgHeight = imgHeight - reHeight;

        cv::Mat reSizeimg;
        cv::resize(img,reSizeimg,cv::Size(imgWidth,imgHeight),0,0,cv::INTER_LINEAR);
        // cv::imshow("reSizeimg", reSizeimg);
        // cv::waitKey(0);
        vector<vector<vector<vector<float>>>> mask = initMask(windowSize);
        vector<int> staticVector = DCTfunc(reSizeimg, mask);
        float result = 0;
        for (int n=1; n < (int)staticVector.size(); n++)
        {
            float pn = log10((n+1)/n);
            result += abs(pn - (staticVector[n]/(imgWidth * imgHeight)));
            // printf("%d: %d\n", n, staticVector[n]);
        }

        // printf("size of image: %d x %d\n", imgWidth, imgHeight);
        // printf("Result in CPP: %.3f\n", result); 
        return result;
    }
}

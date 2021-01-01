#include <cstdio>
#include "opencv2/video/tracking.hpp"
#include "imageio.h"
#include "jzq.h"

using namespace std;

static A2V2f flowToA2V2f(const cv::Mat_<cv::Point2f>& flow)
{
    A2V2f F(flow.cols,flow.rows);

    for (int i = 0; i < flow.rows; ++i)
    {
        for (int j = 0; j < flow.cols; ++j)
        {
            const cv::Point2f u = flow(i, j);
            F(j,i) = V2f(u.x,u.y);
        }
    }
    
    return F;    
}

static cv::Mat_<unsigned char> a2f_to_mat8(const A2f& I)
{
    cv::Mat_<unsigned char> mat(I.height(),I.width());
    
    for(int y=0;y<mat.rows;y++)
    for(int x=0;x<mat.cols;x++)
    {
        mat(y,x) = clamp(I(x,y),0.0f,1.0f)*255.0f;
    }

    return mat;  
}

bool disflow(const std::string& fromImageFileName,
             const std::string& toImageFileName,
             const std::string& outputFlowFileName)
{
    const A2f fromImage = imread<float>(fromImageFileName); if (fromImage.empty()) { printf(spf("Error: missing file %s\n",fromImageFileName.c_str()).c_str()); return false; }
    const A2f   toImage = imread<float>(toImageFileName);   if (  toImage.empty()) { printf(spf("Error: missing file %s\n",toImageFileName.c_str()).c_str()); return false; }

    cv::Mat frame0 = a2f_to_mat8(fromImage);
    cv::Mat frame1 = a2f_to_mat8(toImage);

    cv::Mat_<cv::Point2f> flow;
        
    cv::Ptr<cv::DenseOpticalFlow> dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);
  
    dis->calc(frame0, frame1, flow);
            
    const A2V2f F = flowToA2V2f(flow);

    a2write(F,outputFlowFileName);

    return true;
}

int main(int argc,const char ** argv)
{
    if (argc!=4)
    {
        printf("usage: %s from-image.png to-image.png output-flow.A2V2f\n",argv[0]); return 1;
    }

    if (!disflow(argv[1],argv[2],argv[3])) { return 1; }

    return 0;
}

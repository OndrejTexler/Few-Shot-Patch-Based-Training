#define _CRT_SECURE_NO_DEPRECATE
#include <cstdio>

#include "jzq.h"
#include "imageio.h"
#include <vector>
#include <string>
#include <iostream>

#define FOR(A,X,Y) for(int Y=0;Y<A.height();Y++) for(int X=0;X<A.width();X++)

using namespace std;

double gaussKernel(float x, float mean, float sigma)
{
	double sigmaSq = 2 * sigma * sigma;

	return exp(-(pow((x - mean), 2.0)) / (sigmaSq));
}



V4uc bilateral(V4uc & centralPixel , vector<V4uc> &leftPixels, vector<V4uc> &rightPixels, int range, int sgmDiv) {

  int leftSize = leftPixels.size();
  int rightSize = rightPixels.size();
  const int len = leftSize + rightSize + 1;

  /************* KERNELS **********************/

  /**************    time    *******************/
  vector<double> distKernel(len);
  double sigma = range / 3;
  double mean = leftPixels.size();

  int idx = 0;
  for (idx; idx < len; ++idx) {

    distKernel[idx] = gaussKernel(idx, mean, sigma);

  }

  /***********    color     ********************/
  vector<double> colorKernel(len);
  sigma = sqrt(pow(255.0, 2) * 3) / sgmDiv;
  mean = 0.0;
  double sumWeight = 0.0;
  idx = 0;

  /***  left  ***/
  for (int leftIdx=0; leftIdx < leftSize; ++leftIdx) {

    // RGB distance
    float diff = sqrt( pow((centralPixel[0] - leftPixels[leftIdx][0]), 2) + pow((centralPixel[1] - leftPixels[leftIdx][1]) , 2) + pow((centralPixel[2] - leftPixels[leftIdx][2]),2) );
    colorKernel[idx] = gaussKernel(diff, mean, sigma);
    sumWeight += colorKernel[idx] * distKernel[idx];
    idx++;

  }

  /***  image  ***/
  colorKernel[idx] = gaussKernel(0.0, mean, sigma);
  sumWeight += colorKernel[idx] * distKernel[idx];
  idx++;

  /***  right   ***/
  for (int rightIdx = 0; rightIdx < rightSize; ++rightIdx) {

    // RGB distance
    float diff = sqrt( pow((centralPixel[0] - rightPixels[rightIdx][0]), 2) + pow((centralPixel[1] - rightPixels[rightIdx][1]), 2) +  pow((centralPixel[2] - rightPixels[rightIdx][2]), 2));
    colorKernel[idx] = gaussKernel(diff, mean, sigma);
    sumWeight += colorKernel[idx] * distKernel[idx];
    idx++;

  }

  /************* WEIGHTED COLORS   *******************/
  float red = 0.0; float green = 0; 0; float blue = 0.0;

  /****** left *******/
  idx = 0;
  for (int l = 0; l < leftSize; l++) {

    red += leftPixels[l][0] * distKernel[idx] * colorKernel[idx];
    green += leftPixels[l][1] * distKernel[idx] * colorKernel[idx];
    blue += leftPixels[l][2] * distKernel[idx] * colorKernel[idx];
    idx++;

  }

  /******* image ********/
  red += centralPixel[0] * distKernel[idx] * colorKernel[idx];
  green += centralPixel[1] * distKernel[idx] * colorKernel[idx];
  blue += centralPixel[2] * distKernel[idx] * colorKernel[idx];
  idx++;

  /******* right *********/
  for (int r = 0; r < rightSize; r++) {

    red += rightPixels[r][0] * distKernel[idx] * colorKernel[idx];
    green += rightPixels[r][1] * distKernel[idx] * colorKernel[idx];
    blue += rightPixels[r][2] * distKernel[idx] * colorKernel[idx];
    idx++;
  }

  /*******NORMALIZATION************/
  red /= sumWeight;
  green /= sumWeight;
  blue /= sumWeight;


  V4uc pixel(red, green, blue, centralPixel[3]);

  return pixel;


}

template<int N,typename T>
Vec<N,T> sampleBilinear(const Array2<Vec<N,T>>& I,const V2f& xy)
{
  const int ix = xy(0);
  const int iy = xy(1);

  const float s = xy(0)-ix;
  const float t = xy(1)-iy;

  return Vec<N,T>((1.0f-s)*(1.0f-t)*Vec<N,float>(I(clamp(ix  ,0,I.width()-1),clamp(iy  ,0,I.height()-1)))+
                  (     s)*(1.0f-t)*Vec<N,float>(I(clamp(ix+1,0,I.width()-1),clamp(iy  ,0,I.height()-1)))+
                  (1.0f-s)*(     t)*Vec<N,float>(I(clamp(ix  ,0,I.width()-1),clamp(iy+1,0,I.height()-1)))+
                  (     s)*(     t)*Vec<N,float>(I(clamp(ix+1,0,I.width()-1),clamp(iy+1,0,I.height()-1))));
};

int findPixel (vector<A2V4uc> &images, vector<A2V2f> &flows, vector<V4uc> &pixels, V2f xy, bool left, int depth) {

  if (images.size() == pixels.size()) {
    return 1; // mame vsechny pixly, koncime rekurzicku
  }

  V4uc pixel;
  V2f flow;
  V2f xyNew;

  // pro leve pole obrazku musim jet od konce vektoru, protoze ten nejblizsi obrazek k centralnimu je na konci vektoru
  if (left){
    flow = sampleBilinear(flows[flows.size() - (depth+1)], xy);
    xyNew = xy + flow; // TODO  + nebo -
    pixel = sampleBilinear(images[images.size()-(depth +1)], xyNew);
  }
  // pro prave pole obrazku jedu normalne od zacatku vektoru, tam je nejblizci obrazek k centru
  else {
    flow = sampleBilinear(flows[depth], xy);
    xyNew = xy + flow; // TODO + nebo -
    pixel = sampleBilinear(images[depth], xyNew);

  }
	
  pixels.push_back(pixel);

  depth ++;
  findPixel(images, flows, pixels, xyNew, left, depth);

  return 1;
}


int main(int argc,char** argv)
{  
  // argv 1 - adresa obrazku, ktery rozmazavame, argv 2 - pocet fotek vpred a vzad, se kteryma se to rozmazne, argv3 - delitel sigmy, argv [4] - kam ulozit vysledek
  string imgNameFormat = argv[1];
  string flowFwdNameFormat = argv[2];
  string flowBwdNameFormat = argv[3];
  int thisFrame = atoi(argv[4]);

  A2V4uc image = imread<V4uc>(spf(imgNameFormat.c_str(),thisFrame));
  A2V4uc output(image.width(), image.height());

  int imgNumber = thisFrame;

  string range = argv[5];
  int rangeNum = std::stoi(range);

  string sgm = argv[6];
  int sgmDiv = std::stoi(sgm);

  vector<A2V4uc> leftImgs;
  vector<A2V4uc> rightImgs;
  vector<A2V2f> leftFlows;
  vector<A2V2f> rightFlows;

/*
  vector<int> namesIMGleft;
  vector<int> namesIMGright;
  vector<int> namesFLWright;
  vector<int> namesFLWleft;
*/

  // nacteni vsech okolnich obrazku a optickych toku
  for (int i=1; i<=rangeNum; i++) {



    int imLeft = imgNumber -(rangeNum-i+1);
    //A2V4uc tmpLeft = imread<V4uc>(path + to_string(imLeft) + ".png");
    A2V4uc tmpLeft = imread<V4uc>(spf(imgNameFormat.c_str(),imLeft));
    //A2V2f tmpLflow = a2read<V2f>(path + "\\flow-fwd\\" + to_string(imLeft+1) + ".A2V2f");
    A2V2f tmpLflow = a2read<V2f>(spf(flowFwdNameFormat.c_str(),imLeft+1));

    if (!tmpLeft.empty()){

      leftImgs.push_back(tmpLeft);
      //namesIMGleft.push_back(imLeft);

    }
    if (!tmpLflow.empty()){

      leftFlows.push_back(tmpLflow);
      //namesFLWleft.push_back(imLeft+1);
    }


    int imRight = imgNumber + i;
    //A2V4uc tmpRight = imread<V4uc>(path + to_string(imRight) + ".png"); 
    A2V4uc tmpRight = imread<V4uc>(spf(imgNameFormat.c_str(),imRight));
    //A2V2f tmpRflow = a2read<V2f>(path + "\\flow-bwd\\" + to_string(imRight-1) + ".A2V2f");
    A2V2f tmpRflow = a2read<V2f>(spf(flowBwdNameFormat.c_str(),imRight-1));

    if (!tmpRight.empty()){

      rightImgs.push_back(tmpRight);
      //namesIMGright.push_back(imRight);

    }

    if (!tmpRflow.empty()){

       rightFlows.push_back(tmpRflow);
       //namesFLWright.push_back(imRight-1);
    }

  }

  printf("%d %d %d %d %d\n",imgNumber,leftImgs.size(),leftFlows.size(),rightImgs.size(),rightFlows.size());
  

/*
  cout << "Obrazky left: ";
  for (int i=0; i<namesIMGleft.size(); ++i) {

    cout << namesIMGleft[i] << ", ";

  }
  cout << endl; 
  cout<< "Obrazky right: ";
  for (int i=0; i<namesIMGright.size(); ++i) {

    cout << namesIMGright[i] << ", ";

  }
  cout << endl;

  cout << "Flows left: ";
  for (int i=0; i<namesFLWleft.size(); ++i) {

      cout << namesFLWleft[i] << ", ";

  }
  cout << endl; 
  cout<< "Flows right: ";
  for (int i=0; i<namesFLWright.size(); ++i){

      cout << namesFLWright[i] << ", ";
  }

  cout << endl;
*/
  // for loop pre vsechny pixly
  
  FOR(output,x,y) {

    vector <V4uc> leftPixels;
    vector <V4uc> rightPixels;

    // hledam leve a prave pixly
    int depth = 0;
    bool left = 1;
    findPixel ( leftImgs, leftFlows, leftPixels, V2f(x,y), left, depth); 
    depth = 0;
    left=0;
    findPixel (rightImgs, rightFlows, rightPixels, V2f(x,y), left, depth);

    V4uc pixel = bilateral(image(x,y), leftPixels, rightPixels, rangeNum, sgmDiv);
    output(x, y) = V4uc(pixel[0], pixel[1], pixel[2], pixel[3]);

  }

  imwrite(output,argv[7]);
 

  return 1;
}

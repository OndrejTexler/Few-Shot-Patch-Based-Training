#include <cmath>
#include <vector>

#include "jzq.h"
#include "imageio.h"

#define FOR(A,X,Y) for(int Y=0;Y<A.height();Y++) for(int X=0;X<A.width();X++)

template<int N,typename T>
Vec<N,T> sampleBilinear(const Array2<Vec<N,T>>& I,const V2f& x)
{
  const int w = I.width();
  const int h = I.height();

  const int ix = std::floor(x(0));
  const int iy = std::floor(x(1));

  const float s = x(0)-ix;
  const float t = x(1)-iy;

  return Vec<N,T>((1.0f-s)*(1.0f-t)*Vec<N,float>(I(clamp(ix  ,0,w-1),clamp(iy  ,0,h-1)))+
                  (     s)*(1.0f-t)*Vec<N,float>(I(clamp(ix+1,0,w-1),clamp(iy  ,0,h-1)))+
                  (1.0f-s)*(     t)*Vec<N,float>(I(clamp(ix  ,0,w-1),clamp(iy+1,0,h-1)))+
                  (     s)*(     t)*Vec<N,float>(I(clamp(ix+1,0,w-1),clamp(iy+1,0,h-1))));
}

#include "thinks/poisson_disk_sampling/poisson_disk_sampling.h"

std::vector<V2f> genPts(const A2uc& M,float radius)
{
  namespace pds = thinks::poisson_disk_sampling;
  
  const auto x_min = std::array<float, 2>{{0.f,0.f}};
  const auto x_max = std::array<float, 2>{{float(M.width()-1),float(M.height()-1)}};
  const auto samples = pds::PoissonDiskSampling(radius, x_min, x_max);
  
  std::vector<V2f> O;
  for (const auto& sample : samples) {
    int x = sample[0];
    int y = sample[1];
    if (x>=0 && x<M.width() && y>=0 && y<M.height())
    {
      if (M(x,y)>64)
      {
        O.push_back(V2f(x,y));
      }
    }
  }
  
  return O;
}

const float SQR(float x) { return x*x; }

A2V2f readFlow(const std::string& fileName)
{
  A2V2f F = a2read<V2f>(fileName);
  if (F.empty()) { printf("error: failed to read flow %s\n",fileName.c_str()); exit(1); }
  return F;
}

void drawPts(A2V3f& O,const V2i& size,const std::vector<V2f>& Ps,const float sigma)
{  
  for(int i=0;i<Ps.size();i++)
  {
      const V2f p = Ps[i];
      const int r = 3*sigma;
      const V3f color = V3f(rand()%255,rand()%255,rand()%255)/V3f(255,255,255);
      for (int y=p[1]-r;y<=p[1]+r;y++)
      for (int x=p[0]-r;x<=p[0]+r;x++)

      if (x>=0 && x<O.width() && y>=0 && y<O.height())
      {         
        O(x,y) = lerp(O(x,y),color,std::exp(-(SQR(float(x)-p[0])+SQR(float(y)-p[1]))/(sigma*sigma)));
      }
  }
}

int main(int argc,char** argv)
{
  if (argc<11)
  {
    printf("%s mask\\%%03d.png flow-fwd\\%%03d.A2V2f flow-bwd\\%%03d.A2V2f firstframe lastframe numkeys key1 key2 ... keyN radius sigma output\\%%03d.png",argv[0]);
    return 1;
  }
  
  int argi=1;
  const char* maskFileFormat = argv[argi++];

  const char* flowFwdFormat = argv[argi++];
  const char* flowBwdFormat = argv[argi++];

  const int frameFirst = atoi(argv[argi++]);
  const int frameLast  = atoi(argv[argi++]);
  
  const int numKeys   = atoi(argv[argi++]);
  
  std::vector<int> keys;
  for(int i=0;i<numKeys;i++) { keys.push_back(atoi(argv[argi++])); }

  const float radius = atof(argv[argi++]);
  const float sigma  = atof(argv[argi++]);

  const char* outputFormat  = argv[argi++];

  printf("maskFileFormat: %s\n",maskFileFormat);  
  printf("flowFwdFormat:  %s\n",flowFwdFormat);
  printf("flowBwdFormat:  %s\n",flowBwdFormat);  
  printf("frameFirst:     %d\n",frameFirst);
  printf("frameLast:      %d\n",frameLast);
  printf("numKeys:        %d\n",numKeys);
  for(int i=0;i<numKeys;i++)
  {
    printf("keys[% 4d]:     %d\n",i,keys[i]);
  }
  printf("radius:         %.1f\n",radius);
  printf("sigma:          %.1f\n",sigma);
  printf("outputFormat:   %s\n",outputFormat);  

  Array2<std::vector<V2f>> pts(numKeys,frameLast+1);

  V2i sizeO;
  //std::vector<V3f> colors(8192);
  //for(int j=0;j<8192;j++) { colors[j] = V3f(rand()%255,rand()%255,rand()%255); }

  for(int k=0;k<keys.size();k++)
  {
    const int frameKey = keys[k];
    const A2uc M = imread<unsigned char>(spf(maskFileFormat,frameKey).c_str());
    if (M.empty()) { printf("ERROR: failed to read mask %s\n",spf(maskFileFormat,frameKey).c_str()); return 1; }  
    sizeO = size(M);

    const std::vector<V2f> keyPs = genPts(M,radius);    

    pts(k,frameKey) = keyPs;
    //imwrite(drawPts(size(M),keyPs,colors,sigma),spf(outputFormat,frameKey));

    if (frameLast>frameKey)
    {
      std::vector<V2f> Ps = keyPs;
      for(int frame=frameKey+1;frame<=frameLast;frame++)
      {
        const A2V2f F = a2read<V2f>(spf(flowBwdFormat,frame-1));
        for(int i=0;i<Ps.size();i++) { Ps[i] = Ps[i] + sampleBilinear(F,Ps[i]); }      
        pts(k,frame) = Ps;
        //imwrite(drawPts(size(M),Ps,colors,sigma),spf(outputFormat,frame));
      }
    }

    if (frameFirst<frameKey)
    {
      std::vector<V2f> Ps = keyPs;
      for(int frame=frameKey-1;frame>=frameFirst;frame--)
      {
        const A2V2f F = a2read<V2f>(spf(flowFwdFormat,frame+1));
        for(int i=0;i<Ps.size();i++) { Ps[i] = Ps[i] + sampleBilinear(F,Ps[i]); }      
        pts(k,frame) = Ps;
        //imwrite(drawPts(size(M),Ps,colors,sigma),spf(outputFormat,frame));
      }
    }
  }

  for(int frame=frameFirst;frame<=frameLast;frame++)
  {    
    A2V3f O(sizeO);
    fill(&O,V3f(0,0,0));
    srand(1337);
    for(int k=keys.size()-1;k>=0;k--)
    {
      const int frameKey = keys[k];
      drawPts(O,sizeO,pts(k,frame),sigma);
    }
    imwrite(O,spf(outputFormat,frame));
  }

  return 1;
}

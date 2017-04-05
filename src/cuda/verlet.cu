#include <cstdio>


#define CHUNK_SIZE 1024
#define EPS2 1

extern "C" {

__device__ double3 bodyBodyInteraction(double3 bi, double3 bj,int j, double3 ai, short * mergeCounter, short * devMergeList, int N){
  if(j >=N){
    return;
  }
    double3 r;
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = 0.0;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    if(sqrt( r.x * r.x + r.y * r.y) < (pow(bi.z, 1.0 / 3.0) +  pow(bj.z, 1.0 / 3.0))* 0.6){
      mergeCounter[0]++;
      devMergeList[N * gtid] = mergeCounter[0];
      devMergeList[N * gtid +  mergeCounter[0]] = j;
    }


    double distSqr = r.x * r.x + r.y * r.y + EPS2;
    double distSixth = distSqr * distSqr * distSqr;

    double invDistCube = rsqrt(distSixth);

    double s = bj.z * invDistCube;

    ai.x += r.x * s;
    ai.y += r.y * s;

    return ai;

}

__device__ double3 tile_calculation(double3 myPosition, double3 accel, double3 * shPosition , short * mergeCounter, short * devMergeList, int N, int j){
    int i;


    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    for(i = 0; i < blockDim.x; i++){
        accel = bodyBodyInteraction(myPosition, shPosition[i],(short)(j + i),  accel, mergeCounter, devMergeList, N);
    }

    return accel;
}

__global__
void leapfrog_integrator(void* devX, void* devA, void * devPreviousV, short * devMergeList ,int N){

      int gtid = blockIdx.x * blockDim.x + threadIdx.x;
      if(gtid >= N){
        return;
      }
      short mergeCounter[]={0};
        devMergeList[N * gtid] = 0;


    __shared__ double3 shPosition [CHUNK_SIZE];
    double3 *globalX = (double3 *) devX;
    double3 *globalA = (double3 *) devA;
    double2 *previous_v = (double2 *) devPreviousV;
    double3 myPosition;





    int i, tile;
    double3 acc = {0.0, 0.0, 0.0};

    myPosition = globalX[gtid];
    for(i = 0, tile = 0; i < N; i += CHUNK_SIZE, tile++){
        int idx = tile * blockDim.x + threadIdx.x;
        shPosition[threadIdx.x] = globalX[idx];

        __syncthreads();
          acc = tile_calculation(myPosition, acc, shPosition, mergeCounter, devMergeList, N, i);
        __syncthreads();

    }

    double interval = 1.0 / 100;

    double3 acc3 = {acc.x, acc.y, globalX[gtid].z};
    globalA[gtid] = acc3;
    double2 v;

      v.x = previous_v[gtid].x + interval * globalA[gtid].x;
      v.y = previous_v[gtid].y + interval * globalA[gtid].y;
      previous_v[gtid] = v;
      globalX[gtid].x += interval * v.x;
      globalX[gtid].y += interval * v.y;





}

}

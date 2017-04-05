#include "cuda.h"
#include "findUnion.h"
#include "graphic.h"
#include "verlet.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>

#define CHUNK_SIZE 1024

#define MAX_MASS 100
#define MIN_MASS 50
#define USE_CUDA
#define PI 3.14159265

using namespace std;

const int RADIOUS = 5;
CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction leapfrog_integrator_cuda;
CUdeviceptr devX, devA, devPreviousV, devMergeList;
short* mergeList;
int blocks_per_grid;
int threads_per_block;
/// previous velocity
double2 previous_v[MAX_N] = {};

int flags;

int parent[MAX_N];
int unionRank[MAX_N];
bool toMerge[MAX_N];

inline void initCuda() {
  cuInit(0);
  CUresult res;

  res = cuDeviceGet(&cuDevice, 0);
  if (res != CUDA_SUCCESS) {
    printf("failure 1\n");
    exit(1);
  }

  res = cuCtxCreate(&cuContext, 0, cuDevice);
  if (res != CUDA_SUCCESS) {
    printf("failure 2\n");
    exit(1);
  }

  res = cuModuleLoad(&cuModule, "src/cuda/verlet.ptx");
  if (res != CUDA_SUCCESS) {
    printf("falure 3\n");
    exit(1);
  }

  res = cuModuleGetFunction(&leapfrog_integrator_cuda, cuModule,
                            "leapfrog_integrator");

  if (res != CUDA_SUCCESS) {
    printf("falure 4\n");
    exit(1);
  }

  return;
}

void cudaMemoryRegister(Body *objsX, Body *objsA, short *mergeList, int N) {
  int res;

  res = cuMemHostRegister(objsX, sizeof(Body) * N, 0);

  if (res != CUDA_SUCCESS) {
    printf("falure 5 %d N: %d, sizeof: %d\n", res, N, sizeof(Body));
    exit(1);
  }
  res = cuMemHostRegister(objsA, sizeof(Body) * N, 0);

  if (res != CUDA_SUCCESS) {
    printf("falure 6\n");
    exit(1);
  }

  res = cuMemAlloc(&devX, sizeof(Body) * N);

  if (res != CUDA_SUCCESS) {
    printf("falure 7\n");
    exit(1);
  }
  res = cuMemAlloc(&devA, sizeof(Body) * N);

  if (res != CUDA_SUCCESS) {
    printf("falure 8\n");
    exit(1);
  }

  res = cuMemAlloc(&devPreviousV, sizeof(double2) * N);

  if (res != CUDA_SUCCESS) {
    printf("falure 9\n");
    exit(1);
  }

  res = cuMemAlloc(&devMergeList, sizeof(short) * N * (N + 1));

  if (res != CUDA_SUCCESS) {
    printf("falure 9\n");
    exit(1);
  }
}

void doMerge(Body *A, int i, int j) {
  previous_v[i].first =
      (previous_v[i].first * A[i].mass + A[j].mass * previous_v[j].first) /
      (A[i].mass + A[j].mass);
  previous_v[i].second =
      (previous_v[i].second * A[i].mass + A[j].mass * previous_v[j].second) /
      (A[i].mass + A[j].mass);

  A[i].mass += A[j].mass;
  // printf("new mass %lf %lf %lf\n", A[i].x, A[i].y, A[i].mass);
}

void checkAndMerge(Body *objsX, Body *objsA, short *mergeList) {
  clearToMerge(N);
  for (int i = 0; i < N; i++) {
    if (mergeList[i * N] > 0) {

      for (int j = 1; j <= mergeList[i * N]; j++) {
        if (i == mergeList[i * N + j]) {
          continue;
        }

        myUnion(i, mergeList[i * N + j]);
      }
    }
  }
  int NewN = N;
  int indexN = 0;

  for (int i = 0; i < N; i++) {
    if (toMerge[i]) {
      int p = find(i);
      if (p != i) {
        doMerge(objsX, p, i);
        NewN--;
      }
    }
  }

  for (int i = 0; i < N; i++) {
    if (find(i) == i) {
      objsX[indexN] = objsX[i];
      objsA[indexN] = objsA[i];
      previous_v[indexN] = previous_v[i];
      parent[indexN] = indexN;
      unionRank[indexN] = unionRank[i];
      indexN++;
    }
  }

  if (N != NewN) {
    N = NewN;
    CUresult res;

    res = cuMemcpyHtoD(devX, objsX, sizeof(Body) * N);
    if (res != CUDA_SUCCESS) {
      printf("err memcpy1 res: %d \n", res);
      exit(1);
    }

    res = cuMemcpyHtoD(devA, objsA, sizeof(Body) * N);
    if (res != CUDA_SUCCESS) {
      printf("err memcpy2\n");
      exit(1);
    }

    res = cuMemcpyHtoD(devPreviousV, previous_v, sizeof(double2) * N);
    if (res != CUDA_SUCCESS) {
      printf("copy prev erro err %d\n", res);
      exit(1);
    }
  }
  cuCtxSynchronize();
}

void launch_cuda(Body *objsX, Body *objsA, short *mergeList, int N) {
  CUresult res;

  void *args[] = {&devX, &devA, &devPreviousV, &devMergeList, &N};
  res = cuLaunchKernel(leapfrog_integrator_cuda, blocks_per_grid, 1, 1,
                       threads_per_block, 1, 1, 0, 0, args, 0);
  if (res != CUDA_SUCCESS) {
    printf("cannot run kernel\n");
    exit(1);
  }

  // TODO fix bugs on kernel
  res = cuCtxSynchronize();

  if (res != CUDA_SUCCESS) {
    printf("cuda error kernel down res %d\n", res);
    exit(1);
  }

  res = cuMemcpyDtoH(objsX, devX, sizeof(Body) * N);
  if (res != CUDA_SUCCESS) {
    printf("copy devX -> objsX err res: %d\n", res);
    exit(1);
  }

  res = cuMemcpyDtoH(previous_v, devPreviousV, sizeof(double2) * N);
  if (res != CUDA_SUCCESS) {
    printf("copy devX -> objsX err res: %d\n", res);
    exit(1);
  }

  res = cuMemcpyDtoH(objsA, devA, sizeof(Body) * N);
  if (res != CUDA_SUCCESS) {
    printf("copy devA -> objsA err\n");
    exit(1);
  }

  if(flags & COLISION_FLAG){
      res = cuMemcpyDtoH(mergeList, devMergeList, sizeof(short) * N * (N + 1));
      if (res != CUDA_SUCCESS) {
        printf("copy devA -> objsA err\n");
        exit(1);
      }
  }
    
  if(flags & COLISION_FLAG)
    checkAndMerge(objsX, objsA, mergeList);

  cuCtxSynchronize();
}

// Main loop flag
bool quit = false;
// Event handler
SDL_Event e;

void print_tab(Body *A, int size, double time) {
  cout << "TIME " << time << endl;
  for (int i = 0; i < size; i++)
    cout << "OBJ: " << i << " X: " << A[i].x << " Y: " << A[i].y << endl;
}

long long millis() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return tp.tv_sec * 1000 + tp.tv_usec / 1000;
}

void create_disk(Body *objsA, Body* objsX){
    srand(time(0));
    // center
    objsX[0] = Body(SCREEN_WIDTH/2, SCREEN_HEIGHT/2, 600);
    objsA[0] = Body();
    cout << "Center is: " << SCREEN_WIDTH/2 << " " << SCREEN_HEIGHT/2 << endl;
    double angle;
    int distance;
    double x, y;
    int ax, ay;
    for(int i = 1; i < N; ++i){
        angle = rand()%360;
        angle = angle/360 * 2 * PI;
        distance = rand()%SCREEN_HEIGHT/2;
        x = distance;
        y = 0;
        objsX[i] = Body(SCREEN_WIDTH/2 + x*cos(angle) - y*sin(angle), SCREEN_HEIGHT/2 + y*cos(angle) + x*sin(angle), rand()%(MAX_MASS - MIN_MASS) + MIN_MASS);
        x = 10;
        angle += PI/2;
        previous_v[i] = double2(x*cos(angle) - y*sin(angle), y*cos(angle) + x*sin(angle));
    }
  if(flags & CUDA_FLAG){
      CUresult res;

      res = cuMemcpyHtoD(devPreviousV, &previous_v, sizeof(double2) * N);
      if (res != CUDA_SUCCESS) {
        printf("copy prev erro err %d\n", res);
        exit(1);
      }
      res = cuMemcpyHtoD(devX, objsX, sizeof(Body) * N);
      if (res != CUDA_SUCCESS) {
        printf("err memcpy1 res: %d \n", res);
        exit(1);
      }

      res = cuMemcpyHtoD(devA, objsA, sizeof(Body) * N);
      if (res != CUDA_SUCCESS) {
        printf("err memcpy2\n");
        exit(1);
      }
  }


}

void create_random_objects(Body *objsA, Body *objsX) {

  srand(time(0));
  for (int i = 0; i < N; i++) {
    objsX[i] = Body(abs(rand() % SCREEN_WIDTH), abs(rand() % SCREEN_HEIGHT),
                    abs(rand() % (MAX_MASS - MIN_MASS) + MIN_MASS));
    objsA[i] = Body();
  }
    
  if(flags & CUDA_FLAG){
      CUresult res;

      res = cuMemcpyHtoD(devX, objsX, sizeof(Body) * N);
      if (res != CUDA_SUCCESS) {
        printf("err memcpy1 res: %d \n", res);
        exit(1);
      }

      res = cuMemcpyHtoD(devA, objsA, sizeof(Body) * N);
      if (res != CUDA_SUCCESS) {
        printf("err memcpy2\n");
        exit(1);
      }
  }
}

void init_forces() {
  for (int i = 0; i < N; i++) {
    previous_v[i] = double2(0, 0);
  }

  if(flags & CUDA_FLAG){
      int res = cuMemcpyHtoD(devPreviousV, &previous_v, sizeof(double2) * N);
      if (res != CUDA_SUCCESS) {
        printf("copy prev erro err %d\n", res);
        exit(1);
      }
  }
}

void runPerformanceTest(Body *objsX, Body *objsA, short *mergeList,
                        int rounds) {

  printf("Performance test\n");
  long long hostTime;
  long long cudaTime;
  create_random_objects(objsA, objsX);
  init_forces();
  int NCache = N;

  struct timeval tp;
  gettimeofday(&tp, NULL);
  long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;
  long long freq = 1000 / 30;

  long long lastTime = millis();
  hostTime = millis();
  for (int j = 1; j <= rounds; j++) {
    clearScreen();
    // Handle events on queue

    double interval = 1.0 / 100;

    fflush(stdout);
    for (double i = 0.0; i < 30 * interval; i += interval) {
      // print_tab(objs, N, i);
      // sleep(0.3);
      leapfrog_integrator(objsX, interval);
    }

    // draw
    for (int i = 0; i < N; i++) {
      draw_circle({(int)objsX[i].x, (int)objsX[i].y},
                  pow(objsX[i].mass, 1.0 / 3.0), yellow);
    }

    updateRender();

    // FPS
    SDL_Delay((millis() - lastTime, freq));
  }
  hostTime = millis() - hostTime;

  printf("host time: %d", hostTime);
  cudaTime = millis();
  N = NCache;
  create_random_objects(objsA, objsX);
  init_forces();

  for (int j = 1; j <= rounds; j++) {
    clearScreen();
    // Handle events on queue

    double interval = 1.0 / 100;

    for (int i = 0; i < 30; ++i) {
      launch_cuda(objsX, objsA, mergeList, N);
    }

    // draw
    for (int i = 0; i < N; i++) {
      draw_circle({(int)objsX[i].x, (int)objsX[i].y},
                  pow(objsX[i].mass, 1.0 / 3.0), yellow);
    }

    updateRender();

    // FPS
    SDL_Delay((millis() - lastTime, freq));
  }

  cudaTime = millis() - cudaTime;
  printf(" cuda time: %lld\n", cudaTime);
  printf("cuda advantage: %lld", (hostTime - cudaTime));
}

void runNormal(Body *objsX, Body *objsA, short *mergeList) {
  init_forces();
  create_disk(objsA, objsX);

  printf("cuda init succesful\n");
  fflush(stdout);

  struct timeval tp;
  gettimeofday(&tp, NULL);
  long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;
  long long freq = 1000 / 30;

  long long lastTime = millis();

  while (!quit) {
    clearScreen();
    // Handle events on queue
    while (SDL_PollEvent(&e) != 0) {
      // User requests quit
      if (e.type == SDL_QUIT) {
        quit = true;
      }
    }

    if(flags & HOST_FLAG){
        double interval = 1.0 / 100;
        for (double i = 0.0; i < 30 * interval; i += interval) {
          // print_tab(objs, N, i);
          // sleep(0.3);
          leapfrog_integrator(objsX, interval, flags & COLISION_FLAG);
        }
    }else{
        for (int i = 0; i < 30; ++i) {
          launch_cuda(objsX, objsA, mergeList, N);
        }
    }

    // draw
    for (int i = 0; i < N; i++) {
      draw_circle({(int)objsX[i].x, (int)objsX[i].y},
                  pow(objsX[i].mass, 1.0 / 3.0), yellow);
    }

    updateRender();

    // FPS
    SDL_Delay((millis() - lastTime, freq));
  }
}

int main(int argc, char *args[]) {
  flags = atoi(args[1]);
  cout << "flags are: " << flags << " " << argc <<  endl << "cuda: " << (flags & CUDA_FLAG) << " host: " << (flags & HOST_FLAG) << " colision: " << (flags & COLISION_FLAG) << endl;

  if(flags & CUDA_FLAG)
    initCuda();

  N = 200;
  if(flags & CUDA_FLAG && flags & COLISION_FLAG)
    initFindUnion(N);

  Body objsX[N];
  Body objsA[N];
  if(flags & CUDA_FLAG && flags & COLISION_FLAG)
    mergeList = new short[N * (N + 1)];

  srand(time(0));

  blocks_per_grid = (N + 2 * CHUNK_SIZE - 1) / (2 * CHUNK_SIZE);
  threads_per_block = CHUNK_SIZE;
  
  if(flags & CUDA_FLAG)
    cudaMemoryRegister(objsX, objsA, mergeList, N);

  if (!init()) {
    printf("Failed to initialize SDL!\n");
    exit(0);
  }
    
  //runPerformanceTest(objsX, objsA, mergeList, 100);
   runNormal(objsX, objsA, mergeList);
  // Free resources and close SDL
  close();

  return 0;
}

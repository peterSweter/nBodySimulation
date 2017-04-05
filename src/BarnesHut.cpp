#include <cstdio>
#include "verlet.h"
#include "cuda.h"
#include "BarnesHut.h"
#include "graphic.h"

CUresult res;
CUdevice cuDevice;
CUdeviceptr devA;
CUdeviceptr devX;
CUdeviceptr devQ; // pointer to quadtree
CUdeviceptr devI; // extra array, which enables counting bodies in proper sqares; only on Device
CUcontext cuContext;
CUmodule cuModule;
CUfunction createNextQuadTreeLevel;
CUfunction cleardevIAnddevQ;
CUfunction initialisedevI;
CUfunction filldevI;
CUfunction countMassesAndCoordinates;
int sizeOfQuadTree;
int sideOfFirstSquare;
int gridDimX;
int gridDimY = 1;
int gridDimZ = 1;
int blockDimX = 1024;
int blockDimY = 1;
int blockDimZ = 1;
Node* quadtree;
const double theta = 0.65;

void setSizeOfQuadTree()
{
	sizeOfQuadTree = 1;
	while (sizeOfQuadTree < SCREEN_WIDTH || sizeOfQuadTree < SCREEN_HEIGHT)
		sizeOfQuadTree <<= 1;
	sideOfFirstSquare = sizeOfQuadTree;
	sizeOfQuadTree = (4 * sideOfFirstSquare * sideOfFirstSquare - 1 ) / 3;
	gridDimX = sizeOfQuadTree / blockDimX + 1;
}

void initCUDA()
{
	cuInit(0);
	
	res = cuDeviceGet(&cuDevice, 0);
	if (res != CUDA_SUCCESS)
	{
        printf("cuDeviceGet failure\n"); 
        exit(1);
    }

    res = cuCtxCreate(&cuContext, 0, cuDevice);
    if (res != CUDA_SUCCESS)
	{
        printf("cuCtxCreate failure\n");
        exit(1);
    }

    res = cuModuleLoad(&cuModule, "src/cuda/BarnesHut.ptx");
    if (res != CUDA_SUCCESS) 
	{
        printf("cuModuleLoad failure\n");  
        exit(1); 
    }
    
    res = cuModuleGetFunction(&createNextQuadTreeLevel, cuModule, "createNextQuadTreeLevel");
    if (res != CUDA_SUCCESS)
	{
        printf("cuModuleGetFunction \"createNextQuadTreeLevel\" failure\n");
        exit(1);
    }
    
    res = cuModuleGetFunction(&cleardevIAnddevQ, cuModule, "cleardevIAnddevQ");
    if (res != CUDA_SUCCESS)
	{
        printf("cuModuleGetFunction \"cleardevIAnddevQ\" failure\n");
        exit(1);
    }
    
    res = cuModuleGetFunction(&initialisedevI, cuModule, "initialisedevI");
    if (res != CUDA_SUCCESS)
	{
        printf("cuModuleGetFunction \"initialisedevI\" failure\n");
        exit(1);
    }
    
    res = cuModuleGetFunction(&filldevI, cuModule, "filldevI");
    if (res != CUDA_SUCCESS)
	{
        printf("cuModuleGetFunction \"filldevI\" failure\n");
        exit(1);
    }
    
    res = cuModuleGetFunction(&countMassesAndCoordinates, cuModule, "countMassesAndCoordinates");
    if (res != CUDA_SUCCESS)
	{
        printf("cuModuleGetFunction \"countMassesAndCoordinates\" failure\n");
        exit(1);
    }
}

void cudaMemoryRegister(Body* objsX, Body* objsA, int N)
{
	res = cuMemHostRegister(objsX, sizeof(Body) * N, 0);
	if (res != CUDA_SUCCESS)
	{
		printf("cuMemHostRegister objsX failure\n");
		exit(1);
	}
	
	res = cuMemHostRegister(objsA, sizeof(Body) * N, 0);
	if (res != CUDA_SUCCESS)
	{
		printf("cuMemHostRegister objsA failure\n");
		exit(1);
	}
	
	res = cuMemAlloc(&devX, sizeof(Body) * N);
	if (res != CUDA_SUCCESS)
	{
		printf("cuMemAlloc devX failure\n");
		exit(1);
	}
	
	res = cuMemAlloc(&devA, sizeof(Body) * N);
	if (res != CUDA_SUCCESS)
	{
		printf("cuMemAlloc devA failure\n");
		exit(1);
	}
	
	res = cuMemAlloc(&devQ, sizeof(Node) * sizeOfQuadTree);
	if (res != CUDA_SUCCESS)
	{
		printf("cuMemAlloc devQ failure\n");
		exit(1);
	}
	
	res = cuMemAlloc(&devI, sizeof(int) * sizeOfQuadTree);
	if (res != CUDA_SUCCESS)
	{
		printf("cuMemAlloc devI failure\n");
		exit(1);
	}
}

void copyToDevice(Body* objsX, Body* objsA, int N)
{
	res = cuMemcpyHtoD(devX, objsX, sizeof(Body) * N);
	if (res != CUDA_SUCCESS)
	{
		printf("cuMemcpyHtoD devX failure\n");
		exit(1);
	}
	
	res = cuMemcpyHtoD(devA, objsA, sizeof(Body) * N);
	if (res != CUDA_SUCCESS)
	{
		printf("cuMemcpyHtoD devA failure\n");
		exit(1);
	}
}

void prepare()
{
	void* args1[] = {&devI, &devQ, &sizeOfQuadTree};
	res = cuLaunchKernel(cleardevIAnddevQ, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, 0, args1, 0);
	if (res != CUDA_SUCCESS)
	{
		printf("cuLaunchKernel \"cleardevIAnddevQ\" failure\n");
		printf("%d\n", res);
		exit(1);
	}
}

void findOutCentresOfMass()
{
	int lastIndex = sizeOfQuadTree;
	int i = sideOfFirstSquare;
	int firstIndex = lastIndex - i * i;
	while (i)
	{
		void* args1[] = {&devX, &devQ, &firstIndex, &lastIndex};
		res = cuLaunchKernel(countMassesAndCoordinates, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, 0, args1, 0);
		if (res != CUDA_SUCCESS)
		{
			printf("cuLaunchKernel \"countMassesAndCoordinates\" failure\n");
			exit(1);
		}
		
		i /= 2;
		lastIndex = firstIndex;
		firstIndex -= i * i;
	}
}

void createQuadTree(int N)
{
	int i = 2;
	int firstFreeIndex = 1; // first index in quadtree not analysed yet; we start with 1, since we don't analyse first square
	while (i <= sideOfFirstSquare)
	{
		int sideOfCurrentSquare = sideOfFirstSquare / i;
		void* args1[] = {&devI, &devX, &devQ, &N, &sideOfCurrentSquare, &i, &firstFreeIndex};
		res = cuLaunchKernel(filldevI, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, 0, args1, 0);
		if (res != CUDA_SUCCESS)
		{
			printf("cuLaunchKernel \"filldevI\" failure\n");
			exit(1);
		}
	
		void* args2[] = {&devQ, &devI, &i, &firstFreeIndex};
		res = cuLaunchKernel(createNextQuadTreeLevel, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, 0, args2, 0);
		if (res != CUDA_SUCCESS)
		{
			printf("cuLaunchKernel \"createNextQuadTreeLevel\" failure\n");
			exit(1);
		}
		
		void* args3[] = {&devI, &N};
		res = cuLaunchKernel(initialisedevI, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, 0, args3, 0);
		if (res != CUDA_SUCCESS)
		{
			printf("cuLaunchKernel \"initialisedevI\" failure\n");
			exit(1);
		}
		
		firstFreeIndex += (i * i);
		i <<= 1;
	}
	findOutCentresOfMass();
}

void copyQuadtreeToHost()
{
	res = cuMemcpyDtoH(quadtree, devQ, sizeof(Node) * sizeOfQuadTree);
	if (res != CUDA_SUCCESS)
	{
		printf("cuMemcpyDtoH quadtree failure\n");
		exit(1);
	}
}

void getForce(Body A, Node B, double2& force)
{
	double rx;
	double ry;
	double distSqr = 0;
	double distSixth;
	double invDistCube;
	double EPS2 = 0.1;
	double G = 1;
	rx = -A.x + B.x;
	ry = -A.y + B.y;
	distSqr = rx * rx + ry * ry + EPS2;
	distSixth = distSqr * distSqr * distSqr;
	invDistCube = 1.0 / sqrt(distSixth);
	force.first += rx * invDistCube * A.mass * B.mass * G;
	force.second += ry * invDistCube * A.mass * B.mass * G;
}

double distance(Body A, Node B)
{
	return sqrt((A.x - B.x) * (A.x - B.x) + (A.y - B.y) * (A.y * B.y));
}

void traverse(int indexOfCurrentBody, int node, double2& force, Body* objsX, int s)
{
	double2 res;
	res.first = 0;
	res.second = 0;
	
	if (quadtree[node].objsXIndex == indexOfCurrentBody)
		return;
	if (quadtree[node].objsXIndex >= 0) // this node represents particular body
		getForce(objsX[indexOfCurrentBody], quadtree[node], force);
	if (quadtree[node].objsXIndex == -1) // internal node
	{
		if (s / distance(objsX[indexOfCurrentBody], quadtree[node]) < theta)
		{
			getForce(objsX[indexOfCurrentBody], quadtree[node], force);
			return;
		}
		if (quadtree[node].first >= 0)
			traverse(indexOfCurrentBody, quadtree[node].first, force, objsX, s / 2);
		if (quadtree[node].second >= 0)
			traverse(indexOfCurrentBody, quadtree[node].second, force, objsX, s / 2);
		if (quadtree[node].third >= 0)
			traverse(indexOfCurrentBody, quadtree[node].third, force, objsX, s / 2);
		if (quadtree[node].fourth >= 0)
			traverse(indexOfCurrentBody, quadtree[node].fourth, force, objsX, s / 2);
	}
}

void getForcesAndUpdate(Body* objsX, Body* objsA, int N)
{
	copyQuadtreeToHost();
	double2 force;
	double2 velocity;
	double time = 1.0 / 800;
	for (int i = 0; i < N; i++)
	{
		force.first = force.second = 0;
		traverse(i, 0, force, objsX, sideOfFirstSquare);
		velocity.first = previous_v[i].first + time * force.first;
		velocity.second = previous_v[i].second + time * force.second;
		previous_v[i] = velocity;
		objsA[i] = update_Body(objsX[i], velocity, time); // function from verlet.cpp
	}
	std::swap(objsX, objsA);
}

void BarnesHutSimulation(Body* objsX, Body* objsA, int N)
{
	setSizeOfQuadTree();
	initCUDA();
	cudaMemoryRegister(objsX, objsA, N);
	copyToDevice(objsX, objsA, N);
	quadtree = (Node*)malloc(sizeof(Node) * sizeOfQuadTree);
	while (1)
	{
		prepare();
		createQuadTree(N);
		getForcesAndUpdate(objsX, objsA, N);
	}
	free(quadtree);
}

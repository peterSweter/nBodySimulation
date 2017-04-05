#include <cstdio>
#include "../BarnesHut.h"

extern "C"
{
	__global__ void cleardevIAnddevQ(int* devI, Node* devQ, int sizeOfQuadTree)
	{
		int thid = blockIdx.x * blockDim.x + threadIdx.x;
		if (thid < sizeOfQuadTree)
		{
			devI[thid] = devQ[thid].mass = 0;
			devQ[thid].first = devQ[thid].second = devQ[thid].third = devQ[thid].fourth = devQ[thid].objsXIndex = devQ[thid].mass = devQ[thid].x = devQ[thid].y = -2;
		}
		if (!thid)
			devQ[thid].objsXIndex = -1;
	}
	
	__global__ void initialisedevI(int* devI, int sizeOfQuadTree)
	{
		int thid = blockIdx.x * blockDim.x + threadIdx.x;
		if (thid < sizeOfQuadTree)
			devI[thid] = 0;
	}

	__global__ void filldevI(int* devI, CUDABody* devX, Node* devQ, int N, int sideOfCurrentSquares, int phase, int firstFreeIndex)
	{
		int thid = blockIdx.x * blockDim.x + threadIdx.x;
		int indexOfSquare;
		// count indexOfSquare and increment appropriate cell of devI
		if (thid < N)
		{
			int x = devX[thid].x / sideOfCurrentSquares;
			int y = devX[thid].y / sideOfCurrentSquares;
			indexOfSquare = x * phase + y + firstFreeIndex;
			
			if (devI[indexOfSquare] < 2)
				atomicAdd(&devI[indexOfSquare], 1);
				
			// next 2 lines may distort quadtree structure temporarily; it'll be fixed in "createNextQuadTreeLevel" function
			if (devQ[indexOfSquare].objsXIndex == -2)
				devQ[indexOfSquare].objsXIndex = thid;
		}
	}

	__global__ void createNextQuadTreeLevel(Node* devQ, int* devI, int phase, int firstFreeIndex)
	{
		int thid = blockIdx.x * blockDim.x + threadIdx.x;
		
		if (thid >= firstFreeIndex && thid < firstFreeIndex + phase * phase) // we analyse only particular set of squares
		{
			int x = (thid - firstFreeIndex) / phase;
			int y = (thid - firstFreeIndex) % phase;
			int fatherX = x / 2; // row index of father of the square square being represented by current thread
			int fatherY = y / 2; // column index of father of the square being represented by current thread
			int father = fatherX * (phase / 2) + fatherY + (firstFreeIndex - (phase * phase / 4));
			int childIndex; // first, second, third or fourth
			// counting childIndex
			if (x % 2 == 0)
			{
				if (y % 2 == 0)
					childIndex = 1;
				else
					childIndex = 2;
			}
			else
			{
				if (y % 2 == 0)
					childIndex = 3;
				else
					childIndex = 4;
			}
			
			if (!devI[thid])
				devQ[thid].objsXIndex = -2;
			if (devI[thid] >= 2)
				devQ[thid].objsXIndex = -1;
			if (devI[thid] == 1 && devQ[father].objsXIndex >= 0) // father already represents single body
				devQ[thid].objsXIndex = -2;
			if (devI[thid] == 1 && devQ[father].objsXIndex == -1) // father has at least one child
			{
				if (childIndex == 1)
					devQ[father].first = thid;
				if (childIndex == 2)
					devQ[father].second = thid;
				if (childIndex == 3)
					devQ[father].third = thid;
				if (childIndex == 4)
					devQ[father].fourth = thid;
			}
		}
	}
	
	__global__ void countMassesAndCoordinates(CUDABody* devX, Node* devQ, int firstIndex, int lastIndex)
	{
		int thid = blockIdx.x * blockDim.x + threadIdx.x;
		
		if (thid >= firstIndex && thid < lastIndex)
		{
			if (devQ[thid].objsXIndex != -2) // this node has a child or is a leaf
			{
				if (devQ[thid].objsXIndex == -1) // this node has a child
				{
					int child1 = devQ[thid].first;
					int child2 = devQ[thid].second;
					int child3 = devQ[thid].third;
					int child4 = devQ[thid].fourth;
					int mass1 = 0;
					int mass2 = 0;
					int mass3 = 0;
					int mass4 = 0;
					int x1 = 0;
					int x2 = 0;
					int x3 = 0;
					int x4 = 0;
					int y1 = 0;
					int y2 = 0;
					int y3 = 0;
					int y4 = 0;
				
					if (child1 >= 0)
					{
						mass1 = devQ[child1].mass;
						x1 = devQ[child1].x;
						y1 = devQ[child1].y;
					}
				
					if (child2 >= 0)
					{
						mass2 = devQ[child2].mass;
						x2 = devQ[child2].x;
						y2 = devQ[child2].y;
					}
				
					if (child3 >= 0)
					{
						mass3 = devQ[child3].mass;
						x3 = devQ[child3].x;
						y3 = devQ[child3].y;
					}
				
					if (child4 >= 0)
					{
						mass4 = devQ[child4].mass;
						x4 = devQ[child4].x;
						y4 = devQ[child4].y;
					}
					
					devQ[thid].mass = mass1 + mass2 + mass3 + mass4;
					devQ[thid].x = (x1 * mass1 + x2 * mass2 + x3 * mass3 + x4 * mass4) / (x1 + x2 + x3 + x4);
					devQ[thid].y = (y1 * mass1 + y2 * mass2 + y3 * mass3 + y4 * mass4) / (y1 + y2 + y3 + y4);
				}
				else // this node represents single body
				{
					int index = devQ[thid].objsXIndex;
					devQ[thid].mass = devX[index].mass;
					devQ[thid].x = devX[index].x;
					devQ[thid].y = devX[index].y;
				}
			}
		}
	}
}

#ifndef _BARNESHUT_H_
#define _BARNESHUT_H_

#include <iostream>
#include "verlet.h"

struct Node
{
	// each node contains indexes of its sons; -2 means "no son"
	// we don't use pointers since it's a slower method
	int first = -2;
	int second = -2;
	int third = -2;
	int fourth = -2;
	
	// in entirely built quadtree, -1 will mean that this node is not a leaf, so will have children; -2 - we'll not use this node in the future; every other value - this node is a leaf, so represents a body
	int objsXIndex = -2;
	
	int mass = -2; // total mass of bodies being represented by this node
	// coordinates of centre of gravity of bodies being represented by this node
	int x = -2;
	int y = -2;
};

struct CUDABody
{
	double x;
	double y;
	double mass;
};

void BarnesHutSimulation(Body*, Body*, int);
void getForcesAndUpdate(Body*, Body*, int);
void traverse(int, int, std::pair<double, double>&, Body*, int);
double distance(Body, Node);
void getForce(Body, Node, std::pair<double, double>&);
void copyQuadtreeToHost();
void createQuadTree(int);
void findOutCentresOfMass();
void prepare();
void copyToDevice(Body*, Body*, int);
void cudaMemoryRegister(Body*, Body*, int);
void initCUDA();
void setSizeOfQuadTree();

#endif

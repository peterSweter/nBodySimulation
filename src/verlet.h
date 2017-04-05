#ifndef _VERLET_H_
#define _VERLET_H_
#include <cmath>
#include <iostream>


#define CUDA_FLAG 1
#define HOST_FLAG 2
#define COLISION_FLAG 4


typedef std::pair<double, double> double2;
/// maximum number of objects in arrays
static const int MAX_N = 10000;
extern int N;
extern double2 previous_v[MAX_N];

/// default mass
static double mass = 100.0;

struct Body {
  /// x,y - coordinates
  //  mass - mass of the Body
  //  should add radious
  double x;
  double y;
  double mass;
 // bool ghost = false;

  Body() : x(0), y(0), mass(0){};
  Body(int a, int b) : x(a), y(b), mass(::mass){};
  Body(double a, double b, double c) : x(a), y(b), mass(c){};
};


// utility functions
double dist(Body&, Body&);

// leapfrog functions
double2 get_force(Body*, int);
Body update_pos(Body&, double2, double);
void leapfrog_integrator(Body*, double, bool x = false);

// collisions and merging functions
void check_and_merge(Body*);
void do_merge(Body*, int, int);

#endif

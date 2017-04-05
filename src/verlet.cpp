#include "verlet.h"
#include <cmath>
#include <iostream>

using std::sqrt;
using std::cout;
using std::endl;
/// mass is 1 for now

static const double G = 1;
static const double EPS2 = 1;
int N;

extern double2 previous_v[MAX_N];

double2 get_force(Body *A, int i) {
  double2 res;
  res.first = 0;
  res.second = 0;
  double rx;
  double ry;
  double distSqr = 0;
  double distSixth;
  double invDistCube;
  for (int j = 0; j < N; j++) {
    rx = -A[i].x + A[j].x;
    ry = -A[i].y + A[j].y;
    distSqr = rx * rx + ry * ry + EPS2;
    distSixth = distSqr * distSqr * distSqr;
    invDistCube = 1.0 / sqrt(distSixth);
    res.first += rx * invDistCube * A[j].mass;
    res.second += ry * invDistCube * A[j].mass;
  }
  res.first *= G;
  res.second *= G;
  return res;
}

double dist(Body &A, Body &B) {
  double rx = B.x - A.x;
  double ry = B.y - A.y;

  return sqrt(rx * rx + ry * ry);
}

Body update_Body(Body &currentPos, double2 v, double t) {
  Body newPos;
  newPos.mass = currentPos.mass;
  newPos.x = currentPos.x + t * v.first;
  newPos.y = currentPos.y + t * v.second;
  return newPos;
}

void do_merge(Body *A, int i, int j) {
  previous_v[i].first =
      (previous_v[i].first * A[i].mass + A[j].mass * previous_v[j].first) /
      (A[i].mass + A[j].mass);
  previous_v[i].second =
      (previous_v[i].second * A[i].mass + A[j].mass * previous_v[j].second) /
      (A[i].mass + A[j].mass);

  A[i].mass += A[j].mass;
  A[j] = A[N - 1];
  previous_v[j] = previous_v[N - 1];
  --N;
}

void check_and_merge(Body *A) {
  for (int i = 0; i < N; i++) {
    Body &current = A[i];
    for (int j = 0; j < N; j++) {
      if (i == j)
        continue;

      double distance = dist(current, A[j]);
      if (4 * distance < log2(current.mass) || distance < log2(A[j].mass)) {
        //  cout << "MERGING " << i << ": " << A[i].x << " " << A[i].y << " " <<
        //  j
        //   << ": " << A[j].x << " " << A[j].y << endl;
        if (A[i].mass < A[j].mass)
          std::swap(A[i], A[j]);
        do_merge(A, i, j);
        //  cout << "AFTER MERGE " << A[i].x << " " << A[i].y << endl;
        j--;
      }
    }
  }
}

void leapfrog_integrator(Body *A, double t, bool check_merge) {
  double2 v;
  double2 f;
  /// checking for collisions
  if(check_merge)
    check_and_merge(A);

  for (int i = 0; i < N; i++) {
    f = get_force(A, i);
    v.first = previous_v[i].first + t * f.first;
    v.second = previous_v[i].second + t * f.second;
    previous_v[i] = v;
    A[i] = update_Body(A[i], v, t);
  }
}

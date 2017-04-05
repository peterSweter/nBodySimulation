#include "verlet.h"

extern int parent[MAX_N];
extern int unionRank[MAX_N];
extern bool toMerge[MAX_N];

void myUnion(int a, int b);
int find(int v);
void initFindUnion(int N);
void clearToMerge(int N);

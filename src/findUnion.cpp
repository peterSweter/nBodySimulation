#include "findUnion.h"

extern int parent[MAX_N];
extern int unionRank[MAX_N];
extern bool toMerge[MAX_N];

void initFindUnion(int N) {
  for (int i = 0; i < N; i++) {
    parent[i] = i;
    unionRank[i] = 0;
  }
}

int find(int v) {
  if (parent[v] != v) {
    parent[v] = find(parent[v]);
  }
  return parent[v];
}

void myUnion(int a, int b) {
  toMerge[a] = true;
  toMerge[b] = true;

  a = find(a);
  b = find(b);

  if (a == b) {
    return;
  }
  //  printf("union %d %d\n", a, b);

  if (unionRank[a] > unionRank[b]) {
    parent[b] = a;
  } else if (unionRank[a] < unionRank[b]) {
    parent[a] = b;
  } else {
    parent[a] = b;
    unionRank[b]++;
  }
}

void clearToMerge(int N) {
  for (int i = 0; i < N; i++) {
    toMerge[i] = false;
  }
}

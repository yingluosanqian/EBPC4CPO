#include <bits/stdc++.h>

using namespace std;
#ifdef LOCAL
#include "../debugger/debug.h"
#else
#define debug(...) 420
#endif

int main() {
  cin.tie(0)->sync_with_stdio(0);
  cin.exceptions(cin.failbit);
  int n;
  cin >> n;
  vector<int> p(n + 1);
  for (int i = 1; i <= n; i++) {
    cin >> p[i];
  }
  vector<int> q(n + 1);
  for (int i = 1; i <= n; i++) {
    q[p[i]] = i;
  }
  for (int i = 1; i <= n; i++) {
    cout << q[i] << " \n"[i == n];
  }
  return 0;
}
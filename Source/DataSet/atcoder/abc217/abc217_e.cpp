#include <bits/stdc++.h>
using namespace std;

int main (void) {
  int n;
  cin >> n;
  queue<int> p;
  priority_queue<int, vector<int>, greater<int>> q;
  
  for (int i=0; i<n; i++) {
    int c;
    cin >> c;
    if (c == 1) {
      cin >> c;
      p.push(c);
    }
    else if (c == 2) {
      if (!q.empty()) {
        cout << q.top() << endl;
        q.pop();
      }
      else {
        cout << p.front() << endl;
        p.pop();
      }
    }
    else if (c == 3) {
      while (!p.empty()) {
        c = p.front();
        q.push(c);
        p.pop();
      }
    }
  }
}
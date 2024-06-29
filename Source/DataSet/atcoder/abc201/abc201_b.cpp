#include <bits/stdc++.h>
using namespace std;
 
int main() {
  int n;
  cin >> n;
  vector<pair<int,string>> data(n);
  for (int i = 0; i < n; i++) cin >> data[i].second >> data[i].first;
  sort(data.begin(), data.end(), greater{});
  cout << data[1].second << endl;
}
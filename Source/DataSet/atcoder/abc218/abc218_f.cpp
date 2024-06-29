#include <bits/stdc++.h>

typedef long long ll;
using namespace std;

int main() {
  bool visited[1000];
  memset(&visited, 0, sizeof visited);

  int n, m;
  cin >> n >> m;
  // vector<array<int, 2>> edges;

  unordered_map<int, unordered_set<int>> edges;

  vector<array<int, 2>> original_edges;

  for (int i = 0; i < m; i++) {
    int s, t;
    cin >> s >> t;
    original_edges.push_back({s, t});
    edges[s].emplace(t);
  }

  // find the shortest path with bfs first

  unordered_map<int, int> reverse_map;
  // vector<int> shortest_path;

  deque<int> q;

  q.push_back(1);
  visited[1] = true;
  bool found = false;
  while (!q.empty()) {
    int item = q.front();
    q.pop_front();
    if (item == n) {
      found = true;
      break;
    }
    for (int child : edges[item]) {
      if (visited[child]) continue;
      visited[child] = true;
      reverse_map[child] = item;
      q.push_back(child);
    }
  }

  if (!found) {
    while(m--) cout << -1 << endl;
    return 0;
  }

  int shortest_path_len = 0;

  int vtx = n;
  while (vtx != 1) {
    shortest_path_len++;
    // shortest_path.push_back(vtx);
    vtx = reverse_map[vtx];
  }
  shortest_path_len++;
  // shortest_path.push_back(1);

  // for (int i : shortest_path) cout << i << ", ";
  // cout << endl;

  for (auto& edge : original_edges) {
    auto [l, r] = edge;
    if (reverse_map[r] != l) {  // not one of the shortest path edges
      cout << shortest_path_len - 1 << endl;
    } else {
      // calculate shortest path
      edges[l].erase(r);

      q.clear();
      memset(&visited, 0, sizeof visited);
      q.push_back(1);

      int moves = 0;
      bool found = false;
      while (!q.empty()) {
        int sz = q.size();
        while (sz--) {
          int item = q.front();
          q.pop_front();
          if (item == n) {
            found = true;
            break;
          }
          for (int child : edges[item]) {
            if (visited[child]) continue;
            visited[child] = true;
            q.push_back(child);
          }
        }
        if (found) break;
        moves++;
      }

      if (!found) {
        cout << -1 << endl;
      } else {
        cout << moves << endl;
      }

      edges[l].emplace(r);
    }
  }
}
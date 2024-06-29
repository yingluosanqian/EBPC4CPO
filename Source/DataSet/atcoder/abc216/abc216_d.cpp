#include <bits/stdc++.h>
#define LOOP(n) for (int _i = 0; _i < (n); _i++)
#define REP(i, n) for (int i = 0; i < (n); ++i)
#define RREP(i, n) for (int i = (n); i >= 0; --i)
#define FOR(i, r, n) for (int i = (r); i < (n); ++i)
#define ALL(obj) begin(obj), end(obj)
#define yes "Yes"
#define no "No"
using namespace std;
using ll = long long;
const vector<char> alpha = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'};
int dx[4] = {0, 1, 0, -1};
int dy[4] = {1, 0, -1, 0};
const ll INF = 1000000007;
bool ans=true;
vector<vector<ll>> A(202020);
vector<set<ll>> G(202020);
vector<bool> vis(202020,false),finished(202020,false);
ll N,M;
void dfs(int v){
  for (auto u:G[v]){
    if (vis[u]){
      if (!finished[u]) {
        ans=false;
        return;
      }
      continue;
    }
    vis[u]=true;
    dfs(u);
  }
  finished[v]=true;
}
int main(){
  cin >> N >> M;
  REP(i,M) {
    int k;
    cin >> k;
    REP(j,k) {
      int a;
      cin >> a;
      a--;
      A[i].push_back(a);
      if (j!=0) {
        G[A[i][j-1]].insert(a);
      }
    }
  }
  REP(i,N){
    if (vis[i]) continue;
    vis[i]=true;
    dfs(i);
  }
  if (ans) cout << yes<<endl;
  else cout << no <<endl;
}
// LUOGU_RID: 149299377
#include<bits/stdc++.h>
using namespace std;
const int MOD = 1e9 + 7;
int N, M, vis[200005], dis[200005];
long long cnt[200005];
vector<int>G[200005];
queue<int>Q;
int main(){
	cin >> N >> M;
	for(int i = 1; i <= M; i++){
		int A, B; cin >> A >> B;
		G[A].push_back(B);
		G[B].push_back(A);
	}
	memset(dis, 0x3f, sizeof dis);
	cnt[1] = vis[1] = 1;
	dis[1] = 0;
	Q.push(1);
	while(!Q.empty()){
		int u = Q.front(); Q.pop();
		int d = dis[u];
		for(int i = 0; i < G[u].size(); i++){
			int v = G[u][i];
			if(d + 1 == dis[v])cnt[v] = (cnt[v] + cnt[u]) % MOD;
			if(d + 1 < dis[v]){
				dis[v] = d + 1, cnt[v] = cnt[u];
			}if(!vis[v])Q.push(v), vis[v] = 1;
		}
	}
	cout << cnt[N];
}
// LUOGU_RID: 149102876
#include<bits/stdc++.h>
#define int long long
using namespace std;
const int M=5010;
const int mod=1e9+7;

int n,sz[M],ans[M],cnt;
int f[M][M][3],g[M][3];
vector<int> eg[M];

void dfs(int u,int fa){
	f[u][0][0]=f[u][1][1]=1;
	sz[u]=1; 

	for(auto v:eg[u]){
		if(v==fa) continue;
		dfs(v,u);

		for(int i=0;i<=sz[u];i++){
			g[i][0]=f[u][i][0];
			g[i][1]=f[u][i][1];
			g[i][2]=f[u][i][2];
			f[u][i][0]=f[u][i][1]=f[u][i][2]=0;
			//Í¬²½×ªÒÆ
		}

		for(int i=sz[u];i>=0;i--){
			for(int j=sz[v];j>=0;j--){
				f[u][i+j][0]=(f[u][i+j][0]+(g[i][0]*f[v][j][0])%mod)%mod;
				f[u][i+j+1][2]=(f[u][i+j+1][2]+(g[i][0]*f[v][j][1])%mod)%mod;
				f[u][i+j][0]=(f[u][i+j][0]+(g[i][0]*f[v][j][2])%mod)%mod;
				f[u][i+j+1][1]=(f[u][i+j+1][1]+(g[i][1]*f[v][j][0])%mod)%mod;
				f[u][i+j][1]=(f[u][i+j][1]+(g[i][1]*f[v][j][1])%mod)%mod;
				f[u][i+j][1]=(f[u][i+j][1]+(g[i][1]*f[v][j][2])%mod)%mod;
				f[u][i+j][2]=(f[u][i+j][2]+(g[i][2]*f[v][j][0])%mod)%mod;
				f[u][i+j][2]=(f[u][i+j][2]+(g[i][2]*f[v][j][1])%mod)%mod;
				f[u][i+j][2]=(f[u][i+j][2]+(g[i][2]*f[v][j][2])%mod)%mod;
			}
		}
		sz[u]+=sz[v];
	}
}
signed main(){
	ios::sync_with_stdio(false);
	cin.tie(0);
	cout.tie(0);

	cin>>n;
	int u,v;
	for(int i=1;i<n;i++){
		cin>>u>>v;
		eg[u].push_back(v);
		eg[v].push_back(u);
	}

	dfs(1,0);
	for(int i=0;i<=n;i++){
		cout<<(f[1][i][0]+f[1][i][1]+f[1][i][2])%mod<<endl;
	}

	return 0;
}

// LUOGU_RID: 148915107
//Author: gsczl71
//Copyright (c) 2024 gsczl71 All rights reserved.

#include<bits/stdc++.h>
#define ll long long
#define i128 __int128
#define ull unsigned long long
#define pii pair<int,int>
#define pll pair<long long, long long>
#define fs first
#define sc second
#define endl '\n'
#define debug puts("AK IOI")
#define re register
#define pb push_back
#define mem(a,x) memset((a),(x),sizeof(a))
#define vi vector<int>
using namespace std;
#define int long long
// const int mod = 1e9+7;
const int mod = 998244353;
const int inf = 0x3f3f3f3f,N = 5005,M = 2e5+5;
const ll linf = 0x3f3f3f3f3f3f3f3f;
int n,m;
int dp[N][N];
void solve(){
	cin>>n>>m;
	dp[0][0]=1;
	for(int i = 1;i <= n;i++){
		for(int j = 1;j <= n;j++){
			dp[i][j] = dp[i-1][j-1];
			if(j >= (i-1)/m) dp[i][j] += (dp[i-1][j] * (j - (i-1)/m));
			dp[i][j] %= mod;
		}
	}for(int i = 1;i <= n;i++){
		cout<<dp[n][i]<<endl;
	}
}
signed main(){
	ios::sync_with_stdio(false), cin.tie(nullptr), cout.tie(nullptr);
	int T=1;
//	cin >> T;
	while(T--) solve();
	return 0;
}


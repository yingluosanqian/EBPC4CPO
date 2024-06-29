#include <bits/stdc++.h>
using namespace std;
using ll = long long;
#define rep(i, n) for (int i = 0; i < (int)(n); i++)
#define all(a)  (a).begin(),(a).end()

using P=pair<ll,ll>;
using T=tuple<ll,ll,ll>;

// #include <atcoder/all> 
// using namespace atcoder;
// using mint=modint998244353;
// using mint=modint1000000007;
// int op(int a,int b){return (a+b);}
// int e(){return 0;}

int main(){
    int n,m;
    cin>>n>>m;
    ll inf=pow(10,18);
    vector<vector<ll>> dist(n,vector<ll>(n,inf));
    rep(i,n)dist[i][i]=0;
    rep(i,m){
        int a,b,c;
        cin>>a>>b>>c;
        a--;b--;
        dist[a][b]=c;
    }
    ll ans=0;
    rep(k,n){
        rep(i,n){
            rep(j,n){
                dist[i][j]=min(dist[i][j],dist[i][k]+dist[k][j]);
            }
        }
        rep(i,n){
            rep(j,n){
                if(dist[i][j]!=inf)ans+=dist[i][j];
            }
        }
    }
    cout<<ans<<endl;
}
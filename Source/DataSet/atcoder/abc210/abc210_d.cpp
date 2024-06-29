#include <bits/stdc++.h>
#include <cmath>
#define rep(i,a,b) for(int i=a;i<b;i++)
using namespace std;
using v=vector<int>;
using vv=vector<vector<int>>;
using ll=long long;
struct st {
    int x;
    int y;
    st(int x=0,int y=0):x(x),y(y){}
};

bool cmp(const struct st& p,const struct st& q){
    return p.x<q.x;
}

int main(){//int 構造体型のvectorの取り方。
    int i,m,j=0,qq,g,jj=0,aa,bb,cc,dd,q;
    ll heavy=1e18+7,ans=heavy,c;
    int x=0,y=0,z=0,sum=0,h,w,n,k;
    cin>>h>>w>>c;
    vector<vector<ll>>a(h,vector<ll>(w));
    
    rep(i,0,h)rep(j,0,w)cin>>a[i][j];
    rep(k,0,2){
        vector<vector<ll>>dp(h,vector<ll>(w,heavy));
    rep(i,0,h)rep(j,0,w){
        if(i)dp[i][j]=(dp[i][j],dp[i-1][j]);/*reverseあるから
        dp[i][j]との比較は必要*/
        if(j)dp[i][j]=min(dp[i][j],dp[i][j-1]);
        ans=min(ans,dp[i][j]+a[i][j]+c*(i+j));
        dp[i][j]=min(dp[i][j],a[i][j]-c*(i+j));
    }
    rep(i,0,h){
    reverse(a[i].begin(),a[i].end());
    }
    }
        cout<<ans<<endl;
    
}
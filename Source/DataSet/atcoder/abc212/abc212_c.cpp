#include <bits/stdc++.h>
#include <cmath>
#define rep(i,a,b) for(int i=a;i<b;i++)
using namespace std;
using ll=long long;
using v=vector<ll>;
using vv=vector<vector<ll>>;

struct st {
    ll x;
    ll y;
    st(ll x=0,ll y=0):x(x),y(y){}
};

bool cmp(const struct st& p,const struct st& q){
    return p.x<q.x;
}

int main(){
    int i,m,j=0,g,jj=0,aa,bb,cc,dd,q;
    ll heavy=1e9+7,ans=heavy,c;
    int x=0,y=0,z=0,sum=0,h,w,n,k;
    queue<ll> qu;
    queue<ll> que;
    v a,b;
     cin>>n>>m;
     rep(i,0,n){
        cin>>x;
        a.push_back(x);
     }
     rep(i,0,m){
        cin>>x;
        b.push_back(x);
     }
     sort(a.begin(),a.end()); 
     sort(b.begin(),b.end());
     int up=0,dow=0;
     while(dow<=m && up<=n){
        ans=min(ans,abs(a[up]-b[dow]));
        if(a[up]>b[dow])dow++;
        else up++;
     }
     cout<<ans<<endl;
}
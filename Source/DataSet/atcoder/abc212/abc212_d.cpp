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
    ll x=0,y=0,z=0,sum=0,h,w,n,k;
    queue<ll> qu;
    queue<ll> que;
    priority_queue<ll, vector<ll>, greater<ll>> pq;
    v a,b;
     cin>>n;
     rep(i,0,n){
        cin>>q;
        if(q==1){
            cin>>x;
            pq.push(x-sum);
        }else if(q==2){
            cin>>x;
            sum+=x;
        }else{
            cout<<pq.top()+sum<<endl;
            pq.pop();
        }
     }
}
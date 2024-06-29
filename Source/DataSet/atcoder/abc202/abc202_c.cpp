#include <bits/stdc++.h>
using namespace std;
typedef long long ll;


int main(){
    int n,m,c1,c2,k1,k2;
    ll ans=0;
    cin>>n;
    vector<int> a(n,0),b(n,0),c(n,0);
    map<int,int> mpa,mpcb,mpcc;
    for(int i=0;i<n;i++){ cin>>a[i]; mpa[a[i]]++; }
    for(int i=0;i<n;i++){ cin>>b[i]; }
    for(int i=0;i<n;i++){ cin>>c[i]; mpcc[c[i]]++; }
    for(int i=0;i<n;i++){
        mpcb[i+1]=mpa[b[i]];
    }
    for(auto curr2 : mpcc){
        c2=curr2.first;
        k2=curr2.second;
        ans+=mpcb[c2]*k2;
    }

    cout<<ans<<endl;
    return 0;
}
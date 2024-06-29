/*input
3 2
1 1 1
1 2 1
2 2 2

*/
#include<bits/stdc++.h>
using namespace std;
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
typedef tree<long long,null_type,less_equal<long long>,rb_tree_tag,tree_order_statistics_node_update> indexed_set;
//order_of_key #of elements less than x
// find_by_order kth element
using ll=long long;
using ld=long double;
using pii=pair<ll,ll>;
#define f first
#define s second
#define pb push_back
#define REP(i,n) for(int i=0;i<n;i++)
#define REP1(i,n) for(int i=1;i<=n;i++)
#define FILL(n,x) memset(n,x,sizeof(n))
#define ALL(_a) _a.begin(),_a.end()
#define sz(x) (int)x.size()
#define SORT_UNIQUE(c) (sort(c.begin(),c.end()),c.resize(distance(c.begin(),unique(c.begin(),c.end()))))
#define GET(v,x) lower_bound(ALL(v),x)-v.begin()
const ll maxn=1e3+5;
const ll maxlg=__lg(maxn)+2;
const ll INF64=4e18;
const int INF=0x3f3f3f3f;
const ll MOD=1e9+7;
const ld PI=acos(-1.0l);
const ld eps=1e-6;
#define lowb(x) x&(-x)
#define MNTO(x,y) x=min(x,(__typeof__(x))y)
#define MXTO(x,y) x=max(x,(__typeof__(x))y)
template<typename T1,typename T2>
ostream& operator<<(ostream& out,pair<T1,T2> P){
    out<<P.f<<' '<<P.s;
    return out;
}
template<typename T>
ostream& operator<<(ostream& out,vector<T> V){
    REP(i,sz(V)) out<<V[i]<<((i!=sz(V)-1)?" ":"\n");
    return out;
}
ll mult(ll a,ll b){
    return a*b%MOD;
}
ll mypow(ll a,ll b){
    a%=MOD;
    if(a==0) return 0;
    if(b<=0) return 1;
    ll res=1LL;
    while(b){
        if(b&1) res=(res*a)%MOD;
        a=(a*a)%MOD;
        b>>=1;
    }
    return res;
}
int arr[maxn][maxn];
bool ok[maxn][maxn];
int n;
const int sq=maxn*maxn;
bool in[sq];
int mnx[sq],mxx[sq],mny[sq],mxy[sq];
int pref[maxn][maxn];
int32_t main(){
    ios::sync_with_stdio(false),cin.tie(0);
    int k;
    cin>>n>>k;
    REP(i,n*n) mnx[i]=mny[i]=INF,mxx[i]=mxy[i]=0;
    REP(i,n) REP(j,n){
        cin>>arr[i][j],--arr[i][j];
        int z=arr[i][j];
        MNTO(mnx[z],i);
        MXTO(mxx[z],i);
        MNTO(mny[z],j);
        MXTO(mxy[z],j);
    }
    int ans=0;
    REP(i,n) REP(j,n){
        if(ok[i][j]) continue;
        if(!in[arr[i][j]]) ++ans;
        in[arr[i][j]]=1;
    }
    if(ans<=k){
        cout<<k-ans;
        return 0;
    }
    REP1(x,n){
        REP(i,n+1) REP(j,n+1) pref[i][j]=0;
        REP(i,n*n){
            if(mnx[i]==INF) continue;
            int a=max(0,mxx[i]-x+1),b=max(0,mxy[i]-x+1),c=mnx[i],d=mny[i];
            if(a>c or b>d) continue;
            pref[c+1][d+1]++,pref[a][b]++,pref[a][d+1]--,pref[c+1][b]--;
        }
        REP(i,n-x+1){
            REP(j,n-x+1){
                if(j) pref[i][j]+=pref[i][j-1];
                if(i) pref[i][j]+=pref[i-1][j];
                if(i and j) pref[i][j]-=pref[i-1][j-1];
                //cout<<x<<' '<<i<<' '<<j<<' '<<pref[i][j]<<'\n';
                if(ans-pref[i][j]==k or ans-pref[i][j]==k-1){
                    cout<<1;
                    return 0;
                }
            }
        }
    }
    cout<<2;
}
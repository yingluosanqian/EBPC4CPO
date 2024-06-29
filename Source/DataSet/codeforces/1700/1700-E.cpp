#include<bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp> // Common file
#include <ext/pb_ds/tree_policy.hpp>
#define ll long long
#define int long long
#define rep(i,a,b) for(int i=a;i<b;i++)
#define rrep(i,a,b) for(int i=a;i>=b;i--)
#define repin rep(i,0,n)
#define di(a) int a;cin>>a;
#define precise(i) cout<<fixed<<setprecision(i)
#define vi vector<int>
#define si set<int>
#define mii map<int,int>
#define take(a,n) for(int j=0;j<n;j++) cin>>a[j];
#define give(a,n) for(int j=0;j<n;j++) cout<<a[j]<<' ';
#define vpii vector<pair<int,int>>
#define sis string s;
#define sin string s;cin>>s;
#define db double
#define be(x) x.begin(),x.end()
#define pii pair<int,int>
#define pb push_back
#define pob pop_back
#define ff first
#define ss second
#define lb lower_bound
#define ub upper_bound
#define bpc(x) __builtin_popcountll(x) 
#define btz(x) __builtin_ctz(x)
using namespace std;

using namespace __gnu_pbds;

typedef tree<int, null_type, less<int>, rb_tree_tag,tree_order_statistics_node_update> ordered_set;
typedef tree<pair<int, int>, null_type,less<pair<int, int> >, rb_tree_tag,tree_order_statistics_node_update> ordered_multiset;

const long long INF=1e18;
const long long M=1e9+7;
const long long MM=998244353;
  
int power( int N, int M){
    int power = N, sum = 1;
    if(N == 0) sum = 0;
    while(M > 0){if((M & 1) == 1){sum *= power;}
    power = power * power;M = M >> 1;}
    return sum;
}

int dx[4] = {1,0,-1,0};
int dy[4] = {0,1,0,-1};
vi pos(400001);

int sum = 0;

int calc(int i,int j, vector<vector<int>> &a){
    int n = a.size();
    int m = a[0].size();
    int mn = a[i][j];
    rep(k,0,4){
        if(i+dx[k] < n && i+dx[k] >= 0 && j+dy[k] < m && j+dy[k] >= 0){
            mn = min(mn,a[i+dx[k]][j+dy[k]]);
        }
    }
    return mn;
}

int swp(int p,int q,int r,int s,vector<vector<bool>> &mark,vector<vector<int>> &a){
    int n = a.size();
    int m = a[0].size();
    swap(a[p][q],a[r][s]);
    int res = sum;
    res -= mark[p][q];
    int val = calc(p,q,a);
    if(val == 1)res++;
    else res += (val < a[p][q]);
    res -= mark[r][s];
    val = calc(r,s,a);
    if(val == 1)res++;
    else res += (val < a[r][s]);
    set<pii> s1;
    s1.insert({p,q});
    s1.insert({r,s});
    rep(k,0,4){
        if(p+dx[k] < n && p+dx[k] >= 0 && q+dy[k] < m && q+dy[k] >= 0){
            if(s1.count({p+dx[k],q+dy[k]}))continue;
            s1.insert({p+dx[k],q+dy[k]});
            res -= mark[p+dx[k]][q+dy[k]];
            int val = calc(p+dx[k],q+dy[k],a);
            if(val == 1)res++;
            else res += (val<a[p+dx[k]][q+dy[k]]);
        }
    }
    rep(k,0,4){
        if(r+dx[k] < n && r+dx[k] >= 0 && s+dy[k] < m && s+dy[k] >= 0){
            if(s1.count({r+dx[k],s+dy[k]}))continue;
            s1.insert({r+dx[k],s+dy[k]});
            res -= mark[r+dx[k]][s+dy[k]];
            int val = calc(r+dx[k],s+dy[k],a);
            if(val == 1)res++;
            else res += (val<a[r+dx[k]][s+dy[k]]);
        }
    }
    swap(a[p][q],a[r][s]);
    return res;
}
 
void solve()
{
    int n,m;
    cin>>n>>m;
    vector<vi> a(n,vi(m));
    repin{
        rep(j,0,m){
            cin>>a[i][j];
            pos[a[i][j]] = m*i + j;
        }
    }
    vector<vector<bool>> mark(n,vector<bool> (m,0));
    repin{
        rep(j,0,m){
            if(a[i][j] == 1){
                mark[i][j] = 1;continue;
            }
            int mn = calc(i,j,a);
            if(mn < a[i][j])mark[i][j]=1;
        }
    }
    int sm = 0;
    // repin{
    //     rep(j,0,m){
    //         cout<<mark[i][j]<<" ";
    //     }cout<<"\n";
    // }
    // give(pos,n*m+1);cout<<"\n";
    rep(i,1,m*n+1){
        int row = pos[i]/m;
        int col = pos[i]%m;
        if(mark[row][col])sm++;
        else break;
    }
    if(sm == m*n){
        cout<<"0\n";return;
    }
    repin{
        rep(j,0,m){
            sum += mark[i][j];
        }
    }
    // if(n<=1){cout<<"lelo\n";return;}
    // int res = swp(1,2,0,2,mark,a);
    // cout<<res<<"\n";
    int ans = 0;
    int r = pos[sm+1]/m;
    int c = pos[sm+1]%m;
    rep(i,0,n){
        rep(j,0,m){
            if(i==r && j==c)continue;
            int res = swp(r,c,i,j,mark,a);
            if(res == m*n)ans++;
        }
    }
    rep(i,0,n){
        rep(j,0,m){
            if(i==r && j==c)continue;
            rep(k,0,4){
                if(r+dx[k] < n && r+dx[k] >= 0 && c+dy[k] < m && c+dy[k] >= 0){
                    if(i==r+dx[k] && j==c+dy[k])continue;
                    int res = swp(r+dx[k],c+dy[k],i,j,mark,a);
                    if(res == m*n)ans++;
                }
            }
        }
    }
    if(ans){
        cout<<1<<" "<<ans<<"\n";return;
    }
    else{
        cout<<2<<"\n";
    }
}

signed main(){
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    #ifdef NCR
        init();
    #endif
    #ifdef SIEVE
        sieve();
    #endif
        solve();
    return 0;
}
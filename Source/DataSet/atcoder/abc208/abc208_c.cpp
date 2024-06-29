#include <bits/stdc++.h>
using namespace std;

#define nl '\n'
#define ff first
#define ss second
#define pb push_back
#define popb pop_back
#define rev reverse
#define maxele max_element
#define minele min_element
#define mkt make_tuple
#define lowb lower_bound
#define uppb upper_bound
#define loop(i,m,n) for(int i = m; i < n; i++)
#define rloop(i,m,n) for(int i = m; i >= n;i--)
#define era erase
#define ll long long
#define lld long double
#define ull unsigned long long
#define all(x) (x).begin(), (x).end()
#define rall(x) (x).end(),(x).begin()

//********************** Constants *********************//
const lld pi=3.141592653589793238;
const ll INF=1e18;
const ll mod=1e9+7;

//********************** Typedef ********************//
typedef pair<int, int> pii;
typedef pair<ll,ll> pll;
typedef vector<ll> vll;
typedef vector<int> vi;
typedef vector<string> vst;
typedef vector<char> vc;
typedef vector<pair<string,char>> vpsc;
typedef vector<pair<char,string>> vpcs;
typedef vector<pii> vpii;
typedef vector<pair<char,string>> vpcs;
typedef vector<pll> vpll;
typedef vector<string> vs;
typedef vector<pair<string,char>> vpsc;
typedef stack<int> sti;
typedef stack<ll> stll;
typedef queue<int> qui;
typedef queue<ll> qull;
typedef deque<int> dqi;
typedef deque<ll> dqll;
typedef priority_queue<int> pqi;
typedef priority_queue<ll> pqll;
typedef priority_queue<int,vector<int>,greater<int>> pqmi;
typedef priority_queue<ll,vector<ll>,greater<ll>> pqmll;
typedef unordered_set<ll> usll;
typedef set<ll> sll;
typedef multiset<int> mlsi;
typedef multiset<ll> mlsll;
typedef unordered_map<ll,ll> umll;
typedef map<int,int> mpi;
typedef map<ll,ll> mpll;
typedef multimap<ll,ll> mmpll;

void solve()
{
    ll n,k;cin>>n>>k;

    vpll vec;

    ll x;
    loop(i,0,n)
    {
    	cin>>x;
    	vec.pb({x,i});
    }

    sort(all(vec));

    ll qu=k/n;
    ll rem=k-qu*n;
    vll ans(n,qu);

    loop(i,0,rem)
    {
    	ll index=vec[i].ss;
    	++ans[index];
    }

    loop(i,0,n)
    cout<<ans[i]<<nl;

}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    int t=1;
    // cin>>t;

    while(t--)
    {
        solve();
    }

    return 0;
}
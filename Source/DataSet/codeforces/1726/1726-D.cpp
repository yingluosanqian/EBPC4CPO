#include <bits/stdc++.h>
// #include <ext/pb_ds/assoc_container.hpp>
// #include <ext/pb_ds/tree_policy.hpp>
using namespace std;

#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
// using namespace __gnu_pbds;
#define MODS 100005
#define MOD 1000000007
#define all(x) (x).begin(), (x).end()
#define ll long long
#define ld long double
#define bigint int64_t
#define vll vector<ll>
#define vpll vector<pair<ll,ll>>
#define mp make_pair
#define mt make_tuple
#define pb push_back
#define pf push_front
#define pbb pop_back
#define pff pop_front
#define test ll t;read(t);while(t--)
#define pll pair<ll,ll>
#define vvll vector<vector<ll>>
#define fst first
#define scd second
#define ins insert
#define line "\n"
#define yes "YES"
#define no "NO"
#define arrlen(x) (ll)x.size()
#define tup tuple<ll,ll,ll>

#define debug1(x)  cerr << #x <<"="<<x<<line;
#define debug2(x,y)  cerr << #x <<"="<<x<<" "<<#y<<"="<<y<<line;
#define debug3(x,y,z)cerr << #x <<"="<<x<<" "<<#y<<"="<<y<<" "<<#z<<"="<<z<<line;
#define debug4(x,y,z,a)cerr << #x <<"="<<x<<" "<<#y<<"="<<y<<" "<<#z<<"="<<z<<" "<<#a<<"="<<a<<line;
#define forn(i,n) for(ll (i) = 0 ; (i) < (n) ; ++(i))
#define for1(i,n) for(ll (i) = 1 ; (i) <= (n) ; ++(i))
#define forr(i,n) for(ll (i) = (n)-1 ; (i)>=0 ; --(i))
#define forab(i,a,b,c) for(ll (i) = a ; (i) <= (b) ; (i)+=(c))
#define forba(i,b,a,c) for(ll (i) = b ; (i) >= (a) ; (i)+=(c))

// #define pbds tree<ll, null_type, less<ll>, rb_tree_tag, tree_order_statistics_node_update>

#ifndef ONLINE_JUDGE
#define debug(x)    cerr << #x <<"="; _print(x); cerr << line;
#else
#define debug(x)
#endif

struct myhash
{
    static uint64_t fxn(uint64_t x)
    {
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }
 
    size_t operator ()(uint64_t x) const
    {
        static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();
        return fxn(x + FIXED_RANDOM);
    }
};

template <typename T>                                           void read(T &x){cin >> x;}
template <typename T , typename T0>                             void read(T &x,T0 &y){cin >> x >> y;}
template <typename T , typename T0 , typename T1>               void read(T &x,T0 &y,T1 &z){cin >> x >> y >> z;}
template <typename T , typename T0 , typename T1 , typename T2> void read(T &x,T0 &y,T1 &z,T2 &w){cin >> x >> y >> z >> w;}
template <typename T , typename T0>                             void read(pair< T , T0 > &p){cin >> p.fst >>p.scd;}
template <typename T>                                           void read(vector< T > &oneD,ll n){forn(i,n){T x;read(x);oneD.push_back(x);}}
template <typename T>                                           void read(T oneD[] , ll n){for(ll i=0;i<n;i++){read(oneD[i]);}}
template <typename T>                                           void write(T x){cout << x << " ";}
template <typename T>                                           void wrtline(T x){cout << x << line;}
template <typename T , typename T0>                             void write(T x,T0 y){cout << x << " " << y << "\n";}
template <typename T , typename T0 , typename T1>               void write(T x,T0 y,T1 z){cout << x << " " << y << " " << z << "\n";}
template <typename T , typename T0 , typename T1 , typename T2> void write(T x,T0 y,T1 z,T2 w){cout << x << " " << y << " " << z << " " << w << "\n";}
template <typename T , typename T0>                             void write(pair< T , T0 > &p){write(p.fst);write(p.scd);cout << line;}
template <typename T>                                           void write(vector< T > &oneD,ll n){for(ll i=0;i<n;i++){cout<<oneD[i]<<" ";}cout << line;}
template <typename T>                                           void write(T oneD[] ,int n){for(ll i=0;i<n;i++){write(oneD[i]);}cout <<line;}
template <typename T , typename T0>                             void write(map< T , T0 > &mpp){for(auto it : mpp){write(it.fst);cout << ": ";write(it.scd);cout << "\n";}cout<<line;}
template <typename T>                                           void write(vector<vector<T>> &rc,ll n,ll m,ll w){forn(i,n){forn(j,m){cerr<<setw(w)<<rc[i][j]<<" ";}cerr<<line;}}
template <typename T>                                           void _print(T &t) {cerr << t;}
template <class T, class V>                                     void _print(pair <T, V> &p);
template <class T>                                              void _print(vector <T> &v);
template <class T>                                              void _print(set <T> &v);
template <class T, class V>                                     void _print(map <T, V> &v);
template <class T>                                              void _print(multiset <T> &v);
template <class T>                                              void _print(set <T, greater<T>> &v);
template <class T, class V>                                     void _print(map <T, V, greater<T>> &v);
template <class T>                                              void _print(multiset <T, greater<T>> &v);
template <class T, class V>                                     void _print(unordered_map <T, V> &v);
template <class T, class V>                                     void _print(unordered_map <T, V, myhash> &v);
template <class T>                                              void _print(stack<T> st);
template <class T>                                              void _print(queue<T> q);
template <class T>                                              void _print(deque<T> dq);
template <class T>                                              void _print(priority_queue<T> pq);
template <class T>                                              void _print(priority_queue<T,vector<T>,greater<T>> pq);
template <class T, class V>                                     void _print(pair <T, V> &p) {cerr << "{"; _print(p.fst); cerr << ","; _print(p.scd); cerr << "}";}
template <class T>                                              void _print(vector <T> &v) {cerr << "[ "; for (T i : v) {_print(i); cerr << " ";} cerr << "]";}
template <class T>                                              void _print(set <T> &v) {cerr << "[ "; for (T i : v) {_print(i); cerr << " ";} cerr << "]";}
template <class T>                                              void _print(set <T, greater<T>> &v) {cerr << "[ "; for (T i : v) {_print(i); cerr << " ";} cerr << "]";}
template <class T>                                              void _print(priority_queue <T> pq) {cerr << "[ "; while(!pq.empty()) {T temp=pq.top(); pq.pop(); _print(temp); cerr << " ";} cerr << "]";}
template <class T>                                              void _print(priority_queue <T,vector<T>,greater<T>> pq) {cerr << "[ "; while(!pq.empty()) {T temp=pq.top(); pq.pop(); _print(temp); cerr << " ";} cerr << "]";}
template <class T>                                              void _print(stack <T> st) {cerr << "[ "; while(!st.empty()) {T temp=st.top(); st.pop(); _print(temp); cerr << " ";} cerr << "]";}
template <class T>                                              void _print(queue <T> q) {cerr << "[ "; while(!q.empty()) {T temp=q.front(); q.pop(); _print(temp); cerr << " ";} cerr << "]";}
template <class T>                                              void _print(deque <T> dq) {cerr << "[ "; while(!dq.empty()) {T temp=dq.front(); dq.pop_front(); _print(temp); cerr << " ";} cerr << "]";}
template <class T>                                              void _print(multiset <T> &v) {cerr << "[ "; for (T i : v) {_print(i); cerr << " ";} cerr << "]";}
template <class T>                                              void _print(multiset <T, greater<T>> &v) {cerr << "[ "; for (T i : v) {_print(i); cerr << " ";} cerr << "]";}
template <class T, class V>                                     void _print(map <T, V> &v) {cerr << "[ "; for (auto i : v) {_print(i); cerr << " ";} cerr << "]";}
template <class T, class V>                                     void _print(map <T, V, greater<T>> &v) {cerr << "[ "; for (auto i : v) {_print(i); cerr << " ";} cerr << "]";}
template <class T, class V>                                     void _print(unordered_map <T, V> &v) {cerr << "[ "; for (auto i : v) {_print(i); cerr << " ";} cerr << "]";}
template <class T, class V>                                     void _print(unordered_map <T, V, myhash> &v) {cerr << "[ "; for (auto i : v) {_print(i); cerr << " ";} cerr << "]";}



class UnionFind {
    public :
   ll size2;
   vector<ll> sz,id;
   ll numComponents;

  UnionFind(ll size1) {

    if (size1 <= 0)
    {
        cout<<"Not Allowed";
        return;
    }

    size2 = size1;
    numComponents=size1;

    for (ll i = 0; i <= size1; i++) {
      id.push_back(i);// Link to itself (self root)
      sz.push_back(1); // Each component is originally of size one
    }
  }

   ll find1(ll p) {

    // Find the root of the component/set
    ll root = p;
    while (root != id[root]) root = id[root];

    // Compress the path leading back to the root.
    // Doing this operation is called "path compression"
    // and is what gives us amortized time complexity.
    while (p != root) {
      ll next = id[p];
      id[p] = root;
      p = next;
    }

    return root;
  }

 
   bool connected(ll p,ll q) {
    return find1(p) == find1(q);
  }

   ll componentSize(ll p) {
    return sz[find1(p)];
  }

   ll sizecom() {
    return size2;
  }

  // Returns the number of remaining components/sets
   ll components() {
    return numComponents;
  }

   bool unify(ll p, ll q) {
    ll root1 = find1(p);
    ll root2 = find1(q);
    // These elements are already in the same group!
    if (root1 == root2){
        // debug1("NO")
        return 0;
    }

    // Merge smaller component/set into the larger one.
    if (sz[root1] < sz[root2]) {
      sz[root2] += sz[root1];
      id[root1] = root2;
      sz[root1]=1;
    } else {
      sz[root1] += sz[root2];
      id[root2] = root1;
      sz[root2]=1;
    }

    // Since the roots found are different we know that the
    // number of components/sets has decreased by one
    numComponents--;
    // debug1(numComponents)
    return 1;
  }
};
vll graph[(ll)2e5+5];
void dfs(ll par,ll curr,ll id[],ll lowLink[],vvll &bridge,ll &ct){
    if(id[curr]!=INT_MAX){
        return;
    }
    lowLink[curr]=id[curr]=ct;
    ct++;
    for(auto it: graph[curr]){
        if(it!=par){
            dfs(curr,it,id,lowLink,bridge,ct);
            lowLink[curr]=min(lowLink[curr],lowLink[it]);
            if(id[curr]<lowLink[it]){
                bridge.pb({min(curr,it),max(curr,it)});
            }
        }
    }
}
void graphBuilder(vvll &edges,ll n,ll m,string &ans){
    for1(i,n){
        graph[i].clear();
    }
    forn(i,m){
        if(ans[i]=='1'){
            graph[edges[i][0]].pb(edges[i][1]);
            graph[edges[i][1]].pb(edges[i][0]);
        }
    }
}
void solve()
{
    ll n,m;
    read(n,m);
    vvll edges;
    forn(i,m){
        ll x,y;
        read(x,y);
        edges.pb({min(x,y),max(x,y)});
    }
    // debug(edges)
    ll id[n+1],lowLink[n+1];
    for1(i,n){
        id[i]=lowLink[i]=INT_MAX;
    }
    vvll bridge;
    ll ct=1;
    UnionFind ob(n);
    string ans(m,'1');
    forn(i,m-n+1){
        // debug1(i)
        graphBuilder(edges,n,m,ans);
        dfs(-1,1,id,lowLink,bridge,ct);
        sort(all(bridge));
        forn(j,m){
            // debug(edges[j])
            if(ans[j]=='1' and !binary_search(all(bridge),edges[j]) and ob.unify(edges[j][0],edges[j][1])){
                ans[j]='0';
                break;
            }
        }
        ct=1;
        for1(i,n){
            id[i]=lowLink[i]=INT_MAX;
        }
        // debug(bridge)
        bridge.clear();
        // debug(ans)
    }
    wrtline(ans);
}
int main() {
#ifndef ONLINE_JUDGE
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stderr);
#endif
ios_base::sync_with_stdio(false);
cin.tie(NULL);
//*******************************************************//
test
solve();

return 0;
}
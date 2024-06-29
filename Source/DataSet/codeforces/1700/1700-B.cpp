#include<bits/stdc++.h>
#include<unordered_map>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
#include <functional> // for less
using namespace __gnu_pbds;
using namespace std;
template <typename T>
using ordered_set = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;
#define endl '\n' 
#define ll long long
// #define ll int
#define ull unsigned ll
#define sz(v) (int)(v.size())
#define ld long double
#define pb push_back
#define em emplace_back
#define ppb pop_back
#define all(x) (x).begin(), (x).end()
#define sai(a,n) sort(a,a+n);
#define sad(a,n) sort(a,a+n,greater<int>());
#define svi(x) sort(x.begin(), x.end());
#define svd(a) sort(a.begin(), a.end(), greater<ll>());
#define FAST ios_base::sync_with_stdio(0);cin.tie(0);
#define fi(i,x,n) for (ll i = x; i < n; ++i)
#define fr(i,x,n) for (ll i(x-1);i >= n; --i)
#define vi vector <ll>
#define vvi vector<vi>
#define mp(a1,a2) make_pair(a1,a2)
#define gll ll static
#define rll register ll 
#define reg register
#define pii pair<ll, ll>
#define vp vector<pii>
#define vs vector<string>
#define vb vector<bool>
#define vvb vector<vb>
#define vvp vector<vp>
#define read(a,n) fi(i,0,n){cin>>a[i];}
#define read_tree(g,m) fi(i,0,m) {ll u,v;cin>>u>>v;g[u].em(v);g[v].em(u);}
#define mii map<ll, ll>
#define umii unordered_map<ll, ll>
#define msi map<string, ll>
#define mss map<string, string>
#define pqi priority_queue <ll>
#define pqii priority_queue <ll,vi,greater<ll>>
#define mci map<char, ll>
#define umci unordered_map<char, ll> 
#define vmax(v) *max_element(all(v));
#define vmin(v) *min_element(all(v));
#define mdsu(s,e) fi(i,s,e+1) make_set(i);
#define ret(a) {cout<<a<<endl; return;}
#define F first
#define S second

#ifndef ONLINE_JUDGE
#define debug(x) cerr << #x <<" = "; _print(x); cerr << endl;
#else
#define debug(x)
#endif

template<typename T, typename U> static inline void amin(T &x, U y){if(y<x) x=y;}
template<typename T, typename U> static inline void amax(T &x, U y){if(x<y) x=y;}
template<typename T> vector<T>& operator-- (vector<T> &v){for (auto& i : v) --i;return  v;}
template<typename T> vector<T>& operator++ (vector<T> &v){for (auto& i : v) ++i;return  v;}
template<typename T> istream& operator>>(istream& is,  vector<T> &v){for (auto& i : v) is >> i; return is;}
template<typename T> ostream& operator<<(ostream& os,  vector<T>  &v){for (auto& i : v) os << i << ' '; return os;}
template<typename T> ostream& operator<<(ostream& os,  set<T>  &v){for (auto& i : v) os << i << ' '; return os;}
template<typename T> ostream& operator<<(ostream& os,  unordered_set<T>  &v){for (auto& i : v) os << i << ' '; return os;}
template<typename T, typename U> ostream& operator<<(ostream& os,  unordered_map<T,U>  &v){for (auto& i : v) os << i << ' '; return os;}
template<typename T, typename U> ostream& operator<<(ostream& os,  map<T,U>  &v){for (auto& i : v) os << i << ' '; return os;}
template<typename T, typename U> pair<T,U>& operator-- (pair<T, U> &p){--p.first; --p.second;return  p;}
template<typename T, typename U> pair<T,U>& operator++ (pair<T, U> &p){++p.first; ++p.second;return  p;}
template<typename T, typename U> istream& operator>>(istream& is, pair<T, U> &p){is >> p.first >> p.second;        return is;}
template<typename T, typename U> ostream& operator<<(ostream& os, pair<T, U>  p){os << p.first << ' ' << p.second; return os;}
template<typename T, typename U> pair<T,U> operator-(pair<T,U> a, pair<T,U> b){return mp(a.first-b.first, a.second-b.second);}
template<typename T, typename U> pair<T,U> operator+(pair<T,U> a, pair<T,U> b){return mp(a.first+b.first, a.second+b.second);}

void _print(ll t) {cerr << t;}
template<typename T> static inline void _print(T &t){cerr << t;}

bool pair_sec_inc(pii a, pii b){return a.second > b.second;}
bool pair_sec_dec(pii a, pii b){return a.second < b.second;}
bool pair_fir_dec(pii a, pii b){return a.first > b.first;}
bool pair_fir_dec_sec_inc(pii a, pii b){if (a.first == b.first){return a.second < b.second;}return a.first > b.first;}

template<typename T> 
struct Hasher
{
    size_t operator()(const vector<T> &myVector) const 
    {
        std::hash<T> hasher;
        size_t answer = 0;
        for (auto i : myVector) 
        {
            answer ^= hasher(i) + 0x9e3779b9 + 
                    (answer << 6) + (answer >> 2);
        }
        return answer;
    }
};

bool kahn();
void dfs(ll vertex);
void bfs(ll source);
void grid_dfs(ll i, ll j, vs &s, vvb &vis);
void grid_bfs(ll i, ll j, vs &s);
void dijkstra(ll source, vp g[], ll n);
void primes();
bool prmchk(ll n);
ll binpow(ll a, ll b);
ll binpowm(ll a, ll b, ll mod);
ll lcm(ll a, ll b);
string bin(ll x);
vi previousGreaterElement(vi &a, int n);

ll const MOD = 1e9+7;
ll const N = 2e5+5;
ll const Ng = 2e3+5;
ll const INF = 1e18;
vi in(N,0);

vi g[N];
vb vis(N);
vi level(N,INF);
vvi levelg(Ng, vi(Ng,INF));

int lp[N+1];
vector<int> pr;

int dx[] = {1,-1,0,0,-1,-1,1,1}, dy[] = {0,0,1,-1,1,-1,1,-1};

vi parent(N);
vi sizes(N);

void make_set(ll v)
{
    parent[v] = v;
    sizes[v] = 1;
}

ll find_set(ll v){

    if (parent[v] == v)
        return v;
    return parent[v] = find_set(parent[v]);
}

void union_sets(ll a, ll b)
{
    a = find_set(a);
    b = find_set(b);
    if (a!=b)
    {    
        if (sizes[a] < sizes[b])
            swap(a,b);
        parent[b] = a;
        
        // merge(a,b);
        sizes[a] += sizes[b];
    }
}


// vvb gvis[N][N];
// vvi g[N][N];


string s,m;
// vi an;
// ll ans(0);
bool isSmaller(string str1, string str2)
{
    // Calculate lengths of both string
    int n1 = str1.length(), n2 = str2.length();
 
    if (n1 < n2)
        return true;
    if (n2 < n1)
        return false;
 
    for (int i = 0; i < n1; i++)
        if (str1[i] < str2[i])
            return true;
        else if (str1[i] > str2[i])
            return false;
 
    return false;
}
ll n;
// ll k;
// ll m;
string findDiff(string str1, string str2)
{
    // Before proceeding further, make sure str1
    // is not smaller
    if (isSmaller(str1, str2))
        swap(str1, str2);
 
    // Take an empty string for storing result
    string str = "";
 
    // Calculate length of both string
    int n1 = str1.length(), n2 = str2.length();
 
    // Reverse both of strings
    reverse(str1.begin(), str1.end());
    reverse(str2.begin(), str2.end());
 
    int carry = 0;
 
    // Run loop till small string length
    // and subtract digit of str1 to str2
    for (int i = 0; i < n2; i++) {
        // Do school mathematics, compute difference of
        // current digits
 
        int sub
            = ((str1[i] - '0') - (str2[i] - '0') - carry);
 
        // If subtraction is less then zero
        // we add then we add 10 into sub and
        // take carry as 1 for calculating next step
        if (sub < 0) {
            sub = sub + 10;
            carry = 1;
        }
        else
            carry = 0;
 
        str.push_back(sub + '0');
    }
 
    // subtract remaining digits of larger number
    for (int i = n2; i < n1; i++) {
        int sub = ((str1[i] - '0') - carry);
 
        // if the sub value is -ve, then make it positive
        if (sub < 0) {
            sub = sub + 10;
            carry = 1;
        }
        else
            carry = 0;
 
        str.push_back(sub + '0');
    }
 
    // reverse resultant string
    reverse(str.begin(), str.end());
 
    return str;
}
void sigma(){

    cin>>n>>m;
    string s;
    if(m[0] == '9')
    {
        fi(i,0,n+1)
            s.pb('1');

        // fi(i,0,n)
        // {
        //     s.pb('9');
        // }
        // s.pb('1');

        debug(s)
        string x=findDiff(s,m);

        if(x[0] == '0') x = x.substr(1,n); 

        cout<<x;

        cout<<endl;
    }
    else
    {
        fi(i,0,n)cout<<'9'-m[i];cout<<endl;
    }
    

}


signed main(){

    FAST

    // primes();

    cout << setprecision(15) << fixed;
    ll tc = 1;

    cin>>tc;
    
    while(tc--){

        debug(tc)

        sigma();

    }
    
}

void dfs(ll vertex)
{
    if (vis[vertex]) return;

    debug(vertex)
    
    vis[vertex]=1;
    // visiting the vertex


    for(auto &child : g[vertex])
    {
        // before entering the child
        
        dfs(child);
        // after visiting that child's whole subtree

    }
    // after visiting all the subtrees of the children
    

}

void dijkstra(ll source, vp g[], ll n)
{

    vb vis(N,0);
    vi dist(N, MOD);

    // set<pii> q;
    priority_queue<pii, vp, greater<pii>> q;

    q.push({0,source});
    dist[source] = 0;

    while (!q.empty())
    {
        auto node = q.top();
        q.pop();
        ll v = node.second, d = node.first;
        
        if (vis[v]) continue;
        vis[v] = 1;
        for(auto child : g[v])
        {
            int child_v = child.first, wt = child.second;
            
            if (dist[v] + wt < dist[child_v])
            {
                dist[child_v] = dist[v] + wt;
                q.push({dist[child_v], child_v});
            }
            
        }


    }
    

}

void bfs(ll source){

    queue<ll> q;

    q.push(source);
    vis[source]=1;
    level[source] = 0;

    while (!q.empty())
    {
        auto curr_v = q.front();q.pop();

        // processing the source node       
        

        debug(curr_v)

        for(auto &child : g[curr_v])
        {

            if (!vis[child])
            {
                q.push(child);
                vis[child] = 1;
                level[child] = level[curr_v] + 1;
                // visiting the child now.

            }   
        }
    }
}

void grid_dfs(ll i, ll j, vs &g, vvb &vis)
{
    ll n = sz(g), m = sz(g[0]);

    if (i < 0 || i >= n || j < 0 || j >= m)
    {
        return;
    }
    
    if (vis[i][j]) return;
    
    debug(i) debug(j)

    vis[i][j]=1;
    
        fi(zz,0,4)
        {
            ll cx = i + dx[zz], cy = j + dy[zz];

            grid_dfs(cx,cy,g,vis);       
            

        }


}

void grid_bfs(ll i, ll j, vs &s)
{

    queue<pii> q;
    q.push({i,j});
    
    // auto valid = [&](ll x, ll y)
    // {
    //     return x >= 0 && x < sz(s) && y >= 0 && y < sz(s[0]);
    // };
    // vis[i][j] = 1;    
    levelg[i][j] = 0;
    while (!q.empty())
    {
        auto curr_v = q.front();
        q.pop();
        fi(zz,0,4)
        {
            ll cx = curr_v.first + dx[zz], cy = curr_v.second + dy[zz];

            // if (!valid(cx,cy))
            // {
            //     continue;
            // }

            if (levelg[cx][cy] > 1 + levelg[curr_v.first][curr_v.second])
            {
                
                levelg[cx][cy] = 1 + levelg[curr_v.first][curr_v.second];
                q.push({cx,cy});
            }
        }
    }
}

bool kahn(ll n) // for check if topological sorting is possible (only possible in case of DAGs)
{
    vi an;
    pqii q;
    fi(i,1,n+1)if(!in[i])q.push(i);

    while (!q.empty()) {
        ll v = q.top();
        q.pop();
        an.em(v);
        for (ll u : g[v]) {
        
            in[u]--;
            if (!in[u])
                q.push(u);
        }
    }
    return an.size() == n;
}

ll binpow(ll a, ll b) {
    ll res = 1;
    while (b > 0) {
        if (b & 1)
            res = res * a;
        a = a * a;
        b >>= 1;
    }
    return res;
}
ll binpowm(ll a, ll b, ll mod = MOD)
{
    ll res = 1;
    a%=mod;
    while(b)
    {
        if(b & 1)
            res = ((__int128)res * a) % mod;
        a = ((__int128)a * a) % mod;
        b >>= 1;
    }
    return res;
}

string bin(ll x){
    if (!x)
    {
        return "0";
    }
    string s;
    while (x)
    {
        s += '0' + x%2;
        x/=2;
    }
    // reverse(all(s));
    return s;
    
}
void primes(){

    for (int i=2; i<=N; ++i) {
        if (lp[i] == 0) {
            lp[i] = i;
            pr.push_back (i);
        }
        for (int j=0; j<(int)pr.size() && pr[j]<=lp[i] && i*pr[j]<=N; ++j)
            lp[i * pr[j]] = pr[j];
    }
}
bool prmchk(ll n)
{
    return (lp[n] == n);
}

ll lcm(ll a, ll b){
    return ((a*b)/__gcd(a,b));
}
vi previousGreaterElement(vi &a, int n)
{
    stack<int> s;   
    vi x(n);
    fi(i,0,n)
    {
        while (!s.empty() && a[s.top()] <= a[i])
        {
            s.pop();
        }
        if(!s.empty() && a[i] < a[s.top()])
        {
            x[i] = s.top();
        }
        else x[i] = -1;
        s.push(i);
    }
    
    return x;
}

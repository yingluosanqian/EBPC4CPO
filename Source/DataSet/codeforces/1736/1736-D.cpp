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
ll modi(ll x, ll m);
ll binpow(ll a, ll b);
ll binpowm(ll a, ll b, ll mod);
ll lcm(ll a, ll b);
string bin(ll x);
vi previousGreaterElement(vi &a, int n);
vi preSum(vi const &a);
vi sufSum(vi const &a);


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


string s;
// vi an;
// ll ans(0);

ll n;
// ll k;
// ll m;

// use sqrtl for precise square root of long long number.
// 0110 -> 01 10

void sigma(){
    cin>>n>>s;
    ll c1 = count(all(s),'0');
    if(c1%2)ret(-1)
    vi in;
    ll total_count(1);
    fi(i,0,n)
    {
        if(s[2*i] != s[2*i+1])
        {
            ll i1,i0;
            if(s[2*i] == '1')
            {
                i1 = 2*i;
                i0 = 2*i+1;
            }
            else
            {
                i1 = 2*i+1;
                i0 = 2*i;
            }
            in.em((total_count?i1:i0)+1);
            total_count = 1 - total_count;
        }
    }
    svi(in)
    cout<<sz(in)<<' '<<in<<endl;
    fi(i,0,n)cout<<1+2*i<<' ';
    cout<<endl;

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

vi sufSum(vi const &a){
    vi x=a;fr(i,sz(a)-1,0)x[i]+=x[i+1];return x;
}

vi preSum(vi const &a){
    vi x=a;fi(i,1,sz(a))x[i]+=x[i-1];return x;
}
ll modi(ll x, ll m)
{
    return binpowm(x,m-2,m);
}


/*

Kratos is here 

Killer
Yeah, it's crazy, I'm a (killer)
Made all this money from doin' this
D.A. got that dope
Now count it, five, ten, yeah, fifteen, twenty
Twenty-five, thirty, yeah, get the money
Throw it in the furnace, yeah, this shit be funny
Earn it just to burn it, swag drippin' from me
That's what I do with money, got money up the ass
Call it toilet paper, yeah, flushed with cash
Girl, nice butt, is it up for grabs?
Just wanna touch your ass, is that too much to ask? Yeah
I made it grip, I know it's tough to grasp, get the bag
Call it potato chips, I stuff in duffel bags
On some public transportation shit, 'cause I will bust your ass
Fuck the chain, I'm off the trailer hitch, I got a bunch of swag
Now count it, five, ten, yeah, fifteen, twenty
Twenty-five, thirty, yeah, get the money
Throw it in the furnace, yeah, this shit be funny
Earn it just to burn it, swag drippin' from me
Yeah, I'm a (killer)
Yeah, I'm a (what?), I'm a (what?), I'm a (killer)
Yeah, look, yeah
My income is all that and then some
Girl, your man is nincompoop, a symptom
Of a simp 'cause he'll spend some loot to get some
As for me, I'm like Kim Jong-Un, a pimp, son
Swag dripping, I'm in a pub
Went up to this chick, who was so tipsy, we went to hug
Ended up tripping, I picked her up
She yelled out it's her birthday
She's fifty and in the club
Then it comes on (yeah), that "In da Club" song (yeah)
She's a buzzsaw (what?), we goin' numbskull (uh)
I live on the edge, she's a jump off (yeah)
Call her Cinderella (why?), she loves balls (oh)
Now count it, five, ten, yeah, fifteen, twenty
Twenty-five, thirty, yeah, get the money
Throw it in the furnace, yeah, this shit be funny
Earn it just to burn it, swag drippin' from me
Yeah, I'm a (killer)
Attack like the Ripper
All over the track doin' laps like a stripper
Now (now), now (now), we'll vow (vow) it out (out)
Rap circles around ('round), surround sound (Sound)
John Rambo's back and my ammo stacked
And I'm cocking raps, I'm on your head
Other words, I'm stocking caps and I'm talking facts
Like OfficeMax
Never down, I'll be up like an insomniac
Girl, I got racks you gotta rack
How you got all that back and no body fat
I'm in awe with that
When I stop the Pontiac at the laundry mat that I saw you at
You almost had a heart attack when you met Cardiac
You ran inside, told your boyfriend like, "I'll be back"
But for all you know I probably act like I'm Daniel Wozniak
I'm a psycho-chopathic killer
I'm a cap peeler, caterpillar
With the botanic of bananas you ain't never heard
Better vernacular comin' after your scapular
For the lack of a better word Dracula
'Cause I'm attackin' a rapper in the phrenic nerve
I'm a savage back, put the dagger in the back of competitors
Predator and scavenger
I am a carnivore and a baller, you're at the dollar store
What the fuck you got a wallet for? Y'all are poor
I was livin' in squalor but uh-uh, not no more
Now I'm the one they holla for
Fuckin' shit up like a dinosaur in a China store
Bitch, I'm number five (what?), minus four (haha)
Now count it, five, ten, yeah, fifteen, twenty
Twenty-five, thirty, yeah, get the money
Throw it in the furnace, yeah, this shit be funny
Earn it just to burn it, swag drippin' from me
Yeah, I'm a (killer)
Yeah, I'm a (what?), I'm a (what?), I'm a (killer)
Yeah
*/
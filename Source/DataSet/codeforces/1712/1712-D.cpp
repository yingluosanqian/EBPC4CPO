
#include<bits/stdc++.h>
//#define endl "\n"
#define FF ios_base::sync_with_stdio(0);cin.tie(0)
#define binary(value, size) cout << bitset<size>(value) << '\n'
#define Tp template<class T>
#define Tpp template<typename T>
#define Tppp template<typename T1,typename T2>
#define eps 1e-9
#define pf printf
#define sf scanf
#define clr(arr,val) memset((arr),val,(sizeof(arr)))
#define rep(i,a,b) for(long long int i=a;i<b;i++)
#define repb(i,a,b) for(long long int i=a;i>=b;i--)
#define all(v) (v).begin(),(v).end()
#define asort(a) sort(a.begin(),a.end())
#define arev(a) reverse(a.begin(),a.end())
#define F first
#define S second
#define pb push_back
#define eb emplace_back
#define pbb pop_back
#define mp make_pair
#define V vector
#define P pair
#define M map
#define mt make_tuple
#define BS(v,x) binary_search(v.begin(),v.end(),x) //return true /false
#define LB(v,x) lower_bound(v.begin(),v.end(),x)-v.begin()//found and that value and not found than greater value pos
#define UB(v,x) upper_bound(v.begin(),v.end(),x)-v.begin() //found and greater value pos && not found also greater pos
#define sma(c) towlower(c)
#define rt(x) sqrt(x)
#define cap(c) towupper(c)
#define sq(a) ((a)*(a))
#define cube(a) ((a)*(a)*(a))
#define SUM(v) accumulate (v.begin(),v.end(),0)//sum of the vector
#define MAX(v) *max_element(v.begin(),v.end())//max element of the vector
#define MIN(v) *min_element(v.begin(),v.end())//min element of the vector
#define SZ(x) long long int(x.size())
#define N 10000000
#define Ceil(n) (long long int)ceil(n)
#define Floor(n) (long long int)floor(n)
#define deb(x) std::cout << #x << " = " << x << std::endl;
#define deb2(x,y)  std::cout << #x << " = " << x << ", "; std::cout << #y << " = " << y << std::endl;
#define deb3(x,y,z) std::cout << #x << " = " << x << ", "; std::cout << #y << " = " << y << ", "; std::cout << #z << " = " << z << std::endl;
#define deb4(x,y,z,r) std::cout << #x << " = " << x << ", "; std::cout << #y << " = " << y << ", "; std::cout << #z << " = " << z << ", ";std::cout << #r << " = " << r << std::endl;
#define out(ans) cout<<ans<<endl
#define outs(ans) cout<<ans<<" "<<endl
#define FI freopen ("in.txt", "r", stdin)
#define FO freopen ("out.txt", "w", stdout)

using namespace std;

typedef long long int ll;
typedef double lf;
typedef long double llf;
typedef unsigned long long int ull;
typedef vector<ll> vll;
typedef vector<vector<ll> > v2d;
typedef vector<vector<vector<ll> > > v3d;
typedef pair<ll,ll> pll;
typedef vector<pll> vpll;
typedef map<ll,ll> mpl;
typedef priority_queue<ll> heap;// heap max value from top
typedef priority_queue<ll, vector<ll>, greater<ll> > revheap;//heap min value from top

//vector<vector<int>> grid(n, vector<int>(n, 0)); // 2D Array inititalize with 0 n*n size

bool isLetter(char c) { return (c >= 'A' and c <= 'Z') or (c >= 'a' and c <= 'z');}
bool isUpperCase(char c) {return c >= 'A' and c <= 'Z';}
bool isLowerCase(char c) {    return c >= 'a' and c <= 'z';}
bool isDigit(char c) {return c >= '0' and c <= '9';}
bool isVowel(char c) {return c == 'a' or c == 'e' or c == 'i' or c == 'o' or c == 'u';}
bool isConsonant(char c){    return !isVowel(c); }

Tp void Debug(T v) { for(int i=0; i<(int)v.size(); i++)  cout << v[i] <<" ";   cout<<endl;  }

Tp void Input(T &v){ for(int i=0; i<(int)v.size(); i++) cin >> v[i]; }

Tp string toString(T n) { ostringstream ost; ost << n; ost.flush(); return ost.str();}

//string to int
string intTobinary(int x) { std::string binary = std::bitset<32>(x).to_string(); return binary;}
string lltobinary(ll x)  {std::string binary = std::bitset<64>(x).to_string(); return binary;}
ll toNumber(string s) {stringstream aa(s);ll mm;aa>>mm; return mm;}
ll binaryToDecimal(string n) { string num = n; ll dec_value = 0; ll base = 1; ll len = num.length(); for (int i = len - 1; i >= 0; i--) { if (num[i] == '1') dec_value += base; base = base * 2; } return dec_value;}

//nCr

Tpp T nCr(T n, T r) { if(r > n - r) r = n - r; int  ans = 1,i;for(i = 1; i <= r; i++) { ans *= n - r + i; ans /= i; }  return ans; }

//Binary exponentiation (a^p)
ll Binpow(ll a, ll p){ ll ret = 1; while(p>0){ if(p & 1)ret = (1LL * ret * a) ; a = (1LL * a * a) ; p >>= 1LL; } return ret; }

//BigMod (a^p)%mod
template<typename A, typename P>
ll BigMod(A a, P p, ll mod){ ll ret = 1; while(p){ if(p & 1)ret = ( (ret%mod) * (a%mod)) % mod; a = ( (a%mod) * (a%mod)) % mod; p >>= 1LL; } return ret; }

//Base (value,base)
Tpp T toBase(T n, T base){T ret= 0LL; while(n){ ret += n % base; ret *= 10LL;  n /= base;} return ret;}

// Divisor

Tpp  vector<T> Divisor(T value){ vector<T> v; for(int i = 1LL; i * i <= value; ++i){
    if(value % i == 0){v.push_back(i); if(i * i != value) v.push_back(value / i);} } return v;}

//Primes

Tpp  bool prime(T n){  if(n==2) return 1; if (n % 2 == 0) return 0; for (T i = 3; i*i <= n; i += 2){if (n % i == 0) return 0;}return 1;}

//Sieve
Tpp  void Sieve(T n){bool prime[n+1];memset(prime, true, sizeof(prime));
     for (T p=2; p*p<=n; p++){ if (prime[p] == true){
            for (int i=p*p; i<=n; i += p) prime[i] = false;} } }
//Math

Tpp   ll sum(std::vector<T> &v){return std::accumulate(all(v), 0);}
Tpp   T minval(std::vector<T> &v){return *std::min_element(all(v));}
Tpp   T maxval(std::vector<T> &v){return *std::max_element(all(v));}
Tpp   void make_unique(std::vector<T> &v){v.resize(unique(all(v)) - v.begin());}
Tpp   void make_unique_sorted(std::vector<T> &v){asort(v);v.resize(unique(all(v)) - v.begin());}
Tpp   int lowerBound(std::vector<T> &v, T x){return v.back() < x ? -1 : lower_bound(all(v), x) - v.begin();}
Tpp   int upperBound(std::vector<T> &v, T x){return v.back() < x ? -1 : upper_bound(all(v), x) - v.begin();}

double startTime = clock();
void showCurrentTime(){ printf("%.6lf\n", ((double)clock() - startTime) / CLOCKS_PER_SEC);}



//Bit Hacks
ll MakeOneBit (ll decimalvalue, int pos)  { return (decimalvalue | (1 << pos));}    //make pos bit of value 1//or ta pore na temp diye anle
ll MakeZeroBit(ll decimalvalue, int pos) {return (decimalvalue & ~(1 << pos));}    //make pos of value bit 0
ll FlipBit(ll decimalvalue, int pos)     { return (decimalvalue ^ (1 << pos));}    //flip pos bit of value reverse
bool CheckBit(ll decimalvalue, int pos) { return (decimalvalue & (1 << pos)); }
int MSB(ll k)  { return ( 63 - __builtin_clzll(k));}    // leftmost set bit
int LSB(ll k)  { return __builtin_ffs(k)-1 ;}    // right most set bit
int Totalset(ll decimalvalue)  {return __builtin_popcountll(decimalvalue); }   //total 1 in value

int Totolnotset(ll decimalvalue) { return MSB(decimalvalue) - Totalset(decimalvalue) + 1;}    //total 0 in value
bool ispow2(int i) { return i&&(i&-i)==i; }

ll nC2(ll n) { return (n * (n - 1)) / 2;}
ll nc3(ll n){ return (n * (n - 1) * (n - 2)) / 6; }
ll arithsum(ll n, ll a = 1, ll d = 1){ return (n * (a + a + (n - 1) * d) ) / 2; }
ll LCM(ll a, ll b){return (a / __gcd(a, b) ) * b;}
ll OnetoNsum(ll n){return (n*(n+1))/2;}
ll longdivision(string s,ll a){ ll ans=0; ll temp=0; rep(i,0,s.length()){ temp=temp*10+(s[i]-'0');ans+=(temp/a);ans*=10;temp=temp%a;}return (ans/10);}


const long long int INF=LLONG_MAX;
const double EPS = 1e-9;
const double PI = acos(-1.0);
const double PIby2=asin(1.0);
const int MOD = 1e9 + 7;

//debug
Tppp void dvpll(vector<pair<T1,T2> > v) { for(ll i=0;i<v.size();i++){deb(i);deb(v[i].F);deb(v[i].S);cout<<endl;}}
void dpos(vll v,ll pos){ cout<<pos<<"="<<v[pos]<<endl; }
Tppp void dmap(map<T1,T2> m) {ll i=0;for(auto x:m) {deb(i);deb(x.F);deb(x.S);cout<<endl;i++;}}
Tpp void dset(set<T> s) {for(auto x:s) {deb(x);cout<<endl;}}
Tpp void dmset(multiset<T> s) {for(auto x:s) {deb(x);cout<<endl;}}

Tpp void dpqueue(priority_queue<T> pq) {while(!pq.empty()){deb(pq.top());pq.pop();}    }
Tpp void dstack(stack<T> st) {while(!st.empty()) {deb(st.top());st.pop();}}
Tpp void dqueue(queue<T> q) {while(!q.empty()){ deb(q.front());q.pop(); }}

void dgraph(vector<ll> adj[],ll node){rep(i,0,node+1){deb(i);rep(j,0,adj[i].size()) {cout<<adj[i][j]<<" ";}cout<<endl;}cout<<endl;}

Tpp void d1d(vector<T> v){for(ll i=0;i<v.size();i++){ cout<<v[i]<<" "; }cout<<endl;}
Tpp void d2d(vector<vector<T> >&v,ll n,ll m) {cout<<"2d: "<<endl;for(ll i=0;i<n;i++){for(ll j=0;j<m;j++){cout<<v[i][j]<<" ";}cout<<endl;}}
Tpp void d3d(vector<vector<vector<T> > > &v,ll p,ll q,ll r){cout<<"3d: "<<endl;for(ll i=0; i<p; i++){for(ll j=0; j<q; j++){for(ll k=0; k<r; k++){cout<<"v[i="<<i<<"][j="<<j<<"][k="<<k<<"]=";cout<<v[i][j][k]<<endl;}}}}



//for debugging debfun1,fun2 etc




//Input Output

Tpp void Rvec(vector<T> &v,ll n){rep(i,0,n){T d;cin>>d;v.pb(d);}}
Tppp void Rvpll(vector<pair<T1,T2> > &v,ll n){rep(i,0,n){T1 a;T2 b;cin>>a>>b;v.pb(mp(a,b));} }
void Rgraph(vector<ll> adj[],ll edge ,ll indexbase ,bool directed ){rep(i,0,edge){ll a,b;cin>>a>>b;if(indexbase==0){a--;b--;}adj[a].pb(b);if(directed==0){adj[b].pb(a);}}}



//Graph

//DFS

//vll adj[1000001];bool vis[1000001];void init(ll n){rep(i,0,n+1){adj[i].clear();vis[i]=false;}}
//stack<ll> st;void DFS(ll val){vis[val]=1;st.push(val);for(auto child:adj[val]){if(!vis[child]) DFS(child);}}

//BFS

//vll adj[1000001];bool vis[1000001];
//ll level[1000001];ll parent[1000001];
//void BFS(ll source){ queue<ll> q; q.push(source);level[source]=0;vis[source]=1;while(!q.empty()){ll par=q.front();q.pop();for(auto child:adj[val]){if(!vis[child]){vis[child]=1;level[child]=level[val]+1;parent[child]=par;q.push(child);}}}}



//set/multiset.insert() map/vector/list.push_back()/queue/stack.push()/list.push_front()
//vec/list.pop_back()/stack/queue/pqueue.pop()
//pq.top()/queue.front()/stack.top()
//cout<<fixed<<setprecision(decimal)<<value<<endl;
/*
NCR dp

ll NCR[1010][1010];
const ll MM=10056;ll funNCR(ll n){NCR[1][1]=1;NCR[1][0]=1;for(ll i=2;i<=n;i++){NCR[i][0]=1;NCR[i][i]=1;for(ll j=1;j<=i-1;j++){NCR[i][j]=(NCR[i-1][j-1]%MM+NCR[i-1][j]%MM)%MM;NCR[i][j]%=MM;}}}


*/
//const int dx[]={1,0,-1,0};const int dy[]={0,1,0,-1}; //4 direction
//const int dx[] = {1, -1, 0, 0, -1, -1, 1, 1}; const int dy[] = {0, 0, 1, -1, -1, 1, -1, 1};// 8 direction
//const int dx[]={2,1,-1,-2,-2,-1,1,2};const int dy[]={1,2,2,1,-1,-2,-2,-1};//Knight Direction
//const int dx[]={2,1,-1,-2,-1,1};const int dy[]={0,1,1,0,-1,-1}; //Hexagonal Direction
//-------------------------------------------------------------------------------------------------------------------
//adj matrix cant be done dynamically adj[tot+1] cant be done
//vector<vector<int>> grid(n, vector<int>(n, 0)); 2D Array
//s.erase(remove(s.begin(),s.end(),' '),s.end());//INF=LONG_LONG_MAX;
//vector<vector<int> > adj(maxx+2)      //adjacency matrix
//adj set vector<vector<int> > adj(maxx) //adjacency set
//adj queue adj priority queue etc
// to flip bit do 1 xor
/*
V<ll> v1;
    v1.pb(7);v1.pb(1);v1.pb(18);
    d1d(v1);
    deb(MAX(v1));
    deb(MIN(v1));
    asort(v1);
    d1d(v1);
    deb(LB(v1,18));
    deb(UB(v1,18));
    deb3(v1[0],v1[1],v1[2]);
    deb(v1[0]);

    heap h1;h1.push(1);h1.push(2);
    deb(h1.top());h1.pop();
    revheap h2;h2.push(1);h2.push(2);
    deb(h2.top());h2.pop();
    deb(SUM(v1));
*/
bool check(ll val,vll v,ll n,ll k)
{

    ll cc=k;
    rep(i,0,n)
    {
        if(2*v[i]<val)
        {
            cc--;
            v[i]=1e9;
        }


    }

    if(cc<0) return 0;
    if(cc==0)
    {

        rep(i,0,n-1)
        {
          if(min(v[i],v[i+1])>=val) return 1;
        }
        //return 0;
        //if(maxx>=val) return 1;
        // return 0;
    }
    else if(cc==1)
    {
        if(MAX(v)>=val) return 1;
        else return 0;
    }
    else if(cc>1)
    {
        return 1;
    }
    return 0;


}

int main()
{
    FF;


    ll tc;cin>>tc;
    while(tc--)
    {
         ll n,k;cin>>n>>k;
         vll v;
         rep(i,0,n)
         {
             ll d;cin>>d;v.pb(d);
         }
         ll left=0,right=1e9,ans=0;
         while(left<=right)
         {
             ll mid=(left+right)/2;

             if(check(mid,v,n,k))
             {
                 ans=mid;
                 //deb(ans);
                 left=mid+1;
             }
             else
             {
                 right=mid-1;
             }
         }

         cout<<ans<<endl;


    }














    return 0;
}




 	  	     				 		 		 				
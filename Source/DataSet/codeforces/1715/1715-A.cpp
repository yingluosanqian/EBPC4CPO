// #define ONLINE_JUDGE 69

#include <bits/stdc++.h>
#include <chrono>

using namespace std;

using ll = long long;
using ull = unsigned long long;
using ld = long double;

// Macros
#define int long long
// #define ll long long
#define ld long double
#define vt vector
#define vi vector<int>
#define vvi vector<vi>
#define vii vector<pair<int,int>>
#define pok(ok) print(ok ? "YES" : "NO");

#define pii pair<int,int>

// Constants
constexpr ll SZ = 2e5 + 7;
constexpr ll inf = 9e18;
constexpr ll mod = 1e9 + 7;
constexpr ll MOD = 998244353;
constexpr ld PI = 3.141592653589793238462;
const ld scale = 1e6;


#define pb push_back
#define em emplace_back
#define all(X) (X).begin(), (X).end()
#define allr(X) (X).rbegin(), (X).rend()
#define sz(X) (int)X.size()

#define each(x, a) for (auto &x: a)
#define forn(i, n) for (int i = 0; i < n; ++i)
#define forr(i, n) for (int i = n; i >=0; --i)
#define rep(i,a,b) for(int i = a; (a <= b ? i <= b : i >= b); (a <= b ? i++ : i--))
// #define repr(i,a,b) for(int i = a; i >= b; i--)

#define fi first
#define se second
#define Endl '\n'

#define setbits(X) __builtin_popcountll(X)
#define fix(X) fixed << setprecision(X)
#define mem0(X) memset((X), 0, sizeof((X)))
#define mem1(X) memset((X), -1, sizeof((X)))


// Debug macro
string to_string(string s){return '"'+s+'"';}
string to_string(const char* s){return to_string((string)s);}
string to_string(const bool& b){return(b?"true":"false");}
template<class T>string to_string(T x){ostringstream sout;sout<<x;return sout.str();}
template<class A,class B>string to_string(pair<A,B> p){return "("+to_string(p.first)+", "+to_string(p.second)+")";}
template<class A>string to_string(const vector<A> v){
	int f=1;string res="{";for(const auto x:v){if(!f)res+= ", ";f=0;res+=to_string(x);}res+="}";
	return res;
}
template<class A>string to_string(const set<A> v){
	int f=1;string res="{";for(const auto x:v){if(!f)res+= ", ";f=0;res+=to_string(x);}res+="}";
	return res;
}
template<class A, class B>string to_string(const map<A, B> v){
	int f=1;string res="{";for(const auto x:v){if(!f)res+= ", ";f=0;res+=to_string(x);}res+="}";
	return res;
}
template<class A>string to_string(const multiset<A> v){
	int f=1;string res="{";for(const auto x:v){if(!f)res+= ", ";f=0;res+=to_string(x);}res+="}";
	return res;
}
void debug_out(){puts("");}
template<class T,class... U>void debug_out(const T& h,const U&... t){cerr<<" "<<to_string(h);debug_out(t...);}
#ifndef ONLINE_JUDGE 
#define dbg(...) cerr<<"["<<#__VA_ARGS__<<"]:",debug_out(__VA_ARGS__),cerr<<"\n";
#else
#define dbg(...) 233;
#endif

int modpow(int a, int b, int m = mod) {
    a = a % m; int ans = 1;
    while (b) {
        if (b & 1) { ans = (ans * a) % m; }
        b = b >> 1; a = (a * a) % m;
    }
    return ans;
}

// Inverse Mod (1 / a) % mod
int modinv(int a, int m = mod) { return modpow(a, m - 2, m); }

// Modular Arithematic
int modadd(int a, int b, int m = mod) { a = a % m; b = b % m; return (((a + b) % m) + m) % m; }
int modsub(int a, int b, int m = mod) { a = a % m; b = b % m; return (((a - b) % m) + m) % m; }
int modmul(int a, int b, int m = mod) { a = a % m; b = b % m; return (((a * b) % m) + m) % m; }
int moddiv(int a, int b, int m = mod) { a = a % m; b = b % m; return (modmul(a, modinv(b, m), m) + m) % m; }

// GCD
int gcd(int a, int b) { if (b == 0) { return abs(a); } return gcd(b, a % b); }

// LCM
int lcm(int a, int b) { return (a / gcd(a, b)) * b; }

// Read 
template<typename T1, typename T2> // cin >> pair<T1, T2>
istream& operator>>(istream &istream, pair<T1,T2> &p){ return (istream >> p.first >> p.second); }
template<typename T> // cin >> vector<T>
istream& operator>>(istream &istream, vector<T> &v){
  for(auto &it: v) cin >> it;
  return istream;
}
// Print 
template<typename T1, typename T2> // cout << pair<T1, T2>
ostream& operator<<(ostream &ostream, const pair<T1, T2> &p){ return (ostream << p.first << " " << p.second); }
template<typename T> //cout << vector<T>
ostream& operator<<(ostream &ostream, const vector<T> &c){ for (auto &it: c) cout << it << " "; return ostream;}

// Utility functions
template<typename T>
void print(T &&t) {cout << t << Endl;}
template <typename T, typename... Args>
void print(T &&t, Args &&... args){
  cout << t << " ";
  print(forward<Args>(args)...);
}

template<int M>
struct modint {
 
    static int _pow(int n, int k) {
        int r = 1;
        for (; k > 0; k >>= 1, n = (n*n)%M)
            if (k&1) r = (r*n)%M;
        return r;
    }
 
    int v; modint(int n = 0) : v(n%M) { v += (M&(0-(v<0))); }
    
    friend string to_string(const modint n) { return to_string(n.v); }
    friend istream& operator>>(istream& i, modint& n) { return i >> n.v; }
    friend ostream& operator<<(ostream& o, const modint n) { return o << n.v; }
    template<typename T> explicit operator T() { return T(v); }
 
    friend bool operator==(const modint n, const modint m) { return n.v == m.v; }
    friend bool operator!=(const modint n, const modint m) { return n.v != m.v; }
    friend bool operator<(const modint n, const modint m) { return n.v < m.v; }
    friend bool operator<=(const modint n, const modint m) { return n.v <= m.v; }
    friend bool operator>(const modint n, const modint m) { return n.v > m.v; }
    friend bool operator>=(const modint n, const modint m) { return n.v >= m.v; }
 
    modint& operator+=(const modint n) { v += n.v; v -= (M&(0-(v>=M))); return *this; }
    modint& operator-=(const modint n) { v -= n.v; v += (M&(0-(v<0))); return *this; }
    modint& operator*=(const modint n) { v = (v*n.v)%M; return *this; }
    modint& operator/=(const modint n) { v = (v*_pow(n.v, M-2))%M; return *this; }
    friend modint operator+(const modint n, const modint m) { return modint(n) += m; }
    friend modint operator-(const modint n, const modint m) { return modint(n) -= m; }
    friend modint operator*(const modint n, const modint m) { return modint(n) *= m; }
    friend modint operator/(const modint n, const modint m) { return modint(n) /= m; }
    modint& operator++() { return *this += 1; }
    modint& operator--() { return *this -= 1; }
    modint operator++(signed) { modint t = *this; return *this += 1, t; }
    modint operator--(signed) { modint t = *this; return *this -= 1, t; }
    modint operator+() { return *this; }
    modint operator-() { return modint(0) -= *this; }
 
    // O(logk) modular exponentiation
    modint pow(const int k) const {
        return k < 0 ? _pow(v, M-1-(-k%(M-1))) : _pow(v, k);
    }
    modint inv() const { return _pow(v, M-2); }
}; // in case of error remove ++ to += 1

using modi = modint<998244353>;
using modx = modint<(int)1e9+7>;

int accumulate(vi &a, function<int(int,int)> op = [] (int x, int y) {return x+y;}){
  if(sz(a) == 0) return 0;
  int ans = a[0];
  for(int i = 1; i < sz(a); i++) ans = op(ans, a[i]);
  return ans;
}

int MSB(int n){
  int i = 0;
  while(n){
    i++;
    n >>= 1;
  }
  i--;
  return i;
}

int nc2(int x){
  if(x <= 1) return 0;
  else return (x*(x-1))/2;
}

int ceil(int x, int y){
  if(x >= 0) return (x+y-1)/y;
  else return x/y;
}

const int N = 2e5+5;
// vector<int> prime(N+1, true);
// vector<modi> fact(N+10);
// vi primes;
void preSolve(){
  // prime[0] = prime[1] = false;
  // for(int i = 2; i <= N; i++){
  //   if(prime[i]){
  //     for(int j = 2*i; j <= N; j += i){
  //       prime[j] = 0;
  //     }
  //   }
  // }
  // rep(i,1,N) if(prime[i]) primes.pb(i);
  // fact[0] = 1;
  // for(int i = 1; i <= N; i++) fact[i] = fact[i-1]*i;
}

int n = -inf, m = -inf;
vector<vector<int>> adj;
vector<int> vis;

bool cmp(pair<int,int> a, pair<int,int> b){
    if(a.first != b.first) return a.first < b.first;
    return a.second > b.second;
}

// modi ncr(int n, int r){
//   if(r < 0 || r > n) return 0;
//   return fact[n]/(fact[r]*fact[n-r]);
// }
// modi npr(int n, int r){
//   if(r < 0 || r > n) return 0;
//   return fact[n]/fact[n-r];
// }

void solve(){
  int k,i, j, q, x, y;
  k = i = j = q = x = y = n = m = -1;
  cin >> x >> y;
  if(x == 1 && y == 1) print(0);
  else if(min(x,y) == 1) print(max(x,y));
  else print(2*min(x,y)+max(x,y)-2);
}


signed main(){
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

#ifndef ONLINE_JUDGE 
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
  auto start = std::chrono::system_clock::now();
#endif

  preSolve();

  int t = 1;
  cin >> t;
  rep(tt, 1,t){
    // dbg(t);
    // cout << "Case #" << tt << ": ";
    solve();
#ifndef ONLINE_JUDGE 
    auto end = std::chrono::system_clock::now();
    std::cerr << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << Endl;
#endif
  }
#ifndef ONLINE_JUDGE 
  auto end = std::chrono::system_clock::now();
  std::cerr << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << Endl;
#endif
}

/* stuff you should look for
  * constraints
	* int overflow, array bounds
	* special cases (n=1, n = 0?)
	* do smth instead of nothing and stay organized
	* WRITE STUFF DOWN
	* DON'T GET STUCK ON ONE APPROACH
*/
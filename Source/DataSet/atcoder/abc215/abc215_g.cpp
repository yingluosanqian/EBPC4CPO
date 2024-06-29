#include<bits/stdc++.h>
using namespace std;
using uint = unsigned int;
using ll = long long;
using ull = unsigned long long;
template<class T> using V = vector<T>;
template<class T> using VV = V<V<T>>;
template<class T> using VVV = V<VV<T>>;
template<class T> using VVVV = VV<VV<T>>;
#define rep(i,n) for(ll i=0ll;i<n;i++)
#define REP(i,a,n) for(ll i=a;i<n;i++)
const long long INF = (1LL << 60);
const long long mod99 = 998244353;
const long long mod107 = 1000000007;
const long long mod = mod99;
#define eb emplace_back
#define pb eb
#define be(v) (v).begin(),(v).end()
#define all(i,v) for(auto& i : v)
#define all2(i,j,v) for(auto& [i,j] : v)
template<class T, class U> void chmin(T& t, const U& u) { if (t > u) t = u; }
template<class T, class U> void chmax(T& t, const U& u) { if (t < u) t = u; }

// cin.tie(nullptr);
// ios::sync_with_stdio(false);
// cout << fixed << setprecision(20);

vector<long long> fac, inv, finv;
void COMinit(long long n = 200100, long long p = mod){
  fac.resize(n, 1);
  inv.resize(n, 1);
  finv.resize(n, 1);
  
  for(long long i = 2; i < n; i ++){
    fac[i] = fac[i-1] * i % p;
    inv[i] = p - inv[p % i] * (p / i) % p;
    finv[i] = finv[i-1] * inv[i] % p;
  }
}

long long modcom(long long n, long long k, long long p = mod){
    if (n < k) return 0;
    if (n < 0 || k < 0) return 0;
    return fac[n] * finv[k] % p * finv[n-k] % p;
}

long long modpow(long long n, long long k, long long p = mod){
  long long a = n % p;
  long long ans = 1;
  while(k != 0) {
    if(k & 1) ans = ans * a % p;
    k /= 2;
    a = a * a % p;
  }
  
  return ans;  
}

// n^(-1) ≡ b (mod p) となる b を求める
long long modinv(long long n, long long p = mod) { 
//  if(n == 1) return 1;
//  return p - modinv(p % n) * (p / n) % p;
  return modpow(n, p - 2, p);
}

// n^k ≡ b (mod p) となる最小の k を求める
long long modlog(long long n, long long b, long long p = mod){
  
  long long sqrt_p = sqrt(p);
  map<long long , long long> n_pow;
  long long memo = 1;
  
  for(long long i = 0; i < sqrt_p; i ++){
    if(!n_pow.count(memo)) n_pow[memo] = i;
    memo = memo * n % p; 
  }
  
  memo = modinv(memo, p);
  long long ans = 0;
  while(!n_pow.count(b)){
    if(ans >= p) return -1;
    ans += sqrt_p;
    b = b * memo % p;
  }

  ans += n_pow[b];
  return ans % (p - 1);

}

// ax + by = gcd(a, b) を満たす (x, y) が格納される
long long ext_gcd(long long a, long long b, long long &x, long long &y){
  if(b == 0){
    x = 1;
    y = 0;
    return a;
  }
  long long d = ext_gcd(b, a%b, y, x);
  y -= a/b*x;
  return d;
}


int main(){
  COMinit(100100, mod99);
  ll n;
  cin >> n;
  map<ll,ll> mp;
  rep(i,n){
    ll a;
    cin >> a;
    mp[a]++;
  }
  V<ll> v, w;
  map<ll,ll> mmp;
  for(auto[a,b]:mp) mmp[b]++;
  for(auto[a,b]:mmp){
    v.eb(a);
    w.eb(b);
  }
  
  ll m=v.size();
  REP(i,1,n+1){
    ll ans = 0;
    ll inv = modinv(modcom(n,i));
    rep(j,m){
      ll x=v[j];
      ll y=w[j];
      ll p = 1 - modcom(n-x, i)*inv%mod99;
      ans = (ans + y*p)%mod99;
    }
    ans %= mod99;
    ans += mod99;
    ans %= mod99;
    cout << ans << endl;
  }
  // cout << endl;
  
}

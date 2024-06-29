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

template <typename T>
struct SegmentTree { 
  vector<T> seg;
  ll seg_size;//È~¤ÎÊý
  
  T e_(){
    return 0;
  }
  T op(T L, T R){
    return (L + R)%mod107;
  }
    
  SegmentTree(ll N) : seg(4*N, e_()), seg_size() {
    seg_size = 1;
    while(seg_size < N) seg_size *= 2;
  }
  
  
  void update(ll idx, T x){
    idx += seg_size - 1;
    seg[idx] = x;
    
    while(idx > 0){
      idx = (idx - 1) / 2;
      seg[idx] = op(seg[2*idx + 1], seg[2*idx + 2]);
    }
  }
 

  T quere(ll a, ll b) { return quere_sub(a, b, 0, 0, seg_size); }
  T quere_sub(ll a, ll b, ll idx, ll left, ll right){
    if(a >= right || b <= left) return e_();
    if(a <= left && b >= right) return seg[idx];
  
    T v1 = quere_sub(a, b, 2 * idx + 1, left, (left + right) / 2);
    T v2 = quere_sub(a, b, 2 * idx + 2, (left + right) / 2, right);
    return op(v1, v2);
  }
  
};

int main(){
  string s;
  cin >> s;
  ll n=s.size();
  SegmentTree<ll> seg(n+100);
  V<ll> last(26, -1);
  rep(i,n){
    ll p=s[i]-'a';
    ll a = 0;
    if(last[p] == -1) a++;
    seg.update(i, seg.quere(max(0ll, last[p]-1), max(0ll,i-1))+a);
    last[p] = i;
  }
  // rep(i,n) cout << seg.quere(i,i+1) << endl;
  cout << seg.quere(0, n+50) << endl;
}


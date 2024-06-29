#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define ull unsigned long long
#define fore(i,a) for(auto &i:a)
#define bc(S) __builtin_popcount(S)
#define rep(i,a,b) for(ll i = a; i < b; i++)
#define rrep(i,a,b) for(ll i = a; i >= b; i--)
#define all(obj) (obj).begin(), (obj).end()
using Pl = pair<ll, ll>;
using Pi = pair<int, int>;
using Vl = vector<ll>;
using VVl = vector<Vl>;
using Vi = vector<int>;
using VVi = vector<Vi>;
using Vb = vector<bool>;
using VVb = vector<Vb>;
#define pb push_back
// #define mp make_pair
#define ft first
#define sd second
#define ld long double
const int inf = INT_MAX;
const ll infl = LLONG_MAX;
const ld infd = LDBL_MAX;
using Point = complex<double>;
template <class T, class U>
bool chmax(T &a, const U &b) { return a < b ? (a = b, 1) : 0; }
template <class T, class U>
bool chmin(T &a, const U &b) { return a > b ? (a = b, 1) : 0; }
template<typename T> using min_priority_queue = priority_queue<T, vector<T>, greater<T>>;

const ll MOD = 998244353;

signed main() {
  cin.tie(0);
  ios_base::sync_with_stdio(false);

  int n;cin >> n;
  string S;cin >> S;
  map<char,int> mp;
  Vi word(n);
  int K=-1;
  rep(i,0,n) {
    if (mp.find(S[i])==mp.end())
      mp[S[i]]=++K;
    word[i]=mp[S[i]];
  }
  ++K;
  vector<VVl> dp(n,VVl(1<<K,Vl(K,0)));
  dp[0][1<<word[0]][word[0]]=1;
  ll  ans=0;
  for (int i=1;i<n;i++) {
    for (ll A=0;A<(1<<K);A++) {
      for (int j=0;j<K;j++) {
        int u=word[i];
        ll  sum=0;
        if (A==1<<u and j==u)
          sum=1;
        sum=(sum+dp[i-1][A][j])%MOD;
        if (A>>u&1 and j==u) {
          for (int l=0;l<K;l++) {
            if (l==u) sum=(sum+dp[i-1][A][l])%MOD;
            else sum=(sum+dp[i-1][A^(1<<u)][l])%MOD;
          }
        }
        dp[i][A][j]=sum;
        if (i==n-1) ans=(ans+dp[i][A][j])%MOD;
      }
    }
  }
  if (n==1) ans=1;
  cout << ans << endl;

  return 0;
}
#include <bits/stdc++.h>
using namespace std;
#define FOR(i,m,n) for(int i=(m);i<(n);++i)
#define REP(i,n) FOR(i,0,n)
using ll = long long;
constexpr int INF = 0x3f3f3f3f;
constexpr long long LINF = 0x3f3f3f3f3f3f3f3fLL;
constexpr double EPS = 1e-8;
constexpr int MOD = 998244353;
// constexpr int MOD = 1000000007;
constexpr int DY4[]{1, 0, -1, 0}, DX4[]{0, -1, 0, 1};
constexpr int DY8[]{1, 1, 0, -1, -1, -1, 0, 1};
constexpr int DX8[]{0, -1, -1, -1, 0, 1, 1, 1};
template <typename T, typename U>
inline bool chmax(T& a, U b) { return a < b ? (a = b, true) : false; }
template <typename T, typename U>
inline bool chmin(T& a, U b) { return a > b ? (a = b, true) : false; }
struct IOSetup {
  IOSetup() {
    std::cin.tie(nullptr);
    std::ios_base::sync_with_stdio(false);
    std::cout << fixed << setprecision(20);
  }
} iosetup;

template <int ID>
struct MInt {
  unsigned int v;

  constexpr MInt() : v(0) {}
  MInt(const long long x) : v(x >= 0 ? x % mod() : x % mod() + mod()) {}
  static constexpr MInt raw(const int x) {
    MInt x_;
    x_.v = x;
    return x_;
  }

  static int get_mod() { return mod(); }
  static void set_mod(const unsigned int divisor) { mod() = divisor; }

  static void init(const int x) {
    inv<true>(x);
    fact(x);
    fact_inv(x);
  }

  template <bool MEMOIZES = false>
  static MInt inv(const int n) {
    // assert(0 <= n && n < mod() && std::gcd(x, mod()) == 1);
    static std::vector<MInt> inverse{0, 1};
    const int prev = inverse.size();
    if (n < prev) return inverse[n];
    if constexpr (MEMOIZES) {
      // "n!" and "M" must be disjoint.
      inverse.resize(n + 1);
      for (int i = prev; i <= n; ++i) {
        inverse[i] = -inverse[mod() % i] * raw(mod() / i);
      }
      return inverse[n];
    }
    int u = 1, v = 0;
    for (unsigned int a = n, b = mod(); b;) {
      const unsigned int q = a / b;
      std::swap(a -= q * b, b);
      std::swap(u -= q * v, v);
    }
    return u;
  }

  static MInt fact(const int n) {
    static std::vector<MInt> factorial{1};
    if (const int prev = factorial.size(); n >= prev) {
      factorial.resize(n + 1);
      for (int i = prev; i <= n; ++i) {
        factorial[i] = factorial[i - 1] * i;
      }
    }
    return factorial[n];
  }

  static MInt fact_inv(const int n) {
    static std::vector<MInt> f_inv{1};
    if (const int prev = f_inv.size(); n >= prev) {
      f_inv.resize(n + 1);
      f_inv[n] = inv(fact(n).v);
      for (int i = n; i > prev; --i) {
        f_inv[i - 1] = f_inv[i] * i;
      }
    }
    return f_inv[n];
  }

  static MInt nCk(const int n, const int k) {
    if (n < 0 || n < k || k < 0) [[unlikely]] return MInt();
    return fact(n) * (n - k < k ? fact_inv(k) * fact_inv(n - k) :
                                  fact_inv(n - k) * fact_inv(k));
  }
  static MInt nPk(const int n, const int k) {
    return n < 0 || n < k || k < 0 ? MInt() : fact(n) * fact_inv(n - k);
  }
  static MInt nHk(const int n, const int k) {
    return n < 0 || k < 0 ? MInt() : (k == 0 ? 1 : nCk(n + k - 1, k));
  }

  static MInt large_nCk(long long n, const int k) {
    if (n < 0 || n < k || k < 0) [[unlikely]] return MInt();
    inv<true>(k);
    MInt res = 1;
    for (int i = 1; i <= k; ++i) {
      res *= inv(i) * n--;
    }
    return res;
  }

  MInt pow(long long exponent) const {
    MInt res = 1, tmp = *this;
    for (; exponent > 0; exponent >>= 1) {
      if (exponent & 1) res *= tmp;
      tmp *= tmp;
    }
    return res;
  }

  MInt& operator+=(const MInt& x) {
    if ((v += x.v) >= mod()) v -= mod();
    return *this;
  }
  MInt& operator-=(const MInt& x) {
    if ((v += mod() - x.v) >= mod()) v -= mod();
    return *this;
  }
  MInt& operator*=(const MInt& x) {
    v = (unsigned long long){v} * x.v % mod();
    return *this;
    }
  MInt& operator/=(const MInt& x) { return *this *= inv(x.v); }

  auto operator<=>(const MInt& x) const = default;

  MInt& operator++() {
    if (++v == mod()) [[unlikely]] v = 0;
    return *this;
  }
  MInt operator++(int) {
    const MInt res = *this;
    ++*this;
    return res;
  }
  MInt& operator--() {
    v = (v == 0 ? mod() - 1 : v - 1);
    return *this;
  }
  MInt operator--(int) {
    const MInt res = *this;
    --*this;
    return res;
  }

  MInt operator+() const { return *this; }
  MInt operator-() const { return raw(v ? mod() - v : 0); }

  MInt operator+(const MInt& x) const { return MInt(*this) += x; }
  MInt operator-(const MInt& x) const { return MInt(*this) -= x; }
  MInt operator*(const MInt& x) const { return MInt(*this) *= x; }
  MInt operator/(const MInt& x) const { return MInt(*this) /= x; }

  friend std::ostream& operator<<(std::ostream& os, const MInt& x) {
    return os << x.v;
  }
  friend std::istream& operator>>(std::istream& is, MInt& x) {
    long long v;
    is >> v;
    x = MInt(v);
    return is;
  }

 private:
  static unsigned int& mod() {
    static unsigned int divisor = 0;
    return divisor;
  }
};

// https://twitter.com/catupper/status/1548346566087364609
// https://twitter.com/heno_code/status/1548346980400738305
// https://twitter.com/_su1sen/status/1548348443340713984
// https://twitter.com/_su1sen/status/1548348795477696512
// https://kmjp.hatenablog.jp/entry/2023/08/17/0930
int main() {
  using ModInt = MInt<0>;
  int n, p; cin >> n >> p;
  ModInt::set_mod(p);
  vector<vector<int>> graph(n);
  REP(_, n - 1) {
    int u, v; cin >> u >> v; --u; --v;
    graph[u].emplace_back(v);
    graph[v].emplace_back(u);
  }
  const auto dp_ = [&](auto dp_, const int par, const int ver) -> vector<ModInt> {
    vector<vector<ModInt>> dps, s;
    dps.reserve(graph[ver].size());
    s.reserve(graph[ver].size());
    for (const int u : graph[ver]) {
      if (u == par) continue;
      dps.emplace_back(dp_(dp_, ver, u));
      s.emplace_back(dps.back());
      FOR(i, 2, n) s.back()[i] += s.back()[i - 1];
    }
    const int c = dps.size();
    vector l(c, vector(n, ModInt::raw(1))), r(c, vector(n, ModInt::raw(1)));
    FOR(j, 1, n) FOR(i, 1, c) l[i][j] = l[i - 1][j] * s[i - 1][j];
    FOR(j, 1, n) for (int i = c - 2; i >= 0; --i) {
      r[i][j] = r[i + 1][j] * s[i + 1][j];
    }
    vector<ModInt> dp(n, 0);
    REP(u, c) {
      vector<ModInt> dp_u(n, 0);
      FOR(j, 1, n) dp_u[j] = dp_u[j - 1] + l[u][j] * r[u][j];
      FOR(i, 2, n) dp[i] += dp_u[i - 1] * dps[u][i];
    }
    if (c == 0) {
      fill(next(dp.begin()), dp.end(), 1);
    } else {
      FOR(i, 1, n) dp[i] += l.back()[i] * s.back()[i];
    }
    // cout << '[' << ver + 1 << ']';
    // for (const ModInt& x : dp) cout << ' ' << x;
    // cout << '\n';
    return dp;
  };
  vector<ModInt> dp(n, 1);
  for (const int u : graph[0]) {
    vector<ModInt> dp_u = dp_(dp_, 0, u);
    FOR(i, 2, n) dp_u[i] += dp_u[i - 1];
    FOR(i, 1, n) dp[i] *= dp_u[i];
  }
  vector<ModInt> ans(n, 0);
  FOR(k, 1, n) {
    ans[k] = dp[k];
    REP(j, k) ans[k] -= ModInt::nCk(k, j) * ans[j];
  }
  FOR(k, 1, n) cout << ans[k] << " \n"[k + 1 == n];
}

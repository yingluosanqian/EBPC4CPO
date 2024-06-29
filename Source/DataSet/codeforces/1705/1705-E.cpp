#include <bits/stdc++.h>
using namespace std;

#define _ << ' ' <<
using ll = long long;
using ull = unsigned long long;
using ld = long double;
using str = string;
#define mset multiset
#define all(x) (x).begin(), (x).end()
#define sz(x) int((x).size())
#define g0 get<0>
#define g1 get<1>
#define g2 get<2>
#define g3 get<3>
#define mill (int)(1e6)
#define bill (int)(1e9)
#define fun function
#define vec vector
#define vi vector<int>
#define vvi vector<vi>
#define vvvi vector<vvi>
#define vl vector<ll>
#define vvl vector<vector<ll>>
#define vvvl vector<vector<vector<ll>>>
#define vb vector<bool>
#define vvb vector<vector<bool>>
#define vd vector<double>
#define vvd vector<vector<double>>
#define vld vector<long double>
#define vvld vector<vector<long double>>
#define vc vector<char>
#define vvc vector<vector<char>>
#define vs vector<string>
#define vvs vector<vector<string>>
#define pii pair<int, int>
#define pll pair<ll, ll>
#define mt make_tuple
#define tii tuple<int, int>
#define vtii vector<tuple<int, int>>
#define tiii tuple<int, int, int>
#define vtiii vector<tuple<int, int, int>>
#define tiiii tuple<int, int, int, int>
#define vtiiii vector<tiiii>
#define vtll vector<tuple<ll, ll>>
#define vtli vector<tuple<ll, int>>
#define vtil vector<tuple<int, ll>>
#define vtill vector<tuple<int, ll, ll>>
#define vtlll vector<tuple<ll, ll, ll>>
#define vvtii vector<vtii>
#define vvvtii vector<vvtii>
#define vvtiii vector<vtiii>
#define vvtll vector<vtll>
#define vvtli vector<vtli>
#define vvtil vector<vtil>
#define vvtill vector<vtill>
#define vvtlll vector<vtlll>
#define tll tuple<ll, ll>
#define tli tuple<ll, int>
#define tlii tuple<ll, int, int>
#define til tuple<int, ll>
#define till tuple<int, ll, ll>
#define tlll tuple<ll, ll, ll>
#define qi queue<int>
#define qll queue<ll>
#define qtii queue<tii>
#define si set<int>
#define sl set<ll>
#define usi uset<int>
#define sc set<char>
#define usc uset<char>
#define stii set<tii>
#define ustii uset<tii>
#define mii map<int, int>
#define umii umap<int, int>
#define mci map<char, int>
#define umci umap<char, int>
#define mli map<ll, int>
#define umli umap<ll, int>
#define mil map<int, ll>
#define umil umap<int, ll>
#define mll map<ll, ll>
#define umll umap<ll, ll>
#define mivi map<int, vi>
#define umivi umap<int, vi>
#define mivl map<int, vl>
#define umivl umap<int, vl>
#define mlvi map<ll, vi>
#define umlvi umap<ll, vi>
#define mlvl map<ll, vl>
#define umlvl umap<ll, vl>
#define msi map<string, int>
#define vmii vector<mii>
#define iss istringstream
#define pf push_front
#define ef emplace_front
#define pb push_back
#define popb pop_back
#define eb emplace_back
#define ins insert
#define lb lower_bound
#define up upper_bound
#define sordefault(a) sort(all(a))
#define sorcmp(a, cmp) sort(all((a)), (cmp))
#define GET_SORT(_1, _2, NAME, ...) NAME
#define sor(...) GET_SORT(__VA_ARGS__, sorcmp, sordefault)(__VA_ARGS__)
#define rev(a) reverse(all(a))
#define mine(a) (*min_element((a).begin(), (a).end()))
#define maxe(a) (*max_element((a).begin(), (a).end()))
#define mini(a) (min_element((a).begin(), (a).end()) - (a).begin())
#define maxi(a) (max_element((a).begin(), (a).end()) - (a).begin())
#define lowb(a, x) (lower_bound((a).begin(), (a).end(), (x)) - (a).begin())
#define uppb(a, x) (upper_bound((a).begin(), (a).end(), (x)) - (a).begin())
#define repb(i, b) for (int i = 0; i < (b); ++i)
#define repab(i, a, b) for (int i = (a); i < (b); ++i)
#define repabd(i, a, b, d) for (int i = (a); i < (b); i += (d))
#define GET_REP(_1, _2, _3, _4, NAME, ...) NAME
#define re(...) GET_REP(__VA_ARGS__, repabd, repab, repb)(__VA_ARGS__)
#define repra(i, a) for (int i = (a)-1; i >= 0; i--)
#define reprab(i, a, b) for (int i = (a)-1; i >= (b); i--)
#define reprabd(i, a, b, d) for (int i = (a)-1; i >= (b); i -= (d))
#define GET_REPR(_1, _2, _3, _4, NAME, ...) NAME
#define rer(...) GET_REPR(__VA_ARGS__, reprabd, reprab, repra)(__VA_ARGS__)
#define eachone(a, x) for (auto &a : x)
#define eachtwo(k, v, x) for (auto &[k, v] : x)
#define eachthree(a, b, c, x) for (auto &[a, b, c] : x)
#define GET_EACH(_1, _2, _3, _4, NAME, ...) NAME
#define ea(...) GET_EACH(__VA_ARGS__, eachthree, eachtwo, eachone)(__VA_ARGS__)
#define rvi(v, n) vi v = vi((n)); re(i, n) cin >> v[i];
#define rvi2(v, n, m) vvi v = vvi((n), vi((m))); re(i, n) re(j, m) cin >> v[i][j];
#define rvc(v, n) vc v = vc((n)); re(i, n) cin >> v[i];
#define rvc2(v, n, m) vvc v = vvc((n), vc((m))); re(i, n) re(j, m) cin >> v[i][j];
#define rvl(v, n) vl v = vl((n)); re(i, n) cin >> v[i];
#define rvl2(v, n, m) vvl v = vvl((n), vl((m))); re(i, n) re(j, m) cin >> v[i][j];
#define rvs(v, n) vs v = vs((n)); re(i, n) cin >> v[i];
#define fail() fflush(stdout); cout.flush(); assert(false);
#define YES cout << "YES\n"
#define YESR { cout << "YES\n"; return; }
#define Yes cout << "Yes\n"
#define Yesr { cout << "Yes\n"; return; }
#define yes cout << "yes\n"
#define yesr { cout << "yes\n"; return; }
#define NO cout << "NO\n"
#define NOR { cout << "NO\n"; return; }
#define No cout << "No\n"
#define Nor { cout << "No\n"; return; }
#define no cout << "no\n";
#define nor { cout << "no\n"; return; }
#define n0 cout << "0\n"
#define n0r { cout << "0\n"; return; }
#define n1 cout << "-1\n"
#define n1r { cout << "-1\n"; return; }
template <typename A> using pqmax = priority_queue<A>;
template <typename A> using pqmin = priority_queue<A, vector<A>, greater<A>>;

const int MOD = (int) 1e9 + 7;
// const int MOD = 998244353;
const ld PI = acos((ld) -1);
const int e1 = (int)1e1, e2 = (int)1e2, e3 = (int)1e3, e4 = (int)1e4, e5 = (int)1e5, e6 = (int)1e6, e7 = (int)1e7, e8 = (int)1e8, e9 = (int) 1e9;
const ll e10 = (ll)1e10, e11 = (ll)1e11, e12 = (ll)1e12, e13 = (ll)1e13, e14 = (ll)1e14, e15 = (ll)1e15, e16 = (ll) 1e16, e17 = (ll)1e17, e18 = (ll) 1e18;

#ifdef LOCAL
#define dbg(...) logger(#__VA_ARGS__, __VA_ARGS__)
template <typename... Args> void logger(string vars, Args &&...values) {
  cout << "| " << vars << " = ";
  string delim = "";
  (..., (cout << delim << values, delim = ", "));
  cout << endl;
}
#else
#define dbg(...) 0
#endif

template <typename A>
vector<tuple<int, A>> enumerate(vector<A> v) {
  vector<tuple<int, A>> res(sz(v));
  re(i, sz(v)) res[i] = make_tuple(i, v[i]);
  return res;
}

struct mi {
  ll v;
  explicit operator ll() const { return v; }
  mi() { v = 0; }
  mi(ll _v) {
    v = (-MOD < _v && _v < MOD) ? _v : _v % MOD;
    if (v < 0) v += MOD;
  }
  friend bool operator==(const mi& a, const mi& b) { return a.v == b.v; }
  friend bool operator!=(const mi& a, const mi& b) { return !(a == b); }
  friend bool operator<(const mi& a, const mi& b) { return a.v < b.v; }
  mi& operator+=(const mi& m) { if ((v += m.v) >= MOD) v -= MOD; return *this; }
  mi& operator-=(const mi& m) { if ((v -= m.v) < 0) v += MOD; return *this; }
  mi& operator*=(const mi& m) { v = v * m.v % MOD; return *this; }
  mi& operator/=(const mi& m) { return (*this) *= inv(m); }
  friend mi pow(mi a, ll p) {
    mi ans = 1; assert(p >= 0);
    for (; p; p /= 2, a *= a) if (p&1) ans *= a;
    return ans;}
  friend mi inv(const mi& a) {
    assert(a.v != 0);
    return pow(a, MOD - 2);}
  mi operator-() const { return mi(-v); }
  mi &operator++() { return *this += 1; }
  mi &operator--() { return *this -= 1; }
  mi operator++(int) {
    mi temp = *this += 1;
    return temp;}
  mi operator--(int) {
    mi temp = *this -= 1;
    return temp;}
  friend mi operator+(mi a, const mi &b) { return a += b; }
  friend mi operator-(mi a, const mi &b) { return a -= b; }
  friend mi operator*(mi a, const mi &b) { return a *= b; }
  friend mi operator/(mi a, const mi &b) { return a /= b; }
  friend ostream &operator<<(ostream &os, const mi &m) {
    os << m.v;
    return os;
  }
  friend istream &operator>>(istream &is, mi &m) {
    ll x;
    is >> x;
    m.v = x;
    return is;
  }
};

mi operator"" _mi(ull x) { return x; }
string to_string(mi x) { return to_string(x.v); }

typedef vector<mi> vmi;
typedef vector<vector<mi>> vvmi;
typedef vector<vector<vector<mi>>> vvvmi;
typedef tuple<mi, mi> tmimi;
typedef tuple<mi, mi, mi> tmimimi;
typedef vector<tmimi> vmimi;
typedef vector<tmimimi> vmimimi;
typedef queue<mi> qmi;

template <typename T, typename = void> struct is_container {
  static constexpr bool value = false;
};
template <typename T>
struct is_container<T, void_t<decltype(T{}.begin()), decltype(T{}.end())>> {
  static constexpr bool value = true;
};
template <typename T, typename = void> struct is_tuple_like {
  static constexpr bool value = false;
};
template <typename T>
struct is_tuple_like<T, void_t<typename tuple_size<T>::type>> {
  static constexpr bool value = true;
};
template <typename T, typename = void> struct is_hashable {
  static constexpr bool value = false;
};
template <typename T> struct is_hashable<T, void_t<decltype(hash<T>{}(T{}))>> {
  static constexpr bool value = true;
};

vector<size_t> VALS;
struct custom_hash {
  static size_t getfixed(int i) {
    while ((int) VALS.size() < i + 1)
      if (VALS.empty())
        VALS.push_back((size_t) chrono::steady_clock::now().time_since_epoch().count());
      else
        VALS.push_back(splitmix64(VALS.back()));
    return VALS[i];
  }
  static size_t splitmix64(size_t x) {
    x += (size_t)0x9e3779b97f4a7c15;
    x = (x ^ (x >> (size_t)30)) * (size_t)0xbf58476d1ce4e5b9;
    x = (x ^ (x >> (size_t)27)) * (size_t)0x94d049bb133111eb;
    return x ^ (x >> (size_t)31);
  }
  size_t operator()(size_t x) const { return splitmix64(x + getfixed(0)); }
  template <typename T>
  enable_if_t<is_tuple_like<T>::value, size_t> operator()(const T &t) const {
    int ic = 0;
    size_t rh = 0;
    apply([&](auto &&...args) {
      ((rh ^= splitmix64(hash<decay_t<decltype(args)>>{}(args) + getfixed(ic++))), ...);
    }, t);
    return rh;
  }
  template <typename T>
  enable_if_t<is_container<T>::value, size_t> operator()(const T &t) const {
    int ic = 0; size_t rh = getfixed(0);
    for (auto& x : t) rh &= splitmix64(x + getfixed(ic++));
    return rh;
  }
};

template <typename A, typename B> using umap = unordered_map<A, B, custom_hash>;
template <typename A> using uset = unordered_set<A, custom_hash>;

template <class F> struct y_combinator {
  F f;
  template <class... Args> decltype(auto) operator()(Args &&...args) const {
    return f(*this, forward<Args>(args)...);
  }
};

template <class F> y_combinator<decay_t<F>> lam(F &&f) {
  return {forward<F>(f)};
}

constexpr int pct(int x) { return __builtin_popcount(x); }
constexpr int pct(long long x) { return __builtin_popcountll(x); }
constexpr int bits(int x) { return x == 0 ? -1 : 31 - __builtin_clz(x); }
constexpr int bits(long long x) { return x == 0 ? -1 : 63 - __builtin_clzll(x); }
constexpr int p2(int x) { return 1 << x; }
constexpr int msk2(int x) { return x == 31 ? 0xFFFFFFFF : ((uint32_t)1 << (x + 1)) - 1; }
constexpr bool test(int x, int i) { return (x & (1 << i)) != 0; }
constexpr bool test(long long x, int i) { return (x & (1LL << i)) != 0; }
int bitsrange(int l, int u) { assert(l <= u); return msk2(u) & ~msk2(l - 1); }

template <class T, class U> T ipow(T a, U p) {
  T ans = 1; assert(p >= 0);
  for (; p; p /= 2, a *= a) if (p&1) ans *= a;
  return ans;
}
template <class T, class U> T ipow(T a, U p, T m) {
  T ans = 1; assert(p >= 0);
  for (; p; p /= 2, a = (a * a) % m) if (p&1) ans = (ans * a) % m;
  return ans;
}

template <class T, class U> T fstTrue(T lo, T hi, U f) {
  hi++;
  assert(lo <= hi); // assuming f is increasing
  while (lo < hi) { // find first index such that f is true
    T mid = lo + (hi - lo) / 2;
    f(mid) ? hi = mid : lo = mid + 1;
  }
  return lo;
}

template <class T, class U> T lstTrue(T lo, T hi, U f) {
  lo--;
  assert(lo <= hi); // assuming f is decreasing
  while (lo < hi) { // find first index such that f is true
    T mid = lo + (hi - lo + 1) / 2;
    f(mid) ? lo = mid : hi = mid - 1;
  }
  return lo;
}

// Requirements: f is unimodal. f has adjacent-distinct values except possibly at its mode
template <class T, class U>
T ternary_search(T lo, T hi, U f, bool find_min = false) {
  assert(lo <= hi);
  return fstTrue(lo, hi - 1, [&](T n) { return find_min ? f(n) <= f(n + 1) : f(n) >= f(n + 1); });
}

template <class T> T sum(vector<T> &v) {
  if (v.empty()) return 0LL;
  T sum = v[0];
  for (int i = 1; i < (int) v.size(); i++) {
    sum += v[i];
  }
  return sum;
}

template <class F> F posmod(F a, F b) { return ((a % b) + b) % b; }

// 2d points and lines: https://codeforces.com/blog/entry/48122
template <class F>
struct Point {
  F x, y;
  Point(): x(0), y(0) {}
  Point(const F& cx, const F& cy): x(cx), y(cy) {}
};

template <class F> F floordiv(F a, F d) { F res = a / d; if (res * d != a) res -= 1 & ((a>0)^(d>0)); return res; }
template <class F> F ceildiv(F a, F d) { F res = a / d; if (res * d != a) res += 1 & ((a<0)^(d>0)); return res; }
template <class F> F sq(F a) { return a * a; }

long long inv(long long a, long long b) {
  return 1 < a ? b - inv(b % a, a) * b / a : 1;
}

template <class T> bool ckmin(T &a, const T &b) { return b < a ? a = b, 1 : 0; }
template <class T> bool ckmax(T &a, const T &b) { return a < b ? a = b, 1 : 0; }

const int dx[4] = {1, 0, -1, 0}, dy[4] = {0, 1, 0, -1};
vtii gridadj(int i, int j) {
  vtii v;
  re(d, 4) v.pb({i + dx[d], j + dy[d]});
  return v;
}

vtii griddiag(int i, int j) {
  static const int dxdiag[4] = {1, 1, -1, -1}, dydiag[4] = {1, -1, -1, 1};
  vtii v;
  re(d, 4) v.pb({i + dxdiag[d], j + dydiag[d]});
  return v;
}

// The sequence function from haskell but only for vectors, aka the cartesian product.
// https://hackage.haskell.org/package/base-4.18.0.0/docs/Prelude.html#v:sequence
// example:
// > sequence [[1,2,3],[4,5,6]]
// [[1,4],[1,5],[1,6],[2,4],[2,5],[2,6],[3,4],[3,5],[3,6]]
template <class A>
vec<vec<A>> sequence(const vec<vec<A>>& a) {
  vec<vec<A>> res;
  vec<A> current;
  auto f = [&](auto&& self, int i) {
    if (i == sz(a)) {
      res.pb(current);
      return;
    }
    ea(x, a[i]) {
      current.pb(x);
      self(self, i + 1);
      current.popb();
    }
  };
  f(f, 0);
  return res;
}

double time() { return (double)chrono::steady_clock::now().time_since_epoch().count() / 1e9; }

#ifdef LOCAL
mt19937 rng(2222);
#else
mt19937 rng((uint32_t)chrono::steady_clock::now().time_since_epoch().count());
#endif
int rint(int a, int b) { // inclusive bounds
  return uniform_int_distribution<int>(a, b)(rng);
}
double rdouble(double a, double b) {
  return uniform_real_distribution<double>(a, b)(rng);
}
template <class T> void shuffle(vector<T> &v) {
  auto r0 = [](int x) { return rint(0, x-1); };
  random_shuffle(all(v), r0);
}

template <class T> void remdup(vector<T> &v, bool do_sort = true) { // sort and remove duplicates
  if (do_sort) sort(all(v));
  v.erase(unique(all(v)), end(v));
}

template <class F> void print(const Point<F>& point) {
  cout << "x: " << point.x << "\ty: " << point.y << '\n';
}

template <class F> void print(const F a[], int n) {
  for (int i = 0; i < n; i++) {
    if (i > 0) cout << ' ';
    cout << a[i];
  }
  cout << '\n';
}

template <class F> void print(const vector<F> &a) {
  for (int i = 0; i < (int) a.size(); i++) {
    if (i > 0) cout << ' ';
    cout << a[i];
  }
  cout << '\n';
}

template <class F> void print2(const vector<vector<F>> &a) {
  for (int i = 0; i < (int) a.size(); i++) {
    for (int j = 0; j < (int) a[i].size(); j++) {
      if (j > 0) cout << ' ';
      cout << a[i][j];
    }
    cout << '\n';
  }
}

template <class F> void fill2(vector<vector<F>> &a, F val) {
  for (int i = 0; i < (int) a.size(); i++) {
    for (int j = 0; j < (int) a[i].size(); j++) {
      a[i][j] = val;
    }
  }
}

template <class F> void fill3(vector<vector<vector<F>>> &a, F val) {
  for (int i = 0; i < (int) a.size(); i++) {
    for (int j = 0; j < (int) a[i].size(); j++) {
      for (int k = 0; k < (int) a[i][j].size(); k++) {
        a[i][j][k] = val;
      }
    }
  }
}

template <class F> void r90(vector<vector<F>> &a) { // clockwise
  int n = sz(a), m = sz(a[0]);
  vector<vector<F>> copy(m, vector<F>(n));
  re(i, n) {
    re(j, m) { copy[j][n - 1 - i] = a[i][j]; }
  }
  a = copy;
}

template <class F> void r180(vector<vector<F>> &a) {
  int n = sz(a), m = sz(a[0]);
  vector<vector<F>> copy(n, vector<F>(m));
  re(i, n) {
    re(j, m) { copy[n - 1 - i][m - 1 - j] = a[i][j]; }
  }
  a = copy;
}

template <class F> void r270(vector<vector<F>> &a) {
  int n = sz(a), m = sz(a[0]);
  vector<vector<F>> copy(m, vector<F>(n));
  re(i, n) {
    re(j, m) { copy[m - 1 - j][i] = a[i][j]; }
  }
  a = copy;
}

string reps(string s, int t) {
  string res = ""; res.reserve(sz(s) * t);
  re(i, t) res += s;
  return res;
}

string reps(char c, int t) { return string(t, c); }

ll binary_to_ll(string s) { return stoll(s, nullptr, 2); }
string ll_to_binary(ll x) {
  assert(x >= 0);
  return !x ? "0" : bitset<64>(x).to_string().substr(63 - bits(x), 1 + bits(x));
}

vector<string> split(string s, string delimiter) {
  size_t pos_start = 0, pos_end, delim_len = delimiter.length();
  string token;
  vector<string> res;
  while ((pos_end = s.find(delimiter, pos_start)) != string::npos) {
    token = s.substr(pos_start, pos_end - pos_start);
    pos_start = pos_end + delim_len;
    res.push_back(token);
  }
  res.push_back(s.substr(pos_start));
  return res;
}

vector<string> split(string s, char delimiter) { return split(s, string(1, delimiter)); }

string to_string(string s) { return s; }
string to_string(char c) { return string(1, c); }

template <typename... Ts> string to_string(const tuple<Ts...> t) {
  string out;
  out += '(';
  apply([&](auto &&...args) { ((out += to_string(args) + ", "), ...); }, t);
  out.pop_back();
  out.pop_back();
  out += ')';
  return out;
}

template <typename... Ts>
ostream &operator<<(ostream &output, const tuple<Ts...> t) {
  output << to_string(t);
  return output;
}

template <class T> string to_string(vector<T> v) {
  string out = "[";
  re(i, sz(v)) { if (i > 0) out += ", "; out += to_string(v[i]); }
  out += ']';
  return out;
}

template <class T> ostream &operator<<(ostream &output, const vector<T> v) {
  output << to_string(v);
  return output;
}

int gen_base(const int before, const int after) {
  int base = rint(before + 1, after);
  return base % 2 == 0 ? base - 1 : base;
}

struct phash {
  static const int mod = (int) 1e9 + 123;
  static vi pow1; // powers of base mod
  static vector<ull> pow2; // powers of base 2^64
  static int base;

  vi pref1;
  vector<ull> pref2;

  template<typename A> phash(const A &s): pref1(sz(s) + 1u, 0), pref2(sz(s) + 1u, 0) {
    assert(base < mod);
    const int n = sz(s); // required power
    while (sz(pow1) <= n) {
      pow1.pb((int) ((ll) pow1.back() * base % mod));
      pow2.pb(pow2.back() * base);
    }
    for (int i = 0; i < n; i++) {
      assert(base > s[i]);
      pref1[i + 1] = (int) ((pref1[i] + (ll) s[i] * pow1[i]) % mod);
      pref2[i + 1] = pref2[i] + s[i] * pow2[i];
    }
  }

  // Example of mxPow: for phash("apple")(2, 2) == phash("pl")(0, 2, 5). We set mxpow for the second one to equalize
  inline tuple<int, ull> operator()(const int pos, const int len, int mxPow = -1) const {
    if (mxPow == -1) mxPow = (int) pref1.size() - 1;
    int hash1 = pref1[pos + len] - pref1[pos];
    ull hash2 = pref2[pos + len] - pref2[pos];
    if (hash1 < 0) hash1 += mod;
    if (mxPow != 0) {
      hash1 = (int) ((ll) hash1 * pow1[mxPow - (pos + len - 1)] % mod);
      hash2 *= pow2[mxPow - (pos + len - 1)];
    }
    return {hash1, hash2};
  }
};

vi phash::pow1{1};
vector<ull> phash::pow2{1};
int phash::base = gen_base(e6, phash::mod);


const int MX = 1e6 + 30;
bool facinit = false;
mi facs[MX];
mi facInvs[MX];

mi ncr(ll a, ll b) {
  assert(facinit && a < MX && b < MX);
  assert(a >= 0 && b >= 0 && b <= a); // disable this assert whenever necessary
  if (b > a || a < 0 || b < 0)
    return 0;
  mi cur = facs[a];
  cur = cur * facInvs[b];
  cur = cur * facInvs[a - b];
  return cur;
}

void initfacs() {
  facs[0] = 1;
  re(i, 1, MX) facs[i] = i * facs[i - 1];
  facInvs[MX - 1] = inv(facs[MX - 1]);
  rer(i, MX - 1) facInvs[i] = facInvs[i + 1] * (i + 1);
  facinit = true;
}

void setIO() {
  ios::sync_with_stdio(false);
  cin.tie(0); cout << fixed << setprecision(15);
  #ifdef GDB
  #define STRINGIZE(x) #x
  #define STRINGIZE_VALUE_OF(x) STRINGIZE(x)
  string basename = STRINGIZE_VALUE_OF(FNAME);
  basename = basename.substr(1, sz(basename)-2);
  string file_in = basename + ".in";
  string file_out = basename + ".out";
  freopen(file_in.c_str(), "r", stdin);
  freopen(file_out.c_str(), "w", stdout);
  #endif
}

// supports range queries and range updates
template <class F>
struct bitrange {
  int n, l, r;
  vector<F> b1, b2;

  bitrange(int _l, int _r) : l(_l), r(_r) {
    n = r - l + 1;
    b1.assign(n + 1, 0);
    b2.assign(n + 1, 0);
  }

  bitrange(int _n) : n(_n + 1) {
    l = 0, r = n;
    b1.assign(n + 1, 0);
    b2.assign(n + 1, 0);
  }

  void _pup(vector<F> &b, int x, F v) {
    assert(1 <= x && x <= n);
    for (; x <= n; x += x & (-x))
      b[x] += v;
  }

  void rup(int _l, int _r, F v) {
    _l = _l - l + 1;
    _r = _r - l + 1;
    _pup(b1, _l, v);
    _pup(b2, _l, v * (_l - 1));
    if (_r + 1 <= n) {
      _pup(b1, _r + 1, -v);
      _pup(b2, _r + 1, -v * _r);
    }
  }

  F _pre(vector<F> &b, int x) {
    assert(1 <= x && x <= n);
    F ans = 0;
    for (; x > 0; x -= x & (-x))
      ans += b[x];
    return ans;
  }

  F pre(int _x) {
    int x = _x - l + 1;
    return _pre(b1, x) * x - _pre(b2, x);
  }

  F sum(int _l, int _r) { return pre(_r) - (_l == l ? 0 : pre(_l - 1)); }
};
const int N = 2e5 + 30;

void solve() {
  int n, q; cin >> n >> q;
  rvi(a, n);
  bitrange<int> bit(N);

  auto ins = [&](int x) {
    if (bit.sum(x, x) == 0) {
      bit.rup(x, x, 1);
      return;
    }
    int r = lstTrue(x, N-1, [&](int i) {
      return bit.sum(x, i) == i - x + 1;
    });
    bit.rup(x, r, -1);
    bit.rup(r+1, r+1, 1);
  };

  auto del = [&](int x) {
    if (bit.sum(x, x) == 1) {
      bit.rup(x, x, -1);
      return;
    }
    int r = fstTrue(x, N-1, [&](int i) {
      return bit.sum(x, i) > 0;
    });
    bit.rup(r, r, -1);
    bit.rup(x, r-1, 1);
  };

  ea(x, a) ins(x);

  re(_q, q) {
    int k, l; cin >> k >> l;
    k--;
    del(a[k]);
    a[k] = l;
    ins(a[k]);

    int ans = lstTrue(0, N-1, [&](int i) {
      return bit.sum(i, N-1) > 0;
    });
    cout << ans << '\n';
  }
}

int32_t main() {
  setIO();
  int t = 1;

  #ifdef LOCAL
  double time_start = time();
  #endif
  re(_t, t) solve();
  #ifdef LOCAL
  cout << "done. elapsed: " << setprecision(2) << time() - time_start << endl;
  #endif
  return 0;
}

#include <iostream>
#include <algorithm>
#include <queue>
#include <set>
#include <map>
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <stdlib.h>
#include <stack>
#include <bitset>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>

using namespace std;
using namespace __gnu_pbds;

template<class key, class value, class cmp = std::less<key>>
using ordered_map = tree<key, value, cmp, rb_tree_tag, tree_order_statistics_node_update>;
template<class key, class cmp = std::less<key>>
using ordered_set = tree<key, null_type, cmp, rb_tree_tag, tree_order_statistics_node_update>;

#define ll long long
#define f first
#define s second
#define mp make_pair
#define pb push_back
#define forn(i, n) for (int i = 0; i < n; i++)
#define from(i,j,n) for (int i = j; i < n; i++)
#define sz(v) (int)v.size()
#define all(v) v.begin(),v.end()
#define eb emplace_back

ll INF = 1e18;
ll MOD = 998244353;
vector<ll> fact(3e5 + 5, 1);
vector<ll> fenwick;

template <typename T> vector<T> sort_unique(vector<T> vec) { sort(vec.begin(), vec.end()), vec.erase(unique(vec.begin(), vec.end()), vec.end()); return vec; }
template <typename T> istream &operator>>(istream &is, vector<T> &vec) { for (auto &v : vec) is >> v; return is; }
template <typename T> ostream &operator<<(ostream &os, const vector<T> &vec) { os << '['; for (auto v : vec) os << v << ','; os << ']'; return os; }
template <typename T, size_t sz> ostream &operator<<(ostream &os, const array<T, sz> &arr) { os << '['; for (auto v : arr) os << v << ','; os << ']'; return os; }
#if __cplusplus >= 201703L
template <typename... T> istream &operator>>(istream &is, tuple<T...> &tpl) { std::apply([&is](auto &&... args) { ((is >> args), ...);}, tpl); return is; }
template <typename... T> ostream &operator<<(ostream &os, const tuple<T...> &tpl) { os << '('; std::apply([&os](auto &&... args) { ((os << args << ','), ...);}, tpl); return os << ')'; }
#endif
template <typename T> ostream &operator<<(ostream &os, const deque<T> &vec) { os << "deq["; for (auto v : vec) os << v << ','; os << ']'; return os; }
template <typename T> ostream &operator<<(ostream &os, const set<T> &vec) { os << '{'; for (auto v : vec) os << v << ','; os << '}'; return os; }
template <typename T, typename TH> ostream &operator<<(ostream &os, const unordered_set<T, TH> &vec) { os << '{'; for (auto v : vec) os << v << ','; os << '}'; return os; }
template <typename T> ostream &operator<<(ostream &os, const multiset<T> &vec) { os << '{'; for (auto v : vec) os << v << ','; os << '}'; return os; }
template <typename T> ostream &operator<<(ostream &os, const unordered_multiset<T> &vec) { os << '{'; for (auto v : vec) os << v << ','; os << '}'; return os; }
template <typename T1, typename T2> ostream &operator<<(ostream &os, const pair<T1, T2> &pa) { os << '(' << pa.first << ',' << pa.second << ')'; return os; }
template <typename TK, typename TV> ostream &operator<<(ostream &os, const map<TK, TV> &mp) { os << '{'; for (auto v : mp) os << v.first << "=>" << v.second << ','; os << '}'; return os; }
template <typename TK, typename TV, typename TH> ostream &operator<<(ostream &os, const unordered_map<TK, TV, TH> &mp) { os << '{'; for (auto v : mp) os << v.first << "=>" << v.second << ','; os << '}'; return os; }

ll gcd(ll a, ll b) {
  if (b == 0)
    return a;
  return gcd(b, a % b);
}

struct SNM {
  struct Operation {
    Operation() : empty(true) {}
    Operation(int u, int v, int szv) : u(u), v(v), szv(szv), empty(false) {}

    bool is_empty() {
      return empty;
    }

    int u, v, szv;
    bool empty = false;
  };

  SNM(int n) : n(n), comp_count(n) {
    parent.resize(n, -1);
    sz.resize(n, 1);
  }

  int get(int v) {
    if (parent[v] == -1)
      return v;
    return get(parent[v]);
  }

  void uni(int u, int v) {
    auto pu = get(u);
    auto pv = get(v);
    if (pu == pv) {
      ops.push({});
      return;
    }
    if (sz[pu] > sz[pv]) {
      swap(pu, pv);
    }
    ops.push({pu, pv, sz[pv]});
    parent[pu] = pv;
    sz[pv] += sz[pu];
    comp_count--;
  }

  void rollback() {
    auto op = ops.top();
    ops.pop();
    if (!op.is_empty()) {
      parent[op.u] = -1;
      sz[op.v] = op.szv;
      comp_count++;
    }
  }

  int components() {
    return comp_count;
  }

  void reset() {
    while (!ops.empty())
      rollback();
  }

  int n, comp_count;
  vector<int> parent, sz;
  stack<Operation> ops;
};

struct Query {
  Query(int type, int ts) : type(type), l(ts), r(ts + 1) {}
  Query(int type, int l, int r, int u, int v) : type(type), l(l), r(r), u(u), v(v) {}

  // 1 - [l, r], 0 - get
  int type, l=-1, r=-1, u=-1, v=-1;
};

vector<pair<int, int>> dsu_answer;

void dsu_offline(SNM& snm, const vector<Query>& queries, int l, int r) {
  int ops = 0;
  auto mid = (l + r) / 2;
  vector<Query> left, right;

  for (auto q : queries) {
    if (q.type == 1 && q.l <= l && r <= q.r) {
      snm.uni(q.u, q.v);
      ops++;
      continue;
    }
    if (!(q.l >= mid || q.r <= l)) {
      left.pb(q);
    }
    if (!(q.l >= r || q.r <= mid)) {
      right.pb(q);
    }
  }
  if (l + 1 == r) {
    for (auto q : queries) {
      if (q.type == 0) {
        dsu_answer.pb(mp(l, snm.components()));
      }
    }
  } else {
    dsu_offline(snm, left, l, mid);
    dsu_offline(snm, right, mid, r);
  }
  while (ops > 0) {
    snm.rollback();
    ops--;
  }
}

vector<int> z_function (string s) {
  int n = (int) s.length();
  vector<int> z (n);
  for (int i=1, l=0, r=0; i<n; ++i) {
    if (i <= r)
      z[i] = min (r-i+1, z[i-l]);
    while (i+z[i] < n && s[z[i]] == s[i+z[i]])
      ++z[i];
    if (i+z[i]-1 > r)
      l = i,  r = i+z[i]-1;
  }
  return z;
}

void update(int pos, ll val) { // val- what we add
  for (int i = pos; i < fenwick.size(); i = ((i - 1) | i) + 1) {
    fenwick[i] += val;
  }
}

ll sum(int pos) {
  ll result=0;
  for (int i = pos; i > 0; i = i & (i - 1))
    result += fenwick[i];
  return result;
}

ll range_sum(int l, int r) { // (l, r]
  return sum(r) - sum(l);
}

void fill_fact() {
  for (int i = 2; i < fact.size(); i++)
    fact[i] = (fact[i - 1] * i) % MOD;
}

ll pow(ll a, ll b) {
  if (b == 0) {
    return 1;
  } else if (b % 2 == 0) {
    ll x = pow(a, b / 2);
    return x * x;
  } else {
    return a * pow(a, b - 1);
  }
}

ll pow_mod(ll a, ll b) {
  if (b == 0) {
    return 1;
  } else if (b % 2 == 0) {
    ll x = pow_mod(a, b / 2) % MOD;
    return (x * x) % MOD;
  } else {
    return (a * pow_mod(a, b - 1)) % MOD;
  }
}

ll cnk(ll n, ll k) {
  ll n_fact = fact[n];
  ll k_fact = fact[k];
  ll nk_fact = fact[n - k];
  ll down = (k_fact * nk_fact) % MOD;
  ll up = pow_mod(down, MOD - 2);
  return (n_fact * up) % MOD;
}

namespace ST {
  constexpr struct MinAssignOp {
    /*
     *  妍扭快把忘扯我攸 志抑折我扼抖快扶我攸 技我扶我技批技忘 扶忘 抉找把快戒抗忘 扼 抉扭快把忘扯我快抄 扭把我扼志抉快扶我攸 扶忘
     *  抉找把快戒抗快
     */
    ll init() const {
      return 1e18;
    }

    ll operator()(ll a, ll b) const {
      return min(a, b);
    }

    void assign(vector<pair<ll, ll>>& tree, int v, int tl, int tr, ll val) {
      tree[v].f = min(tree[v].f, val);
      tree[v].s = min(tree[v].s, val);
    }

    void push(vector<pair<ll, ll>>& tree, int v, int tl, int tr) {
      const auto val = tree[v].s;
      const auto vl = 2 * v + 1;
      const auto vr = 2 * v + 2;
      assign(tree, vl, tl, tr, val);
      assign(tree, vr, tl, tr, val);
      tree[v].s = init();
    }
  } Min;

  constexpr struct MaxAssignOp {
    /*
     *  妍扭快把忘扯我攸 志抑折我扼抖快扶我攸 技我扶我技批技忘 扶忘 抉找把快戒抗忘 扼 抉扭快把忘扯我快抄 扭把我扼志抉快扶我攸 扶忘
     *  抉找把快戒抗快
     */
    ll init() const {
      return -1e18;
    }

    ll operator()(ll a, ll b) const {
      return max(a, b);
    }

    void assign(vector<pair<ll, ll>>& tree, int v, int tl, int tr, ll val) {
      tree[v].f = max(tree[v].f, val);
      tree[v].s = max(tree[v].s, val);
    }

    void push(vector<pair<ll, ll>>& tree, int v, int tl, int tr) {
      const auto val = tree[v].s;
      const auto vl = 2 * v + 1;
      const auto vr = 2 * v + 2;
      assign(tree, vl, tl, tr, val);
      assign(tree, vr, tl, tr, val);
      tree[v].s = init();
    }
  } Max;

  constexpr struct SumAddOp {
    /*
     *  妍扭快把忘扯我攸 扼批技技抑 扶忘 抉找把快戒抗快 扼 抉扭快把忘扯我快抄 扭把我忌忘志抖快扶我攸 扶忘 抉找把快戒抗快
     */
    ll init() const {
      return 0;
    }

    ll operator()(ll a, ll b) const {
      return a + b;
    }

    void assign(vector<pair<ll, ll>>& tree, int v, int tl, int tr, ll add) {
      tree[v].f += (tr - tl) * add;
      tree[v].s += add;
    }

    void push(vector<pair<ll, ll>>& tree, int v, int tl, int tr) {
      const auto add = tree[v].s;
      const auto vl = 2 * v + 1;
      const auto vr = 2 * v + 2;
      const auto mid = (tl + tr) / 2;

      tree[vl].f += add * (mid - tl);
      tree[vl].s += add;

      tree[vr].f += add * (tr - mid);
      tree[vr].s += add;
      tree[v].s = 0;
    }
  } Sum;
}

template<typename Operation>
struct SegmentTree {

  SegmentTree(int n, Operation op) : n(n), tree(4 * n, mp(op.init(), op.init())), op(op) {}

  void update(int l, int r, ll add) {
    update_impl(0, 0, n, l, r, add);
  }

  void update(int l, ll add) {
    update_impl(0, 0, n, l, l + 1, add);
  }

  ll get(int l, int r) {
    return get_impl(0, 0, n, l, r);
  }

  ll get(int l) {
    return get_impl(0, 0, n, l, l + 1);
  }

private:
  void update_impl(int v, int tl, int tr, int l, int r, ll add) {
    if (tl >= l && tr <= r) {
      op.assign(tree, v, tl, tr, add);
      return;
    }
    if (tl >= r || tr <= l) {
      return;
    }
    const auto mid = (tl + tr) / 2;
    op.push(tree, v, tl, tr);
    update_impl(2 * v + 1, tl, mid, l, r, add);
    update_impl(2 * v + 2, mid, tr, l, r, add);
    tree[v].f = op(tree[2 * v + 1].f, tree[2 * v + 2].f);
  }

  ll get_impl(int v, int tl, int tr, int l, int r) {
    if (tl >= l && tr <= r) {
      return tree[v].f;
    }
    if (tl >= r || tr <= l) {
      return op.init();
    }
    op.push(tree, v, tl, tr);
    const auto mid = (tl + tr) / 2;
    return op(get_impl(2 * v + 1, tl, mid, l, r), get_impl(2 * v + 2, mid, tr, l, r));
  }

  vector<pair<ll, ll>> tree;
  int n;
  Operation op;
};

int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);
  int t;
  cin >> t;
  while (t--) {
    int n;
    cin >> n;
    vector<int> vec(n);
    forn(i, n) {
      cin >> vec[i];
    }
    sort(all(vec));
    auto mini = vec[0] * 2 - 1;
    ll answer = 0;
    forn(i, n) {
      answer += vec[i] / mini - int(vec[i] % mini == 0);
    }
    cout << answer << endl;
  }
  return 0;
}

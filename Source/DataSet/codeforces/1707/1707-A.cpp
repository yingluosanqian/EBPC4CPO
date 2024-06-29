

#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <tuple>
#include <stack>
#include <queue>
#include <deque>
#include <algorithm>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <bitset>
#include <cmath>
#include <functional>
#include <cassert>
#include <climits>
#include <iomanip>
#include <numeric>
#include <memory>
#include <random>
#include <thread>
#include <chrono>
#define allof(obj) (obj).begin(), (obj).end()
#define range(i, l, r) for(int i=l;i<r;i++)
#define unique_elem(obj) obj.erase(std::unique(allof(obj)), obj.end())
#define bit_subset(i, S) for(int i=S, zero_cnt=0;(zero_cnt+=i==S)<2;i=(i-1)&S)
#define bit_kpop(i, n, k) for(int i=(1<<k)-1,x_bit,y_bit;i<(1<<n);x_bit=(i&-i),y_bit=i+x_bit,i=(!i?(1<<n):((i&~y_bit)/x_bit>>1)|y_bit))
#define bit_kth(i, k) ((i >> k)&1)
#define bit_highest(i) (i?63-__builtin_clzll(i):-1)
#define bit_lowest(i) (i?__builtin_ctzll(i):-1)
#define sleepms(t) std::this_thread::sleep_for(std::chrono::milliseconds(t))
using ll = long long;
using ld = long double;
using ul = uint64_t;
using pi = std::pair<int, int>;
using pl = std::pair<ll, ll>;
using namespace std;

template<typename F, typename S>
std::ostream &operator<<(std::ostream &dest, const std::pair<F, S> &p){
  dest << p.first << ' ' << p.second;
  return dest;
}
template<typename T>
std::ostream &operator<<(std::ostream &dest, const std::vector<std::vector<T>> &v){
  int sz = v.size();
  if(sz==0) return dest;
  for(int i=0;i<sz;i++){
    int m = v[i].size();
    for(int j=0;j<m;j++) dest << v[i][j] << (i!=sz-1&&j==m-1?'\n':' ');
  }
  return dest;
}
template<typename T>
std::ostream &operator<<(std::ostream &dest, const std::vector<T> &v){
  int sz = v.size();
  if(sz==0) return dest;
  for(int i=0;i<sz-1;i++) dest << v[i] << ' ';
  dest << v[sz-1];
  return dest;
}
template<typename T, size_t sz>
std::ostream &operator<<(std::ostream &dest, const std::array<T, sz> &v){
  if(sz==0) return dest;
  for(int i=0;i<sz-1;i++) dest << v[i] << ' ';
  dest << v[sz-1];
  return dest;
}
template<typename T>
std::ostream &operator<<(std::ostream &dest, const std::set<T> &v){
  for(auto itr=v.begin();itr!=v.end();){
    dest << *itr;
    itr++;
    if(itr!=v.end()) dest << ' ';
  }
  return dest;
}
template<typename T, typename E>
std::ostream &operator<<(std::ostream &dest, const std::map<T, E> &v){
  for(auto itr=v.begin();itr!=v.end();){
    dest << '(' << itr->first << ", " << itr->second << ')';
    itr++;
    if(itr!=v.end()) dest << '\n';
  }
  return dest;
}
std::ostream &operator<<(std::ostream &dest, __int128_t value) {
  std::ostream::sentry s(dest);
  if (s) {
    __uint128_t tmp = value < 0 ? -value : value;
    char buffer[128];
    char *d = std::end(buffer);
    do {
      --d;
      *d = "0123456789"[tmp % 10];
      tmp /= 10;
    } while (tmp != 0);
    if (value < 0) {
      --d;
      *d = '-';
    }
    int len = std::end(buffer) - d;
    if (dest.rdbuf()->sputn(d, len) != len) {
      dest.setstate(std::ios_base::badbit);
    }
  }
  return dest;
}
template<typename T>
vector<T> make_vec(size_t sz, T val){return std::vector<T>(sz, val);}
template<typename T, typename... Tail>
auto make_vec(size_t sz, Tail ...tail){
  return std::vector<decltype(make_vec<T>(tail...))>(sz, make_vec<T>(tail...));
}
template<typename T>
vector<T> read_vec(size_t sz){
  std::vector<T> v(sz);
  for(int i=0;i<(int)sz;i++) std::cin >> v[i];
  return v;
}
template<typename T, typename... Tail>
auto read_vec(size_t sz, Tail ...tail){
  auto v = std::vector<decltype(read_vec<T>(tail...))>(sz);
  for(int i=0;i<(int)sz;i++) v[i] = read_vec<T>(tail...);
  return v;
}
long long max(long long a, int b){return std::max(a, (long long)b);}
long long max(int a, long long b){return std::max((long long)a, b);}
long long min(long long a, int b){return std::min(a, (long long)b);}
long long min(int a, long long b){return std::min((long long)a, b);}
long long modulo(long long a, long long m){a %= m; return a < 0 ? a + m : a;}

void io_init(){
  std::cin.tie(nullptr);
  std::ios::sync_with_stdio(false);
}






#include <limits>

struct point_min_range_min{
  template<typename T>
  static T id(){
    return std::numeric_limits<T>::max();
  }
  template<typename T>
  static T update(T a, T b){
    return std::min(a, b);
  }
  template<typename T>
  static T merge(T a, T b){
    return std::min(a, b);
  }
};
struct point_min_range_second_min{
  template<typename T>
  static T id(){
    return {std::numeric_limits<long long>::max(), std::numeric_limits<long long>::max()};
  }
  template<typename T>
  static T update(T a, T b){
    if(a.first <= b.first) return {a.first, std::min(a.second, b.first)};
    else return {b.first, std::min(a.first, b.second)};
  }
  template<typename T>
  static T merge(T a, T b){
    if(a.first <= b.first) return {a.first, std::min(a.second, b.first)};
    else return {b.first, std::min(a.first, b.second)};
  }
};
struct point_max_range_max{
  template<typename T>
  static T id(){
    return std::numeric_limits<T>::min();
  }
  template<typename T>
  static T update(T a, T b){
    return std::max(a, b);
  }
  template<typename T>
  static T merge(T a, T b){
    return std::max(a, b);
  }
  template<typename T>
  static T flip(T a){
    return a;
  }
};
struct point_max_range_second_max{
  template<typename T>
  static T id(){
    return {std::numeric_limits<long long>::min(), std::numeric_limits<long long>::min()};
  }
  template<typename T>
  static T update(T a, T b){
    if(a.first >= b.first) return {a.first, std::min(a.second, b.first)};
    else return {b.first, std::min(a.first, b.second)};
  }
  template<typename T>
  static T merge(T a, T b){
    if(a.first >= b.first) return {a.first, std::min(a.second, b.first)};
    else return {b.first, std::min(a.first, b.second)};
  }
};
struct point_mul_range_mul{
  template<typename T>
  static T id(){
    return 1;
  }
  template<typename T>
  static T update(T a, T b){
    return a * b;
  }
  template<typename T>
  static T merge(T a, T b){
    return a * b;
  }
};
struct point_add_range_min{
  template<typename T>
  static T id(){
    return std::numeric_limits<T>::max();
  }
  template<typename T>
  static T update(T a, T b){
    if(a == id<T>()) return b;
    else if(b == id<T>()) return a;
    return a + b;
  }
  template<typename T>
  static T merge(T a, T b){
    return std::min(a, b);
  }
};

struct point_add_range_max{
  template<typename T>
  static T id(){
    return std::numeric_limits<T>::min();
  }
  template<typename T>
  static T update(T a, T b){
    if(a == id<T>()) return b;
    else if(b == id<T>()) return a;
    return a + b;
  }
  template<typename T>
  static T merge(T a, T b){
    return std::max(a, b);
  }
};

struct point_add_range_sum{
  template<typename T>
  static T id(){
    return 0;
  }
  template<typename T>
  static T update(T a, T b){
    return a + b;
  }
  template<typename T>
  static T merge(T a, T b){
    return a + b;
  }
  template<typename T>
  static T flip(T a){
    return a;
  }
};
struct point_set_range_composite{
  static constexpr int mod = 998244353;
  template<typename T>
  static T id(){
    return {1, 0};
  }
  template<typename T>
  static T update(T a, T b){
    return b;
  }
  template<typename T>
  static T merge(T a, T b){
    int xy = ((long long)a.first * b.first) % mod;
    int ab = ((long long)a.second * b.first + b.second) % mod;
    return {xy, ab};
  }
};
struct excess_value{
  int rank, sum, minl, maxr;
  excess_value(){}
  excess_value(bool f): rank(f), sum(f ? 1 : -1), minl(sum), maxr(sum){}
  excess_value(int a, int b, int c, int d): rank(a), sum(b), minl(c), maxr(d){}
};
struct point_set_range_excess_value{
  const static int inf = std::numeric_limits<int>::max() / 2;
  template<typename T>
  static T id(){
    // '('を1, ')'を　-1としたときの{1の数, 和, 左を固定したsumのmin, 右を固定したsumのmin}
    return {inf, 0, 0, 0};
  }
  template<typename T>
  static T update(T a, T b){
    if(a.rank == inf) return b;
    if(b.rank == inf) return a;
    return {a.rank + b.rank, a.sum + b.sum, std::min(a.min, a.sum + b.minl), std::max(b.maxr, a.maxr + b.sum)};
  }
  template<typename T>
  static T merge(T a, T b){
    if(a.rank == inf) return b;
    if(b.rank == inf) return a;
    return {a.rank + b.rank, a.sum + b.sum, std::min(a.minl, a.sum + b.minl), std::max(b.maxr, a.maxr + b.sum)};
  }
};
struct point_set_range_composite_flip{
  static constexpr int mod = 998244353;
  template<typename T>
  static T id(){
    return {1, 0, 0};
  }
  template<typename T>
  static T update(T a, T b){
    return b;
  }
  template<typename T>
  static T flip(T a){
    return {a[0], a[2], a[1]};
  }
  template<typename T>
  static T merge(T a, T b){
    int xy = ((long long)a[0] * b[0]) % mod;
    int ab = ((long long)a[1] * b[0] + b[1]) % mod;
    int ba = ((long long)b[2] * a[0] + a[2]) % mod;
    return {xy, ab, ba};
  }
};
// merge(a, b) = max(0, a + b)の場合, {x, 0}で初期化
struct hawker{
  static constexpr long long minf = std::numeric_limits<long long>::min() / 2;
  template<typename T>
  static T id(){
    return {0, minf};
  }
  template<typename T>
  static T update(T a, T b){
    return {a.first + b.first, std::max(a.second + b.first, b.second)};
  }
  template<typename T>
  static T merge(T a, T b){
    return {a.first + b.first, std::max(a.second + b.first, b.second)};
  }
};


struct point_add_range_gcd{
  template<typename Val>
  static Val __binary_gcd(Val a, Val b){
    if(!a || !b) return !a ? b : a;
    if(a < 0) a *= -1;
    if(b < 0) b *= -1;
    int shift = __builtin_ctzll(a | b); // [1] gcd(2a', 2b') = 2 * gcd(a', b')
    a >>= __builtin_ctzll(a);
    do{
      // if b is odd
      // gcd(2a', b) = gcd(a', b), if a = 2a'(a is even)
      // gcd(a, b) = gcd(|a - b|, min(a, b)), if a is odd
      b >>= __builtin_ctzll(b); // make b odd
      if(a > b) std::swap(a, b);
      b -= a;
    }while(b); // gcd(a, 0) = a
    return a << shift; // [1]
  }
  template<typename Val>
  static Val id(){
    return 0;
  }
  template<typename Val>
  static Val update(Val a, Val b){
    return a + b;
  }
  template<typename Val>
  static Val merge(Val a, Val b){
    return __binary_gcd(a, b);
  }
};
// 区間set, これまでにsetした物の中ならどれかを取得
struct range_set_get_any{
  template<typename Val>
  static Val id1(){
    return nullptr;
  }
  template<typename Lazy>
  static Lazy id2(){
    return nullptr;
  }
  template<typename Lazy>
  static Lazy propagate(Lazy l, Lazy x){
    return (x == nullptr ? l : x);
  }
  template<typename Val, typename Lazy>
  static Val apply(Val v, Lazy x, int l, int r){
    return (x == nullptr ? v : x);
  }
};

struct range_add_range_sum{
  template<typename T>
  static T id1(){
    return T(0);
  }
  template<typename E>
  static E id2(){
    return E(0);
  }
  template<typename T>
  static T merge(T a, T b){
    return a + b;
  }
  template<typename T, typename E>
  static T apply(T a, E b, int l, int r){
    return a + b * (r - l);
  }
  template<typename E>
  static E propagate(E a, E b){
    return a + b;
  }
  template<typename T>
  static T flip(T a){
    return a;
  }
};

struct range_max_range_max{
  template<typename T>
  static T id1(){
    return std::numeric_limits<T>::min();
  }
  template<typename E>
  static E id2(){
    return std::numeric_limits<E>::min();
  }
  template<typename T>
  static T merge(T a, T b){
    return std::max(a, b);
  }
  template<typename T, typename E>
  static T apply(T a, E b, int l, int r){
    return std::max(a, b);
  }
  template<typename E>
  static E propagate(E a, E b){
    return std::max(a, b);
  }
};

struct range_set_range_min{
  template<typename T>
  static T id1(){
    return std::numeric_limits<T>::max();
  }
  template<typename E>
  static E id2(){
    return std::numeric_limits<E>::max();
  }
  template<typename T>
  static T merge(T a, T b){
    return std::min(a, b);
  }
  template<typename T, typename E>
  static T apply(T a, E b, int l, int r){
    return b;
  }
  template<typename E>
  static E propagate(E a, E b){
    return b;
  }
};
struct range_set_range_sum{
  template<typename T>
  static T id1(){
    return 0;
  }
  template<typename E>
  static E id2(){
    return std::numeric_limits<E>::max();
  }
  template<typename T>
  static T merge(T a, T b){
    return a + b;
  }
  template<typename T, typename E>
  static T apply(T a, E b, int l, int r){
    return b * (r - l);
  }
  template<typename E>
  static E propagate(E a, E b){
    return b;
  }
};
struct range_add_range_max{
  template<typename T>
  static T id1(){
    return std::numeric_limits<T>::min();
  }
  template<typename E>
  static E id2(){
    return 0;
  }
  template<typename T>
  static T merge(T a, T b){
    return std::max(a, b);
  }
  template<typename T, typename E>
  static T apply(T a, E b, int l, int r){
    if(a == id1<T>()) return b;
    return a + b;
  }
  template<typename E>
  static E propagate(E a, E b){
    return a + b;
  }
};

struct range_add_range_min{
  template<typename T>
  static T id1(){
    return std::numeric_limits<T>::max() / 2;
  }
  template<typename E>
  static E id2(){
    return 0;
  }
  template<typename T>
  static T merge(T a, T b){
    return std::min(a, b);
  }
  template<typename T, typename E>
  static T apply(T a, E b, int l, int r){
    if(a == id1<T>()) return b;
    return a + b;
  }
  template<typename E>
  static E propagate(E a, E b){
    return a + b;
  }
};

struct range_add_range_argmin{
  template<typename T>
  static T id1(){
    return {std::numeric_limits<long long>::max(), -1} ;
  }
  template<typename E>
  static E id2(){
    return 0;
  }
  template<typename T>
  static T merge(T a, T b){
    return std::min(a, b);
  }
  template<typename T, typename E>
  static T apply(T a, E b, int l, int r){
    if(a == id1<T>()) return a;
    return {a.first + b, a.second};
  }
  template<typename E>
  static E propagate(E a, E b){
    return a + b;
  }
};

/*
#include <array>
struct range_affine_range_sum{
  const static long long MOD = 998244353;
  template<typename T>
  static T id1(){
    return 0;
  }
  template<typename E>
  static E id2(){
    return E{1, 0};
  }
  template<typename T>
  static T merge(T a, T b){
    return (a + b) % MOD;
  }
  template<typename T, typename E>
  static T apply(T a, E b, int l, int r){
    return (a * b[0] + b[1] * (r - l)) % MOD;
  }
  template<typename E>
  static E propagate(E a, E b){
    return E{(a[0] * b[0]) % MOD, (a[1] * b[0] + b[1]) % MOD};
  }
};
*/
struct range_affine_range_sum{
  const static int MOD = 998244353;
  template<typename T>
  static T id1(){
    return 0;
  }
  template<typename E>
  static E id2(){
    return E{1, 0};
  }
  template<typename T>
  static T merge(T a, T b){
    return (a + b) % MOD;
  }
  template<typename T, typename E>
  static T apply(T a, E b, int l, int r){
    return ((long long)a * b.first + (long long)b.second * (r - l)) % MOD;
  }
  template<typename E>
  static E propagate(E a, E b){
    return E{((long long)a.first * b.first) % MOD, ((long long)a.second * b.first + b.second) % MOD};
  }
};

struct range_add_range_min_count{
  static constexpr int INF = std::numeric_limits<int>::max();
  template<typename T>
  static T id1(){
    return {INF, 0};
  }
  template<typename E>
  static E id2(){
    return 0;
  }
  template<typename T>
  static T merge(T a, T b){
    if(a.first != b.first) return a.first < b.first ? a : b;
    return {a.first, a.second + b.second};
  }
  template<typename T, typename E>
  static T apply(T a, E b, int l, int r){
    if(a.first == INF) return {b, r - l};
    return {a.first + b, a.second};
  }
  template<typename E>
  static E propagate(E a, E b){
    return a + b;
  }
};

struct range_composite_lct{
  static constexpr int MOD = 998244353;
  template<typename T>
  // 1x + 0, 1x + 0
  static T id1(){
    return std::array<int, 3>{1, 0, 0};
  }
  // no use
  template<typename E>
  static E id2(){
    return false;
  }
  // b(a(x)), a(b(x))
  template<typename T>
  static T merge(T a, T b){
    int ba1 = ((long long)b[0] * a[0]) % MOD;
    int ba2 = ((long long)b[0] * a[1] + b[1]) % MOD;
    int ab2 = ((long long)a[0] * b[2] + a[2]) % MOD;
    return std::array<int, 3>{ba1, ba2, ab2};
  }
  // no use
  template<typename T, typename E>
  static T apply(T a, E b, int l, int r){
    return a;
  }
  // no use
  template<typename E>
  static E propagate(E a, E b){
    return false;
  }
  //
  template<typename T>
  static T flip(T a){
    return std::array<int, 3>{a[0], a[2], a[1]};
  }
};


template<typename monoid, typename Val>
struct dynamic_sequence{
private:
  struct node{
    int h, sz;
    Val val, sum;
    node *l, *r;
    node(Val _val = monoid::template id<Val>()): h(1), sz(1), val(_val), sum(val), l(nullptr), r(nullptr){}
    int balanace_factor(){
      return (l ? l->h : 0) - (r ? r->h : 0);
    }
  };
  node *root, *tmp_node;
  static int size(node *v){return v ? v->sz : 0;}
  static void update(node *v){
    v->h = std::max(v->l ? v->l->h : 0,  v->r ? v->r->h : 0) + 1;
    v->sz = 1;
    v->sum = v->val;
    if(v->l){
      v->sz += v->l->sz;
      v->sum = monoid::template merge<Val>(v->l->sum, v->sum);
    }
    if(v->r){
      v->sz += v->r->sz;
      v->sum = monoid::template merge<Val>(v->sum, v->r->sum);
    }
  }
  
  static node *rotate_right(node *v){
    node *l = v->l;
    v->l = l->r;
    l->r = v;
    update(v);
    update(l);
    return l;
  }
  static node *rotate_left(node *v){
    node *r = v->r;
    v->r = r->l;
    r->l = v;
    update(v);
    update(r);
    return r;
  }
  node *build(const std::vector<node*> &nodes, int l, int r){
    int m = (l + r) >> 1;
    node *v = nodes[m];
    if(m > l) v->l = build(nodes, l, m);
    if(r > m + 1) v->r = build(nodes, m + 1, r);
    update(v);
    return v;
  }
  node *balance(node *v){
    int bf = v->balanace_factor();
    assert(-2 <= bf && bf <= 2);
    if(bf == 2){
      if(v->l->balanace_factor() == -1){
        v->l = rotate_left(v->l);
        update(v);
      }
      return rotate_right(v);
    }else if(bf == -2){
      if(v->r->balanace_factor() == 1){
        v->r = rotate_right(v->r);
        update(v);
      }
      return rotate_left(v);
    }
    return v;
  }
  void update_inner(node *v, int k, Val x){
    int szl = v->l ? v->l->sz : 0;
    if(k < szl) update_inner(v->l, k, x);
    else if(k > szl) update_inner(v->r, k - szl - 1, x);
    else v->val = monoid::template update<Val>(v->val, x);
    update(v);
  }
  Val query_inner(node *v, int l, int r){
    if(!v) return monoid::template id<Val>();
    if(l == 0 && r == v->sz) return v->sum;
    int szl = size(v->l), szv = szl + 1;
    Val left_q = monoid::template id<Val>(), right_q = left_q;
    if(l < szl){
      if(r <= szl) return query_inner(v->l, l, r);
      left_q = query_inner(v->l, l, szl);
      l = szl;
    }
    if(szv < r){
      if(szv <= l) return query_inner(v->r, l - szv, r - szv);
      right_q = query_inner(v->r, 0, r - szv);
      r = szv;
    }
    Val res = (l == r ? monoid::template id<Val>() : v->val);
    res = monoid::template merge<Val>(left_q, res);
    res = monoid::template merge<Val>(res, right_q);
    return res;
  }
  node *cut_left_most(node *v){
    if(v->l){
      v->l = cut_left_most(v->l);
      update(v);
      return balance(v);
    }
    tmp_node = v;
    return v->r;
  }
  node *cut_right_most(node *v){
    if(v->r){
      v->r = cut_right_most(v->r);
      update(v);
      return balance(v);
    }
    tmp_node = v;
    return v->l;
  }
  void set_inner(node *v, int k, Val x){
    int szl = v->l ? v->l->sz : 0;
    if(k < szl) set_inner(v->l, k, x);
    else if(k > szl) set_inner(v->r, k - szl - 1, x);
    else v->val = x;
    update(v);
  }
  Val get_inner(node *v, int k){
    int szl = v->l ? v->l->sz : 0;
    if(k < szl) return get_inner(v->l, k);
    else if(k > szl) return get_inner(v->r, k - szl - 1);
    else return v->val;
  }
  node *insert_inner(node *v, int k, Val x){
    assert(size() >= k);
    if(!v) return new node(x);
    int szl = v->l ? v->l->sz : 0;
    if(k <= szl) v->l = insert_inner(v->l, k, x);
    else if(k > szl) v->r = insert_inner(v->r, k - szl - 1, x);
    update(v);
    return balance(v);
  }
  node *erase_inner(node *v, int k){
    assert(0 <= k && k < size());
    int szl = v->l ? v->l->sz : 0;
    if(k < szl) v->l = erase_inner(v->l, k);
    else if(k > szl) v->r = erase_inner(v->r, k - szl - 1);
    else{
      if(!v->r) return v->l;
      node *u = cut_left_most(v->r);
      tmp_node->l = v->l;
      tmp_node->r = u;
      v = tmp_node;
    }
    update(v);
    return balance(v);
  }
  // https://atcoder.jp/contests/practice2/submissions/42114957
  template<typename F>
  int bisect_from_left(node *v, F &f, int l, Val &ok){
    if(!v || v->sz <= l) return -1;
    int szl = v->l ? v->l->sz : 0, szm = szl + 1;
    Val m = monoid::template merge<Val>(ok, v->sum);
    if(!l && !f(m)){
      ok = m;
      return -1;
    }
    int x = bisect_from_left(v->l, f, l, ok);
    if(x != -1) return x;
    if(l <= szl){
      ok = monoid::template merge<Val>(ok, v->val);
      if(f(ok)) return szl;
    }
    int res = bisect_from_left(v->r, f, std::max(l - szm, 0), ok);
    return res == -1 ? res : res + szm;
  }
  // https://atcoder.jp/contests/practice2/submissions/42115281
  template<typename F>
  int bisect_from_right(node *v, F &f, int r, Val &ok){
    if(!v || r < 0) return -1;
    int szl = v->l ? v->l->sz : 0, szm = szl + 1;
    Val m = monoid::template merge<Val>(ok, v->sum);
    if(v->sz <= r && !f(m)){
      ok = m;
      return -1;
    }
    int x = bisect_from_right(v->r, f, r - szm, ok);
    if(x != -1) return x + szm;
    if(szl <= r){
      ok = monoid::template merge<Val>(ok, v->val);
      if(f(ok)) return szl;
    }
    return bisect_from_right(v->l, f, r, ok);
  }
public:
  dynamic_sequence(): root(nullptr){}
  dynamic_sequence(const std::vector<Val> &v){
    if(v.empty()){
      root = nullptr;
      return;
    }
    int n = v.size();
    std::vector<node*> nodes(n);
    for(int i = 0; i < n; i++) nodes[i] = new node(v[i]);
    root = build(nodes, 0, n);
  }
  int size(){
    return size(root);
  }
  void set(int k, Val x){
    assert(0 <= k && k < size());
    set_inner(root, k, x);
  }
  Val get(int k){
    assert(0 <= k && k < size());
    return get_inner(root, k);
  }
  void insert(int k, Val x){
    assert(0 <= k && k <= size());
    root = insert_inner(root, k, x);
  }
  void erase(int k){
    assert(0 <= k && k < size());
    root = erase_inner(root, k);
  }
  void update(int k, Val x){
    assert(0 <= k && k < size());
    update_inner(root, k, x);
  }
  Val query(int l, int r){
    if(l >= r) return monoid::template id<Val>();
    assert(0 <= l && r <= size());
    return query_inner(root, l, r);
  }
  Val query_all(){
    if(!root) return monoid::template id<Val>();
    return root->sum;
  }
  // f(sum[l, r])がtrueになる最左のr. ない場合は-1
  template<typename F>
  int bisect_from_left(int l, const F &f){
    Val x = monoid::template id<Val>();
    return bisect_from_left(root, f, l, x);
  }
  // f(sum[l, r])がtrueになる最右のl. ない場合は-1
  template<typename F>
  int bisect_from_right(int r, const F &f){
    Val x = monoid::template id<Val>();
    return bisect_from_right(root, f, r, x);
  }
};

template<typename monoid, typename Val, typename Lazy>
struct lazy_dynamic_sequence{
private:
  struct node{
    int h, sz;
    Val val, sum;
    Lazy lazy;
    node *l, *r;
    node(Val _val = monoid::template id1<Val>()): h(1), sz(1), val(_val), sum(val),
    lazy(monoid::template id2<Lazy>()), l(nullptr), r(nullptr){}
    int balanace_factor(){
      return (l ? l->h : 0) - (r ? r->h : 0);
    }
  };
  node *root, *tmp_node;
  static int size(node *v){return v ? v->sz : 0;}
  static void update(node *v){
    v->h = std::max(v->l ? v->l->h : 0,  v->r ? v->r->h : 0) + 1;
    v->sz = 1;
    v->sum = v->val;
    if(v->l){
      v->sz += v->l->sz;
      v->sum = monoid::template merge<Val>(v->l->sum, v->sum);
    }
    if(v->r){
      v->sz += v->r->sz;
      v->sum = monoid::template merge<Val>(v->sum, v->r->sum);
    }
  }
  static void propagate(node *v, Lazy x){
    v->lazy = monoid::template propagate<Lazy>(v->lazy, x);
    v->val = monoid::template apply<Val, Lazy>(v->val, x, 0, 1);
    v->sum = monoid::template apply<Val, Lazy>(v->sum, x, 0, v->sz);
  }
  static void push_down(node *v){
    if(v->lazy != monoid::template id2<Lazy>()){
      if(v->l) propagate(v->l, v->lazy);
      if(v->r) propagate(v->r, v->lazy);
      v->lazy = monoid::template id2<Lazy>();
    }
  }
  static node *rotate_right(node *v){
    push_down(v);
    node *l = v->l;
    push_down(l);
    v->l = l->r;
    l->r = v;
    update(v);
    update(l);
    return l;
  }
  static node *rotate_left(node *v){
    push_down(v);
    node *r = v->r;
    push_down(r);
    v->r = r->l;
    r->l = v;
    update(v);
    update(r);
    return r;
  }
  node *build(const std::vector<node*> &nodes, int l, int r){
    int m = (l + r) >> 1;
    node *v = nodes[m];
    if(m > l) v->l = build(nodes, l, m);
    if(r > m + 1) v->r = build(nodes, m + 1, r);
    update(v);
    return v;
  }
  node *balance(node *v){
    int bf = v->balanace_factor();
    assert(-2 <= bf && bf <= 2);
    push_down(v);
    if(bf == 2){
      if(v->l->balanace_factor() == -1){
        v->l = rotate_left(v->l);
        update(v);
      }
      return rotate_right(v);
    }else if(bf == -2){
      if(v->r->balanace_factor() == 1){
        v->r = rotate_right(v->r);
        update(v);
      }
      return rotate_left(v);
    }
    return v;
  }
  void set_inner(node *v, int k, Val x){
    push_down(v);
    int szl = v->l ? v->l->sz : 0;
    if(k < szl) set_inner(v->l, k, x);
    else if(k > szl) set_inner(v->r, k - szl - 1, x);
    else v->val = x;
    update(v);
  }
  Val get_inner(node *v, int k){
    push_down(v);
    int szl = v->l ? v->l->sz : 0;
    if(k < szl) return get_inner(v->l, k);
    else if(k > szl) return get_inner(v->r, k - szl - 1);
    else return v->val;
  }
  void update_inner(node *v, int l, int r, Lazy x){
    if(!v) return;
    if(l == 0 && r == v->sz){
      propagate(v, x);
      return;
    }
    push_down(v);
    int szl = size(v->l), szv = szl + 1;
    if(l < szl){
      if(r <= szl){
        update_inner(v->l, l, r, x);
        update(v);
        return;
      }
      update_inner(v->l, l, szl, x);
      l = szl;
    }
    if(szv < r){
      if(szv <= l){
        update_inner(v->r, l - szv, r - szv, x);
        update(v);
        return;
      }
      update_inner(v->r, 0, r - szv, x);
      r = szv;
    }
    if(r == l + 1) v->val = monoid::template apply<Val, Lazy>(v->val, x, 0, 1);
    update(v);
  }
  Val query_inner(node *v, int l, int r){
    if(!v) return monoid::template id1<Val>();
    if(l == 0 && r == v->sz) return v->sum;
    push_down(v);
    int szl = size(v->l), szv = szl + 1;
    Val left_q = monoid::template id1<Val>(), right_q = left_q;
    if(l < szl){
      if(r <= szl) return query_inner(v->l, l, r);
      left_q = query_inner(v->l, l, szl);
      l = szl;
    }
    if(szv < r){
      if(szv <= l) return query_inner(v->r, l - szv, r - szv);
      right_q = query_inner(v->r, 0, r - szv);
      r = szv;
    }
    Val res = (l == r ? monoid::template id1<Val>() : v->val);
    res = monoid::template merge<Val>(left_q, res);
    res = monoid::template merge<Val>(res, right_q);
    return res;
  }
  node *cut_left_most(node *v){
    push_down(v);
    if(v->l){
      v->l = cut_left_most(v->l);
      update(v);
      return balance(v);
    }
    tmp_node = v;
    return v->r;
  }
  node *cut_right_most(node *v){
    push_down(v);
    if(v->r){
      v->r = cut_right_most(v->r);
      update(v);
      return balance(v);
    }
    tmp_node = v;
    return v->l;
  }
  node *insert_inner(node *v, int k, Val x){
    if(!v) return new node(x);
    push_down(v);
    int szl = v->l ? v->l->sz : 0;
    if(k <= szl) v->l = insert_inner(v->l, k, x);
    else if(k > szl) v->r = insert_inner(v->r, k - szl - 1, x);
    update(v);
    return balance(v);
  }
  node *erase_inner(node *v, int k){
    push_down(v);
    int szl = v->l ? v->l->sz : 0;
    if(k < szl) v->l = erase_inner(v->l, k);
    else if(k > szl) v->r = erase_inner(v->r, k - szl - 1);
    else{
      if(!v->r) return v->l;
      node *u = cut_left_most(v->r);
      tmp_node->l = v->l;
      tmp_node->r = u;
      v = tmp_node;
    }
    update(v);
    return balance(v);
  }
  // https://atcoder.jp/contests/practice2/submissions/42114957
  template<typename F>
  int bisect_from_left(node *v, F &f, int l, Val &ok){
    if(!v || v->sz <= l) return -1;
    int szl = v->l ? v->l->sz : 0, szm = szl + 1;
    Val m = monoid::template merge<Val>(ok, v->sum);
    if(!l && !f(m)){
      ok = m;
      return -1;
    }
    push_down(v);
    int x = bisect_from_left(v->l, f, l, ok);
    if(x != -1) return x;
    if(l <= szl){
      ok = monoid::template merge<Val>(ok, v->val);
      if(f(ok)) return szl;
    }
    int res = bisect_from_left(v->r, f, std::max(l - szm, 0), ok);
    return res == -1 ? res : res + szm;
  }
  // https://atcoder.jp/contests/practice2/submissions/42115281
  template<typename F>
  int bisect_from_right(node *v, F &f, int r, Val &ok){
    if(!v || r < 0) return -1;
    int szl = v->l ? v->l->sz : 0, szm = szl + 1;
    Val m = monoid::template merge<Val>(ok, v->sum);
    if(v->sz <= r && !f(m)){
      ok = m;
      return -1;
    }
    push_down(v);
    int x = bisect_from_right(v->r, f, r - szm, ok);
    if(x != -1) return x + szm;
    if(szl <= r){
      ok = monoid::template merge<Val>(ok, v->val);
      if(f(ok)) return szl;
    }
    return bisect_from_right(v->l, f, r, ok);
  }
public:
  lazy_dynamic_sequence(): root(nullptr){}
  lazy_dynamic_sequence(const std::vector<Val> &v){
    if(v.empty()){
      root = nullptr;
      return;
    }
    int n = v.size();
    std::vector<node*> nodes(n);
    for(int i = 0; i < n; i++) nodes[i] = new node(v[i]);
    root = build(nodes, 0, n);
  }
  int size(){
    return size(root);
  }
  void set(int k, Val x){
    assert(0 <= k && k < size());
    set_inner(root, k, x);
  }
  Val get(int k){
    assert(0 <= k && k < size());
    return get_inner(root, k);
  }
  void insert(int k, Val x){
    assert(0 <= k && k <= size());
    root = insert_inner(root, k, x);
  }
  void erase(int k){
    assert(0 <= k && k < size());
    root = erase_inner(root, k);
  }
  void update(int l, int r, Lazy x){
    if(l >= r) return;
    assert(0 <= l && r <= size());
    update_inner(root, l, r, x);
  }
  Val query(int l, int r){
    if(l >= r) return monoid::template id1<Val>();
    assert(0 <= l && r <= size());
    return query_inner(root, l, r);
  }
  Val query_all(){
    if(!root) return monoid::template id1<Val>();
    return root->sum;
  }
  // f(sum[l, r])がtrueになる最左のr. ない場合は-1
  template<typename F>
  int bisect_from_left(int l, const F &f){
    Val x = monoid::template id1<Val>();
    return bisect_from_left(root, f, l, x);
  }
  // f(sum[l, r])がtrueになる最右のl. ない場合は-1
  template<typename F>
  int bisect_from_right(int r, const F &f){
    Val x = monoid::template id1<Val>();
    return bisect_from_right(root, f, r, x);
  }
};


void solve(){
  int n, q;
  std::cin >> n >> q;
  auto a = read_vec<int>(n);

  // q > 0の時のみ参加できる
  // ai > qならq--
  // ai <= qなら何もしない

  // 仕切りの左のq以下のコンテストに全て参加
  
  vector<int> left_lower(n, 0);
  range(i, 0, n) left_lower[i] = (i == 0 ? 0 : left_lower[i - 1]) + (a[i] <= q);
  
  lazy_dynamic_sequence<range_add_range_min, ll, ll> seg;

  int ans = left_lower[n - 1], ansidx = n;
  int low_count = 0;
  for(int i = n - 1; i >= 0 && low_count <= q; i--){
    seg.insert(0, q - a[i]);
    while(seg.size() && seg.query_all() < 0){
      low_count++;
      if(low_count > q) break;
      int idx = seg.bisect_from_left(0, [&](ll x){return x < 0;});
      assert(idx != -1);
      seg.update(idx, seg.size(), -1);
      seg.erase(idx);
    }
    if(low_count <= q){
      int tmp = n - i + (i ? left_lower[i - 1] : 0);
      if(ans < tmp){
        ans = tmp;
        ansidx = i;
      }
    }
  }
  string s(n, '0');
  range(i, 0, ansidx){
    if(a[i] <= q){
      s[i] = '1';
    }
  }
  range(i, ansidx, n){
    s[i] = '1';
  }
  std::cout << s << '\n';
}
int main(){
  io_init();
  int t;
  std::cin >> t;
  range(i, 0, t) solve();
}

#ifdef ONLINE_JUDGE
#define NDEBUG
#endif

#include <cassert>
#include <climits>

#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <set>
#include <utility>
#include <vector>

#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>

static_assert(sizeof(int) * CHAR_BIT >= 32);

template <typename T>
using OrderedSet =
    __gnu_pbds::tree<T, __gnu_pbds::null_type, std::less<>,
                     __gnu_pbds::rb_tree_tag,
                     __gnu_pbds::tree_order_statistics_node_update>;

constexpr int D_MAX = 1000000;
constexpr int INF = std::numeric_limits<int>::max();

bool test = false;

// https://www.luogu.com.cn/record/78000275
namespace Golem // 1.0.0.4
{
#define fcc(i, j, k) for (int(i) = (j); (i) <= (k); ++(i))
#define ccf(i, j, k) for (int(i) = (j); (i) >= (k); --(i))

template <class V> class block {
public:
  std::vector<V> val;
  int operator~() { return val.size(); }
  auto find(V v) { return std::lower_bound(val.begin(), val.end(), v); }
  void operator+=(V v) { val.emplace(find(v), v); }
  bool operator-=(V v) {
    auto it = find(v);
    return *it == v ? (val.erase(it), 1) : 0;
  }
  V &operator[](int k) { return val[k - 1]; }
  int rnk(V v) { return find(v) - val.begin() + 1; }
  V pre(V v) { return *--find(v); }
  V nxt(V v) { return *std::upper_bound(val.begin(), val.end(), v); }
};

template <class V> class box {
#define B blo[b]
  int blo_len = 500, siz = 0;
  std::vector<block<V>> blo;
  V head(int b) { return B.val.front(); }
  V tail(int b) { return B.val.back(); }
  int pos(V v) {
    if (!siz)
      return 0;
    int res = (int)blo.size() - 1;
    for (int l = 0, r = res, m = l + r >> 1; l <= r; m = l + r >> 1)
      !(tail(m) < v) ? (res = m, r = m - 1) : (l = m + 1);
    return res;
  }
  void erase(int b) {
    if (!~B)
      blo.erase(blo.begin() + b);
  }

public:
  void clear() { siz = 0, blo.clear(); }
  bool empty() { return !siz; }
  int size() { return siz; }
  void fit(int len) { blo_len = std::max(3, len); }
  void insert(V v) {
    if (!siz)
      blo.resize(1);
    int b = pos(v);
    if ((++siz, B += v, ~B) > blo_len) {
      auto it = B.val.begin() + blo_len / 2;
      block<V> bl;
      bl.val.assign(it, B.val.end());
      B.val.resize(blo_len / 2), blo.emplace(blo.begin() + b + 1, bl);
    }
  }
  bool remove(V v) {
    int b = pos(v);
    return (B -= v) ? (--siz, erase(b), 1) : 0;
  }
  V kth(int k) {
    int cur = 0;
    for (; k > ~blo[cur];)
      k -= ~blo[cur++];
    return blo[cur][k];
  }
  int rnk(V v) {
    int b = pos(v), res = B.rnk(v);
    ccf(i, --b, 0) res += ~blo[i];
    return res;
  }
  V pre(V v) {
    int b = pos(v);
    for (; v < head(b) || v == head(b);)
      --b;
    return B.pre(v);
  }
  V nxt(V v) {
    int b = pos(v);
    for (; tail(b) < v || v == tail(b);)
      ++b;
    return B.nxt(v);
  }
  V front() { return blo.front().val.front(); }
  V back() { return blo.back().val.back(); }
  void pop_front() {
    --siz, blo.front().val.erase(blo.front().val.begin()), erase(0);
  }
  void pop_back() {
    --siz, blo.back().val.pop_back(), erase((int)blo.size() - 1);
  }
  bool find(V v) {
    if (!siz || back() < v)
      return 0;
    int b = pos(v);
    return *B.find(v) == v;
  }
#undef B
};
} // namespace Golem

template <typename T> using Set = Golem::box<T>;

namespace Global {
OrderedSet<int> validChoices;
}
using namespace Global;

class Tree {
private:
  std::vector<int> val;
  std::vector<std::vector<int>> edges;

  struct Block {
    int bound;
    std::vector<int> nodes;

    Block(int bound) : bound(bound) {}

    bool operator<(const Block &other) const { return bound < other.bound; }
  };

  std::vector<Block> blocks;

  bool dfs(int u, Block &block) {
    for (const auto &v : edges[u]) {
      if (val[u] && val[v] && val[u] < val[v])
        return false;
      if (val[u]) {
        Block newBlock(val[u]);
        if (!dfs(v, newBlock))
          return false;
        if (!newBlock.nodes.empty())
          blocks.push_back(std::move(newBlock));
      } else {
        if (!dfs(v, block))
          return false;
      }
    }
    if (!val[u])
      block.nodes.push_back(u);
    return true;
  }

public:
  Tree(const std::vector<int> &val) : val(val), edges(val.size()) {}

  void addEdge(int u, int v) { edges[u].push_back(v); }

  // 返回任一可行值；无则为 -1
  int prepare(int root, Set<int> set) {
    {
      Block block(std::numeric_limits<int>::max());
      if (!dfs(root, block))
        return -1;
      if (!block.nodes.empty())
        blocks.push_back(std::move(block));
    }
    std::sort(blocks.begin(), blocks.end());
    if (test)
      return -1;

    bool used = false;
    int extraVal = -1;
    std::vector<int> tmpVal = val;
    set.insert(INF);
    const auto work = [&tmpVal, this](const Block &block, Set<int> &s) {
      for (const auto &u : block.nodes) {
        int bound = 0;
        for (const auto &v : edges[u])
          bound = std::max(bound, tmpVal[v]);
        const int x = s.nxt(bound);
        if (x == INF || x > block.bound) {
          for (const auto &w : block.nodes) {
            if (w == u)
              break;
            s.insert(tmpVal[w]);
          }
          return false;
        }
        tmpVal[u] = x;
        s.remove(x);
      }
      return true;
    };
    int p = -1;
    Set<int> oldSet;
    for (int index = 0; index < static_cast<int>(blocks.size()); ++index) {
      const auto &block = blocks[index];
      if (used) {
        assert(extraVal != -1);
        if (!work(block, set)) {
          extraVal = -1;
          break;
        }
        continue;
      }
      if (work(block, set))
        continue;
      int minBound = std::numeric_limits<int>::max();
      for (const auto &u : block.nodes) {
        int bound = 0;
        for (const auto &v : edges[u]) {
          if (val[v] == 0) {
            bound = -1;
            break;
          }
          if (val[v] > bound)
            bound = val[v];
        }
        if (bound == -1)
          break;
        if (bound < minBound)
          minBound = bound;
      }
      p = index;
      used = true;
      oldSet = set;
      const auto iter = validChoices.upper_bound(minBound);
      if (iter == validChoices.end())
        break;
      extraVal = *iter;
      set.insert(extraVal);
      if (!work(block, set)) {
        extraVal = -1;
        break;
      }
    }
    assert(used);
    if (extraVal != -1)
      return extraVal;
    if (test) {
      std::cout << "f";
    }
    set = std::move(oldSet);
    const auto iter = validChoices.lower_bound(blocks[p].bound);
    if (iter != validChoices.begin()) {
      extraVal = *std::prev(iter);
      set.insert(extraVal);
      for (int index = p; index < static_cast<int>(blocks.size()); ++index) {
        if (!work(blocks[index], set)) {
          extraVal = -1;
          break;
        }
      }
    }
    return extraVal;
  }

  bool check(Set<int> set) const {
    set.insert(INF);
    std::vector<int> tmpVal = val;
    for (const auto &block : blocks) {
      for (const auto &u : block.nodes) {
        int bound = 0;
        for (const auto &v : edges[u])
          bound = std::max(bound, tmpVal[v]);
        const int x = set.nxt(bound);
        if (x == INF || x > block.bound)
          return false;
        tmpVal[u] = x;
        set.remove(x);
      }
    }
    assert(set.size() == 1);
    return true;
  }
};

class SparseTable {
private:
  std::vector<int> val, log2V;
  std::vector<std::vector<int>> data;

  int getMaxIndexOfTwo(int p, int q) const { return val[p] > val[q] ? p : q; }

public:
  SparseTable(const std::vector<int> &val) : val(val) {
    const int n = val.size();

    log2V.resize(n + 1);
    log2V[0] = -1;
    for (int i = 1; i <= n; ++i)
      log2V[i] = log2V[i / 2] + 1;

    data.resize(log2V[n] + 1);
    data[0].resize(n);
    for (int i = 0; i < n; ++i)
      data[0][i] = i;
    for (int i = 1; i <= log2V[n]; ++i) {
      data[i].resize(n - (1 << i) + 1);
      for (int j = 0; j + (1 << i) <= n; ++j)
        data[i][j] =
            getMaxIndexOfTwo(data[i - 1][j], data[i - 1][j + (1 << (i - 1))]);
    }
  }

  int getMaxIndex(int l, int r) const {
    const int t = log2V[r - l + 1];
    return getMaxIndexOfTwo(data[t][l], data[t][r - (1 << t) + 1]);
  }
};

int main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);
  for (int i = 1; i <= D_MAX; ++i)
    validChoices.insert(i);
  int t;
  std::cin >> t;
  while (t--) {
    int n, q;
    std::cin >> n >> q;
    std::vector<int> p(n), a(n);
    // if (n == 205261)
    //   test = true;
    for (auto &item : p)
      std::cin >> item;
    int k = 0;
    for (auto &item : a) {
      std::cin >> item;
      if (item == 0)
        ++k;
      else
        validChoices.erase(item);
    }
    Set<int> set;
    std::vector<int> v;
    for (int i = 1; i < k; ++i) {
      int x;
      std::cin >> x;
      validChoices.erase(x);
      set.insert(x);
      v.push_back(x);
    }
    SparseTable st(p);
    Tree tree(a);
    const std::function<int(int, int)> construct = [&st, &tree,
                                                    &construct](int l, int r) {
      const int index = st.getMaxIndex(l, r);
      if (index > l)
        tree.addEdge(index, construct(l, index - 1));
      if (index < r)
        tree.addEdge(index, construct(index + 1, r));
      return index;
    };
    const int s = tree.prepare(construct(0, n - 1), set);
    if (s == -1) {
      while (q--) {
        int d;
        std::cin >> d;
        std::cout << "NO\n";
      }
    } else {
      assert(validChoices.find(s) != validChoices.end());
      const int index = validChoices.order_of_key(s);
      int l = 0, r = index;
      while (l < r) {
        const int mid = (l + r) / 2;
        const int x = *validChoices.find_by_order(mid);
        set.insert(x);
        if (tree.check(set))
          r = mid;
        else
          l = mid + 1;
        set.remove(x);
      }
      const int mn = *validChoices.find_by_order(l);
      l = index;
      r = validChoices.size() - 1;
      while (l < r) {
        const int mid = (l + r + 1) / 2;
        const int x = *validChoices.find_by_order(mid);
        set.insert(x);
        if (tree.check(set))
          l = mid;
        else
          r = mid - 1;
        set.remove(x);
      }
      const int mx = *validChoices.find_by_order(l);
      while (q--) {
        int d;
        std::cin >> d;
        std::cout << (d >= mn && d <= mx ? "YES\n" : "NO\n");
      }
    }
    for (const auto &item : a) {
      if (item != 0)
        validChoices.insert(item);
    }
    for (const auto &item : v)
      validChoices.insert(item);
    // if (test)
    //   break;
  }
  return 0;
}
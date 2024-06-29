#include <bits/stdc++.h>
using namespace std;

template <class T> istream &operator>>(istream &is, vector<T> &v) { for (auto &x : v) is >> x; return is; }
template <class T> ostream &operator<<(ostream &os, const vector<T> &v) { auto sep = ""; for (auto &x : v) os << exchange(sep, " ") << x; return os; }
template <class T> vector<T> &operator+=(vector<T> &v, const T &a) { for (auto &x : v) x += a; return v; }
template <class T> vector<T> &operator-=(vector<T> &v, const T &a) { for (auto &x : v) x -= a; return v; }
template <class T, class U> inline bool chmax(T &a, const U &b) { return (a < b ? a = b, 1 : 0); }
template <class T, class U> inline bool chmin(T &a, const U &b) { return (b < a ? a = b, 1 : 0); }
#define si(x) int(x.size())
#define all(x) (x).begin(),(x).end()
#define rall(x) (x).rbegin(),(x).rend()
#define fi first
#define se second
using ll = long long;
constexpr int inf = 1001001001;
constexpr long long infll = 4004004004004004004LL;

int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);

  int N, K;
  cin >> N >> K;
  vector a(N, vector<int>(N));
  for (int i = 0; i < N; i++) {
    cin >> a[i];
  }
  int l = 0, r = inf;
  auto valid = [&](int x) {
    vector p(N+1, vector<int>(N+1));
    bool found = 0;
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        p[i+1][j+1] = p[i+1][j] + p[i][j+1] - p[i][j] + (a[i][j] <= x);
        if (i >= K - 1 && j >= K - 1) {
          found |= (p[i+1][j+1] - p[i+1-K][j+1] - p[i+1][j+1-K] + p[i+1-K][j+1-K]) >= ((K * K + 1) / 2);
        }
      }
    }
    return found;
  };
  while (l < r) {
    int mid = (l + r) / 2;
    valid(mid) ? r = mid : l = mid + 1;
  }

  cout << l << '\n';
}

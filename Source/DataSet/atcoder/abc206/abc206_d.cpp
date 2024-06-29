#include <bits/stdc++.h>
using namespace std;

#ifdef LOCAL_JUDGE
#include "mydebug.h"
#else
#define print(...)
#endif

#define int long long
#define rep(i,a,b) for(int i=a; i<=b; i++)
#define per(i,a,b) for(int i=b; i>=a; i--)
#define sz(x) (int)x.size()

const int N = 200005;
int n, A[N], ans, fa[N];
int leader(int x) {
    return x == fa[x] ? x : fa[x] = leader(fa[x]);
}
void join(int x, int y) {
    x = leader(x);
    y = leader(y);
    if (x != y) {
        fa[x] = y;
        ans += 1;
    }
}
void solve() {
    cin >> n;
    rep(i,1,n) cin >> A[i], fa[A[i]] = A[i];
    for (int i = 1, j = n; i < j; i++, j--) {
        join(A[i], A[j]);
    }
    cout << ans << "\n";
}

signed main() {
    cin.tie(0)->sync_with_stdio(0);
    int t = 1;
    while (t--) solve();
    return 0;
}
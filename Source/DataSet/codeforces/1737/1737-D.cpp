// clang-format off
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include <bitset>
#include <queue>
#include <stack>
#include <sstream>
#include <cstring>
#include <numeric>
#include <ctime>
#include <cassert>
#include <random>
#include <deque>
#include <valarray>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <deque>
#include <functional>
#include <list>

using namespace std;

//#define DEBUG

#define re return
#define fi first
#define se second
#define mp make_pair
#define pb emplace_back
#define all(x) x.begin(), x.end()
#define mfor(i, start, end) for (int i = (start); i < (end); i++)
#define rep(i, n) for (int i = 0; i < (n); i++)
#define rrep(i, n) for (int i = (n) - 1; i >= 0; i--)
#define fill(x, y) memset(x, y, sizeof(x))
#define sqr(x) ((x)*(x))
#define unq(x) (x.resize(unique(all(x)) - x.begin()))
#define ba back()
#define popcount __builtin_popcountll
#define fastIO() {ios::sync_with_stdio(0); cin.tie(0); cout.tie(0);}
#define readArr(arr, n) {for (int i = 0; i < n; i++) cin >> arr[i];}
#define nl '\n'
#define uset unordered_set
#define umap unordered_map
#define int long long

#ifdef DEBUG

#define deb(x) cout << #x << "=" << x << ' '
#define debnl(x) cout << #x << "=" << x << endl
#define name(x) cout << #x << ' ';
#define print(v) {cout << #v << endl; for (auto x : v) cout << x << ' '; cout << endl;}
#define printnl(v) {cout << #v << endl; for (auto x : v) cout << x << endl;}
#define print1(v) {cout << #v << endl; for (auto x : v) cout << x + 1 << ' '; cout << endl;}
#define print2d(v) { cout << #v << endl; for (auto x : v) { for (auto y : x) cout << y << ' '; cout << endl; }}
#define print2d1(v) { cout << #v << endl; for (auto x : v) { for (auto y : x) cout << y + 1 << ' '; cout << endl; }}
#define printpair(v) { cout << #v << endl; for (auto x : v) cout << '(' << x.first << ',' << x.second << ')' << ' '; cout << endl; };
#define print2dpair(v) { cout << #v << endl; for (auto x : v) { for (auto y : x) cout << '(' << y.first << ',' << y.second << ')' << ' '; cout << endl; }}

#else


#define deb(x)
#define debnl(x)
#define name(x)
#define print(v)
#define printnl(v)
#define print1(v)
#define print2d(v)
#define print2d1(v)
#define printpair(v)
#define print2dpair(v)


#endif // DEBUG

typedef vector<int> vi;
typedef vector<vi> vvi;
typedef vector<vvi> vvvi;
typedef pair<int, int> ii;
typedef vector<ii> vii;
typedef vector<vii> vvii;
typedef vector<vvii> vvvii;
typedef long long ll;
typedef unsigned long long ull;
typedef vector<char> vc;
typedef vector<vc> vvc;
typedef pair<ll, ll> pll;
typedef vector<ll> vll;
typedef vector<vll> vvll;
typedef vector<vvll> vvvll;
typedef vector<pll> vpll;
typedef vector<vpll> vvpll;
typedef vector<vvpll> vvvpll;
typedef long double LD;
typedef double D;
typedef pair<D, D> pdd;
typedef pair<LD, LD> pldd;

template<class T1, class T2> ostream& operator << (ostream &o, pair<T1, T2> x) {re o << '(' << x.fi << ", " << x.se << ')';}
template<class T1, class T2> istream& operator >> (istream &o, pair<T1, T2> &x) {re o >> x.fi >> x.se;}
template<class T1, class T2> pair<T1, T2> operator + (pair<T1, T2> a, pair<T1, T2> b) {a.fi += b.fi; a.se += b.se; re a;}
template<class T1, class T2> pair<T1, T2> operator - (pair<T1, T2> a, pair<T1, T2> b) {a.fi -= b.fi; a.se -= b.se; re a;}
template<class T1, class T2> void operator += (pair<T1, T2> &a, pair<T1, T2> b) {a.fi += b.fi; a.se += b.se;}
template<class T1, class T2> void operator -= (pair<T1, T2> &a, pair<T1, T2> b) {a.fi -= b.fi; a.se -= b.se;}
template<class T> vector<T> operator + (vector<T> a, vector<T> b) { a.insert(a.end(), b.begin(), b.end()); re a; }
template<class T> void operator += (vector<T> &a, vector<T> b) { a.insert(a.end(), b.begin(), b.end()); }

mt19937 rnd(12);

const ll INF = 1e18 + 1;
const ll MINIINF = 1e9+1;
const ll mod = 1e9 + 7;

void solve();

signed main() {
	fastIO();
	//freopen("file.in", "r", stdin);
	//freopen("file.out", "w", stdout);
	int T = 1;
	cin >> T;
	while (T--) {
		//		cout << "Case #" << tt + 1 << ": ";
		solve();
	}
}


/* ================ actual code starts here ================ */

struct Edge{
    ll u, v, w;
};

vvii gr;
vector<Edge> es;

void bfs(int s, int forbidden, vi &d, vi &pred){
    queue<int> q;
    d[s] = 0;
    q.push(s);
    while (!q.empty()){
        int u = q.front();
        q.pop();
        for (auto tmp: gr[u]){
            int v = tmp.first;
            int ind = tmp.second;
            if (ind == forbidden) continue;
            if (d[u] + 1 < d[v]){
                pred[v] = u;
                d[v] = d[u] + 1;
                q.push(v);
            }
        }
    }
}

void solve() {
    int n, m;
    cin >> n >> m;
    gr = vvii(n);
    es.clear();
    for (int i = 0; i < m; i++){
        int u, v, w;
        cin >> u >> v >> w;
        u--, v--;
        gr[u].push_back({v, i});
        gr[v].push_back({u, i});
        es.push_back({u, v, w});
    }
    vvll d(n, vi(n, MINIINF));
    vii banned(n);
    vvll d1(n, vi(n, MINIINF));
    vvll dn(n, vi(n, MINIINF));
    vvi pred(n, vi(n, -1));
    for (int s = 0; s < n; s++){
        bfs(s, -1, d[s], pred[s]);
        int ban1 = -1, bann = -1;
        if (s != 0){
            int u = 0;
            while (pred[s][u] != s) u = pred[s][u];
            for (auto tmp: gr[s]){
                int v = tmp.first, ind = tmp.second;
                if (v == u){
                    ban1 = ind;
                    break;
                }
            }
        }
        if (s != n - 1){
            int u = n - 1;
            while (pred[s][u] != s) u = pred[s][u];
            for (auto tmp: gr[s]){
                int v = tmp.first, ind = tmp.second;
                if (v == u){
                    bann = ind;
                    break;
                }
            }
        }
        bfs(s, ban1, d1[s], pred[s]);
        bfs(s, bann, dn[s], pred[s]);
        banned[s] = {ban1, bann};
    }

    print2d(d);
    print2d(d1);
    print2d(dn);

    ll ans = INF;
    for (int u = 0; u < n; u++){
        for (int i = 0; i < m; i++){
            int x = es[i].u, y = es[i].v, w = es[i].w;
            ll t = min(d[x][u], d[y][u]) + 2;
            ll value = (t + d[u][0] + d[u][n - 1]) * w;
            ans = min(ans, value);
            deb(u); deb(i); debnl(ans);
        }
    }
    for (int i = 0; i < m; i++){
        int u = es[i].u, v = es[i].v;
        ll w = es[i].w;
        ll dist1 = 0, dist2 = 0;
        // 1 - u, u - v, v - n
        if (banned[u].first != i) dist1 += d[u][0];
        else dist1 += d1[u][0];
        if (banned[v].second != i) dist1 += d[v][n - 1];
        else dist1 += dn[v][n - 1];

        // n - u, u - v, v - 1
        if (banned[u].second != i) dist2 += d[u][n  -1];
        else dist2 += dn[u][n - 1];
        if (banned[v].first != i) dist2 += d[v][0];
        else dist2 += d1[v][0];

        ll value = (min(dist1, dist2) + 1) * w;
        ans = min(ans, value);
        deb(i); debnl(ans);
    }

    cout << ans << nl;
}









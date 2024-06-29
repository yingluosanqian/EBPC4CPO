#include <bits/stdc++.h>
#define x first
#define y second
#define debug(x) cout<<#x<<":"<<x<<endl;
using namespace std;
typedef long double ld;
typedef long long LL;
typedef pair<int, int> PII;
typedef pair<double, double> PDD;
typedef unsigned long long ULL;
const int N = 2e5 + 10, M = 2 * N, INF = 0x3f3f3f3f, mod = 1e9 + 7;
const double eps = 1e-8, pi = acos(-1), inf = 1e20;
int dx[] = {-1, 0, 1, 0}, dy[] = {0, 1, 0, -1};
int h[N], e[M], ne[M], w[M], idx;
void add(int a, int b, int v = 0) {
    e[idx] = b, w[idx] = v, ne[idx] = h[a], h[a] = idx ++;
}
int n, m, k;
int a[N];
struct Node {
    int x, y, w;
    bool operator<(Node t)const {
        return w < t.w;
    }
}edges[N];
int f[N];
int find(int x) {
    return x == f[x] ? x : f[x] = find(f[x]);
}
int fa[N][25], dep[N];
void dfs(int u, int v) {
    dep[u] = dep[v] + 1, fa[u][0] = v;
    for (int j = 1; j <= 20; j ++)
        fa[u][j] = fa[fa[u][j-1]][j-1];
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == v) continue; 
        dfs(j, u);
    }
}
int lca(int x, int y) {
    if (dep[x] < dep[y]) return lca(y, x);
    for (int j = 20; j >= 0; j --)
        if (dep[fa[x][j]] >= dep[y])
            x = fa[x][j];
    
    if (x == y) return x;
    for (int j = 20; j >= 0; j --)
        if (fa[x][j] != fa[y][j]) 
            x = fa[x][j], y = fa[y][j];
    return fa[x][0];
}
int jump(int u, int d) {
    for (int j = 20; j >= 0; j --)
        if (dep[fa[u][j]] >= d)
            u = fa[u][j];
    return u;
}
int s[N];
void dfs_1(int u, int v) {
    s[u] += s[v];
    for (int i = h[u]; ~i; i = ne[i]) {        
        int j = e[i];        
        if (j == v) continue ;        
        dfs_1(j, u);
    }
}
int main() {
    ios::sync_with_stdio(false), cin.tie(0);
    cin >> n >> m;
    memset(h, -1, sizeof h);
    for (int i = 1; i <= n; i ++) f[i] = i;
    for (int i = 1; i <= m; i ++) cin >> edges[i].x >> edges[i].y, edges[i].w = i;
    sort(edges + 1, edges + 1 + m);

    vector<Node> q;
    int cnt = 0;
    for (int i = 1; i <= m; i ++) {
        if (cnt == n - 1) {
            q.push_back(edges[i]);
            continue ;
        }
        int x = edges[i].x, y = edges[i].y;
        if (find(x) != find(y))  cnt ++, f[find(x)] = find(y), add(x, y), add(y, x);
        else q.push_back(edges[i]);
    }

    dfs(1, 0);   
    for (auto t : q) {
        int root = lca(t.x, t.y);
        if (root == t.x) {
            int d = dep[t.x] + 1;
            int u = jump(t.y, d);
            s[u] ++, s[t.y] --;
        }
        else if (root == t.y) {
            swap(t.x, t.y);
            int d = dep[t.x] + 1;
            int u = jump(t.y, d);
            s[u] ++, s[t.y] --;
        }
        else  s[1] ++, s[t.x] --, s[t.y] --;        
        
    }

    dfs_1(1, 0);

   // for (int i = 1; i <= n; i ++) cout << s[i] << " \n"[i == n];
    for (int i = 1; i <= n; i ++)
        if (s[i]) cout << 0;
        else cout << 1;
    cout << '\n';
    return 0;
}
// LUOGU_RID: 148845480
#include<bits/stdc++.h>
using namespace std;
#define int long long
const int N = 200005;
int n, m, idx = 1, in1, in2, in3, in4;
int to[N], nxt[N], head[N], vis[N], dis[N], c[N], d[N];
void add(int u, int v, int x, int y) {
    idx++; to[idx] = v; nxt[idx] = head[u]; head[u] = idx; c[idx] = x; d[idx] = y;
}
struct Node {
    int x, time;
    bool operator < (const Node& b) const {
        return time > b.time;
    }
}now;
priority_queue <Node> q;
void Dijkstra() {
    memset(dis, 0x3f, sizeof dis);
    dis[1] = 0; q.push(Node{ 1,0 });
    while (!q.empty()) {
        now = q.top(); q.pop();
        if (vis[now.x]) continue;
        vis[now.x] = 1;
        for (int i = head[now.x]; i; i = nxt[i]) {
            int v = to[i], t;
            if (now.time <= round(sqrt(d[i])) - 1) t = round(sqrt(d[i])) - 1;
            else t = now.time;
            if (dis[v] > t + c[i] + d[i] / (t + 1)) {
                dis[v] = t + c[i] + d[i] / (t + 1);
                q.push(Node{ v,dis[v] });
            }
        }
    }
}
signed main() {
    scanf("%lld%lld", &n, &m);
    for (int i = 1; i <= m; i++)
        scanf("%lld%lld%lld%lld", &in1, &in2, &in3, &in4),
        add(in1, in2, in3, in4), add(in2, in1, in3, in4);
    Dijkstra();
    if (dis[n] != 0x3f3f3f3f3f3f3f3f)
        cout << dis[n];
    else cout << -1;
    return 0;
}
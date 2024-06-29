#include <bits/stdc++.h>
using namespace std;
#define int long long

const int N = 2010;
int h[N], e[N], ne[N], idx;
bool st[N];
int cnt;

void add(int a, int b) {
	e[idx] = b, ne[idx] = h[a], h[a] = idx++;
}

void bfs(int u) {
	queue<int> q;
	q.push(u);
	st[u] = true;
	
	while (q.size()) {
		int t = q.front();
		q.pop();
		
		for (int i = h[t]; i != -1; i = ne[i]) {
			int j = e[i];
			if (!st[j]) {
				cnt++;
				q.push(j);
				st[j] = true;
			}
		}
	}
}

void solve() {
    int n, m;
    cin >> n >> m;
	memset(h, -1, sizeof h);
    while (m--) {
		int a, b;
		cin >> a >> b;
		add(a, b);
	}
	for (int i = 1; i <= n; i++) {
		memset(st, 0, sizeof st);
		bfs(i);
		// cout << cnt << endl;
	}
	cout << cnt + n;
}

signed main() {
	ios::sync_with_stdio(false);
	cin.tie(0);
	int t = 1;
	// cin >> t;
	for (int i = 1; i <= t; i++) {
		solve();
	}
	return 0;
}
#include <bits/stdc++.h>

using namespace std;

template <typename T> inline void read(T &x){
	static int c = getchar(); static bool f; f = false, x = 0;
	for (; c > 57 || c < 48; c = getchar()) f |= c == 45;
	for (; c >= 48 && c <= 57; c = getchar()) x = (x << 1) + (x << 3) + (c ^ 48);
	if (f) x = -x;
}

template <typename T, typename ...Args> inline void read(T &x, Args &...args){
	read(x), read(args...);
}

using ll = long long;
const int N = 4e6 + 5;
int n;
vector <int> e[N];
vector <int> prime, vec[N];
int lsh[N], tot, cnt;
int dfn[N], low[N], bel[N], idx, scc, in[N];
stack<int> stk;

void init() {
	vector <int> vis(2e6, 0);
	for (int i = 2; i <= 2e6; i++) {
		if (!vis[i]) prime.emplace_back(i), lsh[i] = ++tot;
		for (auto j : prime) if (i * j <= 2e6) {
			vis[i * j] = 1;
			if (i % j == 0) break;
		} else break;
	}
}

void work(int x, int id) {
	for (auto i : prime) {
		if (i * i > x) break;
		if (x % i != 0) continue;
		vec[lsh[i]].emplace_back(id);
		while (x % i == 0) x /= i;
	}
	if (x != 1) vec[lsh[x]].emplace_back(id);
}

#define C(x) ((x) > n ? (x) - n : (x) + n)

void link(int x, int y) { e[x].emplace_back(y); }

void connect(int x) {
	if (vec[x].empty()) return ;
	link(++cnt, C(vec[x][0]));
	for (int i = 1; i < vec[x].size(); i++) {
		cnt++;
		link(vec[x][i], cnt - 1);
		link(cnt, cnt - 1);
		link(cnt, C(vec[x][i]));
	}
	link(++cnt, C(vec[x].back()));
	for (int i = (int) vec[x].size() - 2; ~i; i--) {
		cnt++;
		link(vec[x][i], cnt - 1);
		link(cnt, cnt - 1);
		link(cnt, C(vec[x][i]));
	}
}

void tarjan(int u) {
	dfn[u] = low[u] = ++idx, in[u] = 1;
	stk.emplace(u);
	for (auto v : e[u]) {
		if (!dfn[v]) {
			tarjan(v);
			low[u] = min(low[u], low[v]);
		} else if (in[v]) {
			low[u] = min(low[u], dfn[v]);
		}
	}
	if (dfn[u] == low[u]) {
		int x;
		scc++;
		do {
			x = stk.top(), stk.pop();
			bel[x] = scc, in[x] = 0;
		} while (x != u);
	}
}

signed main() {

#ifdef LOCAL
	freopen("test.in", "r", stdin);
	freopen("test.out", "w", stdout);
#endif

	init();

	read(n);
	for (int i = 1, x, y; i <= n; i++) {
		read(x, y);
		work(x, i), work(y, i + n);
	}

	cnt = 2 * n;
	for (int i = 1; i <= tot; i++) connect(i);

	for (int i = 1; i <= cnt; i++) if (!dfn[i]) tarjan(i);

	for (int i = 1; i <= n; i++) {
		if (bel[i] == bel[i + n]) {
			cout << "No\n";
			return  0;
		}
	}

	cout << "Yes\n";

	return 0;
}





#include <map>
#include <set>
#include <queue>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
using namespace std;
#define inf 1e18
#define int long long
#define mp make_pair
const int mod = 1e9 + 7;
int read () {
	int x = 0, f = 1;
	char ch = getchar ();
	while (ch < '0' || ch > '9') {
		if (ch == '-') f = -1;
		ch = getchar ();
	}
	while (ch >= '0' && ch <= '9') {
		x = x * 10 + ch - '0';
		ch = getchar ();
	}
	return x * f;
}
void write (int x) {
	if (x < 0) x = -x, putchar ('-');
	if (x >= 10) write (x / 10);
	putchar (x % 10 + '0');
}
int quickmod (int x, int y) {
	int Ans = 1;
	while (y) {
		if (y & 1) Ans = (Ans * x) % mod;
		x = (x * x) % mod;
		y >>= 1;
	}
	return Ans;
}
inline void Mx (int &x, int y) {
	if (y > x) x = y;
}
int n, m;
int a[200005], b[200005];
int pa[200005], pb[200005]; 
int t[200005], pre[200005];
int whi[200005];
int pa1, pb1; 
int get_ans (int i, int j, int fl) {
	int fir = n * m - pre[whi[i]] - i * (m - whi[i]);
	return fir + (i - n) * j + pre[j] - (fl * (pa1 > i && pb1 > j));
}
/*
<= mid : 0
> mid : 1
*/
int check (int mid) {
	int j = n;
	for (int i = 1; i <= m; i++) {
		while (j >= 1 && pa[j] + pb[i] > mid) j--;
		t[i] = j;
	}
	for (int i = 1; i <= m; i++) pre[i] = pre[i-1] + t[i];
	j = 0;
	for (int i = n; i >= 0; i--) {
		while (t[j+1] >= i && j < m) j++;
		whi[i] = j;
	}
	int ans1 = -inf, ans2 = -inf;
	j = 0;
	for (int i = 0; i <= n; i++) {
		while (j < m && get_ans (i, j, 0) <= get_ans (i, j + 1, 0)) j++;
		Mx (ans1, get_ans (i, j, 0));
	}
	j = 0;
	for (int i = 0; i <= n; i++) {
		while (j < m && get_ans (i, j, 1) <= get_ans (i, j + 1, 1)) j++;
		Mx (ans2, get_ans (i, j, 1));
	}
	return ans1 == ans2;
}
signed main () {
// 	freopen ("chess.in", "r", stdin);
// 	freopen ("chess.out", "w", stdout);
	n = read (), m = read ();
	for (int i = 1; i <= n; i++) a[i] = read (), pa[i] = a[i];
	for (int i = 1; i <= m; i++) b[i] = read (), pb[i] = b[i]; 
	sort (pa + 1, pa + 1 + n), sort (pb + 1, pb + 1 + m);
	pa1 = lower_bound (pa + 1, pa + 1 + n, a[1]) - pa;
	pb1 = lower_bound (pb + 1, pb + 1 + m, b[1]) - pb;
	int l = 0, r = a[1] + b[1] - 1, ans = a[1] + b[1]; 
	while (l <= r) {
		int mid = (l + r) / 2;
		if (check (mid)) r = mid - 1, ans = mid;
		else l = mid + 1; 
	}
	write (ans), putchar ('\n');
	return 0;
}
/*
*/
#include <bits/stdc++.h>
typedef long long LL;
typedef std::pair<int, int> pii;
#define MP std::make_pair
#define fi first
#define se second
#define cpy(f, g) memcpy(f, g, sizeof(f))
int read()
{
	int s = 0, f = 1;
	char c = getchar();
	while (!(c >= '0' && c <= '9'))
		f ^= c == '-', c = getchar();
	while (c >= '0' && c <= '9')
		s = s * 10 + (c ^ 48), c = getchar();
	return f ? s : -s;
}
const int MAXN = 3005, MOD = 1e9 + 7;
auto fplus = [](int x, int y){ return x + y >= MOD ? x + y - MOD : x + y; };
auto fminus = [](int x, int y){ return x >= y ? x - y : x + MOD - y; };
auto Fplus = [](int &x, int y){ return x = fplus(x, y); };
auto Fminus = [](int &x, int y){ return x = fminus(x, y); };
int fpow(int x, int y = MOD - 2)
{
	int res = 1;
	for (; y; y >>= 1, x = (LL)x * x % MOD)
		if (y & 1) res = (LL)res * x % MOD;
	return res;
}
int n, pos[MAXN], a[MAXN];
int f[MAXN][MAXN][2], g[MAXN], h[MAXN], tmp[MAXN];
bool vis[MAXN];
void clr(int n)
{
	for (int i = 0; i <= n; i++)
		for (int j = 0; j <= n; j++)
			f[i][j][0] = f[i][j][1] = 0;
}
void DP(int n)
{
	memset(g, 0, (n + 1) << 2);
	if (n == 1) return g[0] = g[1] = 1, void();
	//1-->not chosen
	clr(n);
	f[1][0][0] = 1;
	for (int i = 2; i <= n; i++)
		for (int j = 0; j < i; j++)
		{
			Fplus(f[i][j][0], fplus(f[i - 1][j][0], f[i - 1][j][1]));//not chosen
			Fplus(f[i][j + 1][0], f[i - 1][j][0]);//choose i
			Fplus(f[i][j + 1][1], fplus(f[i - 1][j][0], f[i - 1][j][1]));//choose a[i]
		}
	for (int i = 0; i <= n; i++)
		Fplus(g[i], fplus(f[n][i][0], f[n][i][1]));
	//1-->1
	clr(n);
	f[1][1][0] = 1;
	for (int i = 2; i <= n; i++)
		for (int j = 0; j < i; j++)
		{
			Fplus(f[i][j][0], fplus(f[i - 1][j][0], f[i - 1][j][1]));//not chosen
			Fplus(f[i][j + 1][0], f[i - 1][j][0]);//choose i
			if (i < n) Fplus(f[i][j + 1][1], fplus(f[i - 1][j][0], f[i - 1][j][1]));//choose a[i]
		}
	for (int i = 0; i <= n; i++) Fplus(g[i], f[n][i][0]);
	//1-->not chosen
	clr(n);
	f[1][1][1] = 1;
	for (int i = 2; i <= n; i++)
		for (int j = 0; j < i; j++)
		{
			Fplus(f[i][j][0], fplus(f[i - 1][j][0], f[i - 1][j][1]));//not chosen
			Fplus(f[i][j + 1][0], f[i - 1][j][0]);//choose i
			Fplus(f[i][j + 1][1], fplus(f[i - 1][j][0], f[i - 1][j][1]));//choose a[i]
		}
	for (int i = 0; i <= n; i++)
		Fplus(g[i], fplus(f[n][i][0], f[n][i][1]));
}
int main()
{
	n = read();
	for (int i = 1; i <= n; i++) pos[read()] = i;
	for (int i = 1; i <= n; i++) a[i] = pos[read()];
	h[0] = 1;
	for (int x = 1; x <= n; x++)
	{
		if (vis[x]) continue;
		int len = 0;
		for (int y = x; !vis[y]; y = a[y])
			len++, vis[y] = true;
		DP(len);
		memcpy(tmp, h, (n + 1) << 2);
		memset(h, 0, (n + 1) << 2);
		for (int i = 0; i <= n; i++)
			for (int j = 0; j <= len && i + j <= n; j++)
				Fplus(h[i + j], (LL)tmp[i] * g[j] % MOD);
	}
	int ans = 0; 
	for (int i = n, fac = 1; i >= 0; fac = (LL)fac * (n - (--i)) % MOD)
		((i & 1) ? Fminus : Fplus)(ans, (LL)h[i] * fac % MOD);
	printf("%d\n", ans);
	return 0;
}
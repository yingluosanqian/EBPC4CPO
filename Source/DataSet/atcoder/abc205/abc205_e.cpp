#include <bits/stdc++.h>
using namespace std; typedef long long int ll; typedef long double ld; const ll inf = 1e18, mod = 1e9 + 7, maxn = 2e6;

int main() {
	ios::sync_with_stdio(0); cin.tie(0); cout.tie(0); cout << fixed << setprecision(20);
	vector<ll> f(maxn + 10, 1), finv(maxn + 10, 1), inv(maxn + 10, 1);
	for (ll i = 2; i <= maxn + 5; ++i) {
		f[i] = f[i - 1] * i % mod;
		inv[i] = (mod - (mod / i) * inv[mod % i] % mod) % mod;
		finv[i] = finv[i - 1] * inv[i] % mod;
	}
	auto C = [&](ll n, ll r) {
		return n < 0 or r < 0 or n < r ? 0ll : f[n] * finv[r] % mod * finv[n - r] % mod;
	};
	ll n, m, k; cin >> n >> m >> k;
	if (n > m + k) return cout << 0 << "\n", 0;
	cout << (mod + C(n + m, n) - C(n + m, m + k + 1)) % mod << "\n";
	return 0;
}
#include <bits/stdc++.h>
using namespace std;

template<int mod> class modint {
private:
	int x;
public:
	modint() : x(0) {}
	modint(long long x_) : x(x_ >= 0 ? x_ % mod : (x_ + 1) % mod + (mod - 1)) {}
	int val() const { return x; }
	modint operator-() const { return modint(-x); }
	modint operator+=(modint m) { x = (x + m.x) % mod; return *this; }
	modint operator-=(modint m) { x = (x - m.x + mod) % mod; return *this; }
	modint operator*=(modint m) { x = 1LL * x * m.x % mod; return *this; }
	modint operator+(modint m) const { return modint(*this) += m; }
	modint operator-(modint m) const { return modint(*this) -= m; }
	modint operator*(modint m) const { return modint(*this) *= m; }
	modint pow(long long b) const {
		modint res(1), a(*this);
		while (b >= 1) {
			if (b % 2 == 1) {
				res *= a;
			}
			a *= a;
			b /= 2;
		}
		return res;
	}
	modint inv() const {
		return pow(mod - 2);
	}
};

using mint = modint<998244353>;

vector<vector<mint> > matrix_mul(int n, vector<vector<mint> > a, vector<vector<mint> > b) {
	vector<vector<mint> > c(n, vector<mint>(n));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			for (int k = 0; k < n; k++) {
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}
	return c;
}

vector<vector<mint> > matrix_pow(int n, vector<vector<mint> > a, long long b) {
	vector<vector<mint> > res(n, vector<mint>(n));
	for (int i = 0; i < n; i++) {
		res[i][i] = 1;
	}
	while (b >= 1) {
		if (b % 2 == 1) {
			res = matrix_mul(n, res, a);
		}
		a = matrix_mul(n, a, a);
		b /= 2;
	}
	return res;
}

int num_patterns(int L, int filled) {
	vector<int> dp(L + 1);
	dp[0] = 1;
	for (int i = 1; i <= L; i++) {
		dp[i] = dp[i - 1];
		if (i >= 2 && (filled & (3 << (i - 2))) == 0) {
			dp[i] += dp[i - 2];
		}
	}
	return dp[L];
}

vector<vector<mint> > make_matrix(int H) {
	vector<vector<mint> > res(1 << H, vector<mint>(1 << H));
	for (int i = 0; i < (1 << H); i++) {
		for (int j = 0; j < (1 << H); j++) {
			if ((i & j) == 0) {
				int num = num_patterns(H, i + j);
				res[i][j] += num;
			}
		}
	}
	return res;
}

mint solve(int H, long long W) {
	vector<vector<mint> > mat = make_matrix(H);
	vector<vector<mint> > res = matrix_pow(1 << H, mat, W);
	return res[0][0];
}

int main() {
	int H; long long W;
	cin >> H >> W;
	mint ans = solve(H, W);
	cout << ans.val() << endl;
	return 0;
}
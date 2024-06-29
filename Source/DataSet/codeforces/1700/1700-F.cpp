// BEGIN BOILERPLATE CODE

#include <bits/stdc++.h>
using namespace std;

#ifdef KAMIRULEZ
	const bool kami_loc = true;
	const long long kami_seed = 69420;
#else
	const bool kami_loc = false;
	const long long kami_seed = chrono::steady_clock::now().time_since_epoch().count();
#endif

const string kami_fi = "kamirulez.inp";
const string kami_fo = "kamirulez.out";
mt19937_64 kami_gen(kami_seed);

long long rand_range(long long rmin, long long rmax) {
	uniform_int_distribution<long long> rdist(rmin, rmax);
	return rdist(kami_gen);
}

void file_io(string fi, string fo) {
	if (fi != "" && (!kami_loc || fi == kami_fi)) {
		freopen(fi.c_str(), "r", stdin);
	}
	if (fo != "" && (!kami_loc || fo == kami_fo)) {
		freopen(fo.c_str(), "w", stdout);
	}
}

void set_up() {
	if (kami_loc) {
		file_io(kami_fi, kami_fo);
	}
	ios_base::sync_with_stdio(0);
	cin.tie(0);
}

void just_do_it();

void just_exec_it() {
	if (kami_loc) {
		auto pstart = chrono::steady_clock::now();
		just_do_it();
		auto pend = chrono::steady_clock::now();
		long long ptime = chrono::duration_cast<chrono::milliseconds>(pend - pstart).count();
		string bar(50, '=');
		cout << '\n' << bar << '\n';
		cout << "Time: " << ptime << " ms" << '\n';
	}
	else {
		just_do_it();
	}
}

int main() {
	set_up();
	just_exec_it();
	return 0;
}

// END BOILERPLATE CODE

// BEGIN MAIN CODE

const int ms = 2e5 + 20;
const long long inf = (long long)1e18 + 20;
int a[2][ms];
int b[2][ms];

void just_do_it() {
	int n;
	cin >> n;
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < n; j++) {
			cin >> a[i][j];
		}
	}
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < n; j++) {
			cin >> b[i][j];
		}
	}
	long long res = 0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < 2; j++) {
			if (a[j][i] > 0 && b[j][i] > 0) {
				int d = min(a[j][i], b[j][i]);
				a[j][i] -= d;
				b[j][i] -= d;
			}
		}
		for (int j = 0; j < 2; j++) {
			if (a[0 ^ j][i] > 0 && b[1 ^ j][i] > 0) {
				int d = min(a[0 ^ j][i], b[1 ^ j][i]);
				a[0 ^ j][i] -= d;
				b[1 ^ j][i] -= d;
				res += d;
			}
		}
		res += a[0][i];
		a[0][i + 1] += a[0][i];
		res += a[1][i];
		a[1][i + 1] += a[1][i];
		res += b[0][i];
		b[0][i + 1] += b[0][i];
		res += b[1][i];
		b[1][i + 1] += b[1][i];
	}
	if (a[0][n] + a[1][n] + b[0][n] + b[1][n] == 0) {
		cout << res;
	}
	else {
		cout << -1;
	}
}

// END MAIN CODE
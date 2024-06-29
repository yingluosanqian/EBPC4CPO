#include <iostream>
#include <vector>
#include <algorithm>
#include<stdio.h>
#include<math.h>
#include<iomanip>
#include<string>
#include<utility>
#include<tuple>
#include<set>
#include<map>
#include<stack>
#include<queue>
#include <array>
#include <unordered_set>
using namespace std;

constexpr int INF = 1 << 30;
const double pi = 3.141592653589793;
typedef long long ll;
// std::cout << std::setprecision(20) << std::fixed << ans << std::endl;

// 最大公約数
ll gcd(ll a, ll b) {
	while (a > 0 && b > 0) {
		if (a > b) a = a % b;
		else b = b % a;
	}
	return max(a, b);
}

// 最小公倍数
ll lcm(ll a, ll b) {
	return a / gcd(a, b) * b;
}

// 桁数を返す
ll digit(ll num) {
	ll digit_num = 0;
	while (num > 0) {
		digit_num += 1;
		num /= 10;
	}
	return digit_num;
}

// 素数どうか判定 素数ならtrueを返す
bool isPrime(int x) {
	for (int i = 2; i * i <= x; i++) {
		if (x % i == 0) return false;
	}
	return true;
}

// 回文ならtrueを返す
bool isPalindrome(string s) {
	int n = s.size();
	for (int i = 0; i < n; i++) {
		if (s[i] != s[n - i - 1]) return false;
	}
	return true;
}

// 順列全探索
//string S;
//int K;
//cin >> S >> K;
//vector<string> cands;
//sort(S.begin(), S.end());
//do {
//	cands.push_back(S);
//} while (next_permutation(S.begin(), S.end()));
//cout << cands[K - 1] << endl;

// 素因数分解
vector<pair<ll, ll>> prime_factorize(ll N) {
	vector<pair<ll, ll > > res;
	for (ll a = 2; a * a <= N; ++a) {
		if (N % a != 0) continue;
		ll ex = 0;

		while (N % a == 0) {
			++ex;
			N /= a;
		}

		res.push_back({ a, ex });
	}

	if (N != 1) res.push_back({ N, 1 });
	return res;
}

int main() {
	int N; string S; cin >> N >> S;
	if (S[N - 1] == 'o') cout << "Yes" << endl;
	else cout << "No" << endl;
}
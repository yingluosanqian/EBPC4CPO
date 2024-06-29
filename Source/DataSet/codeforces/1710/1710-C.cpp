#include <bits/stdc++.h>

#define fr(i, a, b) for (int i = (a); i <= (b); i++)
#define rf(i, a, b) for (int i = (a); i >= (b); i--)
#define fe(x, y) for (auto& x : y)

#define fi first
#define se second
#define pb push_back
#define mp make_pair
#define mt make_tuple

#define pw(x) (1LL << (x))
#define sz(x) (int)(x).size()
#define all(x) (x).begin(), (x).end()

using namespace std;

mt19937_64 rng(chrono::system_clock::now().time_since_epoch().count());

#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
template<typename T>
using oset = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;
#define fbo find_by_order
#define ook order_of_key

template<typename T>
bool umn(T& a, T b) { return a > b ? a = b, 1 : 0; }
template<typename T>
bool umx(T& a, T b) { return a < b ? a = b, 1 : 0; }

using ll = long long;
using ld = long double;
using pii = pair<int, int>;
using pll = pair<ll, ll>;
template<typename T>
using ve = vector<T>;

const int MOD = 998244353;

int add(int a, int b) { return (a += b) >= MOD ? a - MOD : a; }
int sub(int a, int b) { return (a -= b) < 0 ? a + MOD : a; }
int mult(int a, int b) { return 1LL * a * b % MOD; }
void addSelf(int& a, int b) { (a += b) >= MOD ? a -= MOD : 0; }
void subSelf(int& a, int b) { (a -= b) < 0 ? a += MOD : 0; }
void multSelf(int& a, int b) { a = 1LL * a * b % MOD; }

int bpow(int a, int b) {
    int res = 1;
    while(b) {
        if(b & 1) {
            res = mult(res, a);
        }
        a = mult(a, a);
        b >>= 1;
    }
    return res;
}

string s;
int dp[2][2][2][8];
int ndp[2][2][2][8];
int ans;

int main() {
#ifndef LOCAL
    // freopen("input.txt", "r", stdin);
    // freopen("output.txt", "w", stdout);
    ios::sync_with_stdio(0);
    cin.tie(0);
#else
    // freopen("input.txt", "r", stdin);
    // freopen("output.txt", "w", stdout);
#endif

    cin >> s;

    fr (a, 0, 1) {
        fr (b, 0, 1) {
            fr (c, 0, 1) {
                int x = a ^ b;
                int y = a ^ c;
                int z = b ^ c;
                int f1 = 0;
                int f2 = 0;
                int f3 = 0;
                if (x && y) f1 = 1;
                if (x && z) f2 = 1;
                if (y && z) f3 = 1;
                int mask = 0;
                if (!a) mask |= pw(0);
                if (!b) mask |= pw(1);
                if (!c) mask |= pw(2);
                dp[f1][f2][f3][mask]++;
            }
        }
    }

    fr (i, 0, sz(s) - 2) {
        fr (f1, 0, 1) {
            fr (f2, 0, 1) {
                fr (f3, 0, 1) {
                    fr (mask, 0, 7) {
                        if (!dp[f1][f2][f3][mask]) {
                            continue;
                        }

                        fr (a, 0, 1) {
                            fr (b, 0, 1) {
                                fr (c, 0, 1) {
                                    if (!(mask & pw(0)) && a > s[i + 1] - '0') {
                                        continue;
                                    }
                                    if (!(mask & pw(1)) && b > s[i + 1] - '0') {
                                        continue;
                                    }
                                    if (!(mask & pw(2)) && c > s[i + 1] - '0') {
                                        continue;
                                    }
                                    int nmask = mask;
                                    if (a < s[i + 1] - '0') {
                                        nmask |= pw(0);
                                    }
                                    if (b < s[i + 1] - '0') {
                                        nmask |= pw(1);
                                    }
                                    if (c < s[i + 1] - '0') {
                                        nmask |= pw(2);
                                    }
                                    int x = a ^ b;
                                    int y = a ^ c;
                                    int z = b ^ c;
                                    int nf1 = f1;
                                    int nf2 = f2;
                                    int nf3 = f3;

                                    if (x && y) nf1 = 1;
                                    if (x && z) nf2 = 1;
                                    if (y && z) nf3 = 1;

                                    addSelf(ndp[nf1][nf2][nf3][nmask], dp[f1][f2][f3][mask]);
                                }
                            }
                        }
                    }
                }
            }
        }
        fr (f1, 0, 1) {
            fr (f2, 0, 1) {
                fr (f3, 0, 1) {
                    fr (mask, 0, 7) {
                        dp[f1][f2][f3][mask] = ndp[f1][f2][f3][mask];
                        ndp[f1][f2][f3][mask] = 0;
                    }
                }
            }
        }
    }

    fr (mask, 0, 7) {
        addSelf(ans, dp[1][1][1][mask]);
    }

    cout << ans << "\n";

    return 0;
}
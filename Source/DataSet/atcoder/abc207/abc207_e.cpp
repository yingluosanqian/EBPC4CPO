#include <bits/stdc++.h>
using namespace std;
using ll = long long;
#define ALL(v) v.begin(),v.end()
#define dbg(x) cerr << #x << ": " << (x) << endl;
template<class Iter>
void print(Iter beg, Iter end) {
    for (Iter itr = beg; itr != end; ++itr) {
        cerr << *itr << ' ';
    }
    cerr << '\n';
}
int N;
vector<ll> A;
ll MOD = 1e9 + 7;

int main() {
    ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0);
    cin >> N;
    A.resize(N+1);
    for (int i = 1; i <= N; ++i) {
        cin >> A[i];
        A[i] += A[i-1];
    }
    // [i¤Ç¸î¤Ã¤¿¤é][Óà¤ê¤¬j]
    vector sum(N+2, vector<ll>(N+1));
    // [i·¬Ä¿¤Þ¤Ç][j·Ö¸î]
    vector dp(N+1, vector<ll>(N+1));
    sum[1][0] = 1;

    for (int i = 1; i <= N; ++i) {
        for (int j = 1; j <= i; ++j) {
            dp[i][j] = sum[j][A[i] % j];
        }
        for (int j = 1; j <= i; ++j) {
            sum[j + 1][A[i] % (j + 1)] += dp[i][j];
            sum[j + 1][A[i] % (j + 1)] %= MOD;
        }
    }
    // debug
    // for (int i = 1; i <= N; ++i) {
    //     for (int j = 1; j <= i; ++j) {
    //         cerr << dp[i][j] << ' ';
    //     }
    //     cerr << '\n';
    // }

    ll ans = 0;
    for (int i = 1; i <= N; ++i) {
        ans += dp[N][i];
        ans %= MOD;
    }
    cout << ans << '\n';
}
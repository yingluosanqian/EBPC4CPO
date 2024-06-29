#include <bits/stdc++.h>
#include <iostream>
using namespace std;
#define mod 998244353
#define inf 0x3f3f3f3f
#define rep(i, a, b) for (int i = a; i <= b; ++i)
#define drep(i, a, b) for (int i = a; i >= b; --i)
#define IO ios::sync_with_stdio(0), cin.tie(0), cout.tie(0)
#define endl '\n'
#define Endl '\n'
#define pb push_back
#define fi first
#define se second
//#define ll long long
typedef long long ll;
typedef unsigned long long ull;
typedef pair<int, int> pii;
#define VI vector<Z>
int norm(int x)
{
    if (x < 0)
    {
        x += mod;
    }
    if (x >= mod)
    {
        x -= mod;
    }
    return x;
}
ll FZqpow(ll a, ll b, ll c)
{
    ll res = 1;
    for (; b; b /= 2, (a *= a) %= c)
    {
        if (b % 2)
        {
            (res *= a) %= c;
        }
    }
    return res;
}
template <class T>
T qpow(T a, long long b)
{
    T res = 1;
    for (; b; b /= 2, a *= a)
    {
        if (b % 2)
        {
            res *= a;
        }
    }
    return res;
}
struct Z
{
    int x;
    Z(int x = 0) : x(norm(x)) {}
    int val() const
    {
        return x;
    }
    Z operator-() const
    {
        return Z(norm(mod - x));
    }
    Z inv() const
    {
        assert(x != 0);
        return qpow(*this, mod - 2);
    }
    Z& operator*=(const Z& rhs)
    {
        x = ll(x) * rhs.x % mod;
        return *this;
    }
    Z& operator+=(const Z& rhs)
    {
        x = norm(x + rhs.x);
        return *this;
    }
    Z& operator-=(const Z& rhs)
    {
        x = norm(x - rhs.x);
        return *this;
    }
    Z& operator/=(const Z& rhs)
    {
        return *this *= rhs.inv();
    }
    friend Z operator*(const Z& lhs, const Z& rhs)
    {
        Z res = lhs;
        res *= rhs;
        return res;
    }
    friend Z operator+(const Z& lhs, const Z& rhs)
    {
        Z res = lhs;
        res += rhs;
        return res;
    }
    friend Z operator-(const Z& lhs, const Z& rhs)
    {
        Z res = lhs;
        res -= rhs;
        return res;
    }
    friend Z operator/(const Z& lhs, const Z& rhs)
    {
        Z res = lhs;
        res /= rhs;
        return res;
    }
    friend std::istream& operator>>(std::istream& is, Z& a)
    {
        ll v;
        is >> v;
        a = Z(v);
        return is;
    }
    friend std::ostream& operator<<(std::ostream& os, const Z& a)
    {
        return os << a.val();
    }
};

const int maxn = 500 + 5;
struct HHASH
{
    ull h1[maxn], h2[maxn], base, p[maxn] = { 1 };
    string str;
    HHASH(ull x = 0) : base(norm(x)) {}
    bool jud(int l, int r) {
        int x = r - (r - l + 1) / 2, y = l + (r - l + 1) / 2;
        int a = h1[x] - h1[l - 1] * p[x - l + 1];
        int b = h2[y] - h2[r + 1] * p[r - y + 1];
        return a == b;
    }
    void init(int n) {
        for (int i = 1, j = n; i <= n; ++i, j--) {
            h1[i] = h1[i - 1] * base + str[i];
            h2[j] = h2[j + 1] * base + str[j];
            //cout << 111 << " " << h1[i] << " " << h2[j] << endl;
            p[i] = p[i - 1] * base;
        }
    }
}one(13331), two(1000000007);
int a[maxn], dp[2][maxn][maxn];
void Asoul_Diana()
{
    int n;
    cin >> n;
    rep(i, 1, n)cin >> a[i];
    memset(dp, -1, sizeof dp);
    dp[0][0][0] = 0;//
    for (int i = 1; i <= n; ++i) {
        memset(dp[i % 2], -1, sizeof dp[i % 2]);
        for (int j = 0; j < i; ++j) {//È¡j+1ÂÖ
            for (int k = max(0,i - (j+1)-1); k <= j; ++k) {
                if (dp[(i + 1) % 2][j][k] == -1)continue;
                //cout << i << " " << j << ' ' << k << endl;
                dp[i % 2][j+1][k - (i - (j + 1) - 1)] = max(dp[i % 2][j+1][k - (i - (j + 1) - 1)], dp[(i + 1) % 2][j][k] + a[i]);
            }
        }
        for (int j = 1; j <= n; ++j) {
            for (int k = 0; k <= j; ++k) {
                if (dp[i % 2][j - 1][k] == -1)continue;
                //cout << i << " " << j << ' ' << k << endl;
                dp[i % 2][j][k] = max(dp[i % 2][j][k], dp[i % 2][j - 1][k] + a[i]);
            }
        }
        for (int j = 0; j <= n; ++j) {
            for (int k = 0; k <= j; ++k) {
                dp[i % 2][j][k] = max(dp[i % 2][j][k], dp[(i + 1) % 2][j][k]);
                //cout << i << " " << j << " " << k << " " << dp[i % 2][j][k] << endl;
            }
        }
    }
    int ans = -1;
    rep(i, 0, 1) {
        rep(j, 0, n) {
            ans = max(ans, dp[i][n][j]);
        }
    }
    cout << ans << endl;
}
int main()
{
    int _tt = 1;
    //cin >> _tt;
    while (_tt--)
    {
        Asoul_Diana();
    }
    return 0;
}
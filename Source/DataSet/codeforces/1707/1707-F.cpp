#include <bits/stdc++.h>
 
using namespace std;
 
#define mp make_pair
#define pb emplace_back
#define rep(i, s, e) for (int i = s; i <= e; ++i)
#define drep(i, s, e) for (int i = s; i >= e; --i)
#define file(a) freopen(#a".in", "r", stdin), freopen(#a".out", "w", stdout)
#define pv(a) cout << #a << " = " << a << endl
#define pa(a, l, r) cout << #a " : "; rep(_, l, r) cout << a[_] << ' '; cout << endl
 
const int N = 1e7 + 10;
 
int read() {
    int x = 0, f = 1; char c = getchar();
    for (; c < '0' || c > '9'; c = getchar()) if (c == '-') f = -1;
    for (; c >= '0' && c <= '9'; c = getchar()) x = x * 10 + c - 48;
    return x * f;
}
 
int qpow(int a, int b, int p) {
    int res = 1;
    for (; b; b >>= 1, a = 1ll * a * a % p) {
        if (b & 1) res = 1ll * res * a % p;
    }
    return res;
}
 
int n, m, lim, o, t, w, s[N], c[N], cnt[N], c0, cur, rev[N];
 
struct node {
    int typ, pos, z;
    node(int a = 1, int b = -1, int c = 0) { typ = a, pos = b, z = c; }
    friend node operator + (node a, node b) { // 点乘
        if (a.typ > b.typ) swap(a, b);
        if (a.typ == 0) {
            if (b.typ == 0) return node(0, -1, a.z + b.z);
            else if (~b.pos) return node(1, b.pos, a.z + b.z);
            else return node(1, -1, 0);
        }
        else {
            if (~a.pos && a.pos == b.pos) return node(1, a.pos, a.z + b.z);
            else return node(1, -1, 0);
        }
    }
    friend node operator * (node a, node b) { // xor 卷积
        if (a.typ > b.typ) swap(a, b);
        if (a.typ == 0) {
            if (b.typ == 0) return node(0, -1, a.z + b.z + 1);
            else if (~b.pos) return node(0, -1, a.z + b.z);
            else return node(1, -1, 0);
        }
        else {
            if (~a.pos && ~b.pos) return node(1, a.pos ^ b.pos, a.z + b.z);
            return node(1, -1, 0);
        }
    }
    int val(int mod = 998244353) {
        if (typ == 0) return qpow(2, w * (z + 1), mod);
        else if (~pos) return qpow(2, w * z, mod);
        else return 0;
    }
} mask[30], dat[N];
 
#define ls lson[u]
#define rs rson[u]
#define mid (l + r >> 1)
 
int rt, tot, lson[N], rson[N];
 
int newnode(int l, int r) {
    dat[++ tot] = mask[__lg(r - l + 1)];
    return tot;
}
 
void maintain(int u, int l, int r) {
    if (t & (1 << o - __lg(r - l + 1))) dat[u] = dat[ls] + dat[rs];//奇数点乘
    else dat[u] = dat[ls] * dat[rs];
}
 
int modify(int p, int x, int u, int l, int r) {
    if (!u) u = newnode(l, r);
    if (l == r) {
        dat[u] = ~x ? node(1, x, 0) : node(0, -1, 0);
        return u;
    }
    if (p <= mid) {
        ls = modify(p, x, ls, l, mid);
        if (!rs) rs = newnode(mid + 1, r);
    }
    else {
        rs = modify(p, x, rs, mid + 1, r);
        if (!ls) ls = newnode(l, mid);
    }
    return maintain(u, l, r), u;
}
 
void Modify(int p, int x) {
    int lst = c[p]; c[p] = x;
    p %= lim, cur -= cnt[p] == 0 && s[p] > 0;
    if (~lst) s[p] ^= lst;
    if (~x) {
        s[p] ^= x;
        if (~lst || --cnt[p] == 0) rt = modify(rev[p], s[p], rt, 0, lim - 1);
        else --c0;
    }
    else {
        if (!~lst || ++cnt[p] == 1) rt = modify(rev[p], -1, rt, 0, lim - 1);
        else ++ c0;
    }
    cur += cnt[p] == 0 && s[p] > 0;//非0位置
}
 
int main() {
    n = read(), m = read(), t = read(), w = read();
    lim = n & -n, o = __lg(lim);
    rep(i, 0, lim - 1) {
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << o - 1);
    }
    rep(i, 0, n - 1) {
        c[i] = -1;
        if (++ cnt[i % lim] != 1) ++c0;
    }
    mask[0] = node(0, -1, 0);
    rep(i, 1, o) {
        if (t & (1 << o - i)) mask[i] = mask[i - 1] + mask[i - 1];
        else mask[i] = mask[i - 1] * mask[i - 1];
    }
    rt = newnode(0, lim - 1);
    rep(i, 1, m) {
        int p = read() - 1, x = read();
        Modify(p, x);
    }
    for (int q = read(); q; -- q) {
        int p = read() - 1, x = read(), mod = read();
        Modify(p, x);
        int res = 1ll * (t < lim ? dat[rt].val(mod) : !cur) * qpow((1 << w) % mod, c0, mod) % mod;
        printf("%d\n", res);
    }
    return 0;
}
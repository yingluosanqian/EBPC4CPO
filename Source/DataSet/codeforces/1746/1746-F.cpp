#include <iostream>
#include <string>
#include <sstream>
#include <iomanip> 
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <queue>
#include <stack>
#include <vector>
#include <map>
#include <set>
#include <functional>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <bitset>
#include <complex>

using namespace std;

typedef long long LL;
typedef pair<LL, LL> PL;
typedef vector<LL> VL;
typedef vector<PL> VPL;
typedef vector<VL> VVL;

typedef pair<int, int> PI;
typedef vector<int> VI;
typedef vector<PI> VPI;
typedef vector<vector<int>> VVI;
typedef vector<vector<PI>> VVPI;

typedef long double LD;
typedef pair<LD, LD> PLDLD;

typedef complex<double> CD;
typedef vector<CD> VCD;

typedef vector<string> VS;

#define MP make_pair
#define PB push_back
#define F first
#define S second
#define LB lower_bound
#define UB upper_bound

#define SZ(x) ((int)x.size())
#define LEN(x) ((int)x.length())
#define ALL(x) begin(x), end(x)
#define RSZ resize
#define ASS assign
#define REV(x) reverse(x.begin(), x.end());



#define FOR(i, a, b) for (int i = (a); i < (b); i++)
#define F0R(i, a) for (int i = 0; i < (a); i++)
#define FORd(i,a,b) for (int i = (b)-1; i >= (a); i--)
#define F0Rd(i,a) for (int i = (a)-1; i >= 0; i--)
#define trav(a, x) for (auto& a : x)


const LL INF = 1E18;
const int MAXX = 300005;
const LD PAI = 4 * atan((LD)1);

template <typename T>
class fenwick_tree {
public:
    vector<T> fenw;
    int n;

    fenwick_tree(int _n) : n(_n) {
        fenw.resize(n);
    }

    void update(int x, T v) {
        while (x < n) {
            fenw[x] += v;
            x |= (x + 1);
            //x += (x & (-x));
        }
    }

    T query(int x) {
        T v{};
        while (x >= 0) {
            v += fenw[x];
            x = (x & (x + 1)) - 1;
        }
        return v;
    }

    T query_full(int a, int b) {		// range query
        return query(b) - ((a <= 1) ? 0 : query(a - 1));
    }
};

template <typename T>
vector<T> serialize(vector<T> a, int startvalue = 0) {
    int n = a.size(), i, j, k, ct;
    vector<T> ans(n);
    map<T, T> id;
    for (auto p : a) id[p] = 0;
    ct = startvalue;
    for (auto p : id) id[p.first] = ct++;
    for (i = 0; i < n; i++) ans[i] = id[a[i]];
    return ans;
}

template <typename T>
class segment_tree {
    vector<T> t;
    T VERYBIG;
    bool ISMAXRANGE;
    int size;
public:
    segment_tree(int n, bool range_max = true) {
        if (is_same<T, int>::value) VERYBIG = (1 << 30);
        else if (is_same<T, LL>::value) VERYBIG = (1LL << 60);
        //else if (is_same<T, PII>::value) VERYBIG = PII({ 1E9, 1E9 });
        //else if (is_same<T, PLL>::value) VERYBIG = { 1LL << 60, 1LL << 60 };

        ISMAXRANGE = range_max;

        if (ISMAXRANGE) t.assign(4 * n + 1, 0);
        else t.assign(4 * n + 1, VERYBIG);
        size = n;
    }

    void initialize_array(vector<T>& v) {
        initialize_with_array(1, 0, size - 1, v);
    }

    void initialize_with_array(int startpos, int l, int r, vector<T>& v) {
        if (l == r) {
            t[startpos] = v[l];
        }
        else {
            int m = (l + r) / 2;
            initialize_with_array(2 * startpos, l, m, v);
            initialize_with_array(2 * startpos + 1, m + 1, r, v);

            if (ISMAXRANGE == 1) t[startpos] = max(t[startpos * 2], t[startpos * 2 + 1]);
            else  t[startpos] = min(t[startpos * 2], t[startpos * 2 + 1]);
        }
    }

    void update(int index, T val) { // insert val into location index
        update_full(1, 0, size - 1, index, val);
    }

    void update_full(int startpos, int l, int r, int index, T val) {
        if (l == r) {
            t[startpos] = val;
        }
        else {
            int m = (l + r) / 2;
            if (index <= m) update_full(2 * startpos, l, m, index, val);
            else update_full(2 * startpos + 1, m + 1, r, index, val);

            if (ISMAXRANGE) t[startpos] = max(t[startpos * 2], t[startpos * 2 + 1]);
            else t[startpos] = min(t[startpos * 2], t[startpos * 2 + 1]);
        }
    }

    T query(int l, int r) {  // get range min/max between l and r
        if (l > r) {
            if (ISMAXRANGE) return 0;
            else return VERYBIG;
        }
        return query_full(1, 0, size - 1, l, r);
    }

    T query_full(int startpos, int left, int right, int l, int r) {	 // left/right = current range, l/r = intended query range
        if ((left >= l) && (right <= r)) return t[startpos];
        int m = (left + right) / 2;
        T ans;
        if (ISMAXRANGE) ans = -VERYBIG;
        else ans = VERYBIG;
        if (m >= l) {
            if (ISMAXRANGE) ans = max(ans, query_full(startpos * 2, left, m, l, r));
            else ans = min(ans, query_full(startpos * 2, left, m, l, r));
        }
        if (m + 1 <= r) {
            if (ISMAXRANGE) ans = max(ans, query_full(startpos * 2 + 1, m + 1, right, l, r));
            else ans = min(ans, query_full(startpos * 2 + 1, m + 1, right, l, r));
        }
        return ans;
    }
};

//#define MOD 1000000007
int MOD = 1, root = 2; // 998244353

template<class T> T invGeneral(T a, T b) {
    a %= b; if (a == 0) return b == 1 ? 0 : -1;
    T x = invGeneral(b, a);
    return x == -1 ? -1 : ((1 - (LL)b * x) / a + b) % b;
}

template<class T> struct modular {
    T val;
    explicit operator T() const { return val; }
    modular() { val = 0; }
    modular(const LL& v) {
        val = (-MOD <= v && v <= MOD) ? v : v % MOD;
        if (val < 0) val += MOD;
    }

    friend ostream& operator<<(ostream& os, const modular& a) { return os << a.val; }
    friend bool operator==(const modular& a, const modular& b) { return a.val == b.val; }
    friend bool operator!=(const modular& a, const modular& b) { return !(a == b); }
    friend bool operator<(const modular& a, const modular& b) { return a.val < b.val; }

    modular operator-() const { return modular(-val); }
    modular& operator+=(const modular& m) { if ((val += m.val) >= MOD) val -= MOD; return *this; }
    modular& operator-=(const modular& m) { if ((val -= m.val) < 0) val += MOD; return *this; }
    modular& operator*=(const modular& m) { val = (LL)val * m.val % MOD; return *this; }
    friend modular pow(modular a, LL p) {
        modular ans = 1; for (; p; p /= 2, a *= a) if (p & 1) ans *= a;
        return ans;
    }
    friend modular inv(const modular& a) {
        auto i = invGeneral(a.val, MOD); assert(i != -1);
        return i;
    } // equivalent to return exp(b,MOD-2) if MOD is prime
    modular& operator/=(const modular& m) { return (*this) *= inv(m); }

    friend modular operator+(modular a, const modular& b) { return a += b; }
    friend modular operator-(modular a, const modular& b) { return a -= b; }
    friend modular operator*(modular a, const modular& b) { return a *= b; }

    friend modular operator/(modular a, const modular& b) { return a /= b; }
};

typedef modular<int> mi;
typedef pair<mi, mi> pmi;
typedef vector<mi> vmi;
typedef vector<pmi> vpmi;

//mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

namespace vecOp {
    template<class T> vector<T> rev(vector<T> v) { reverse(ALL(v)); return v; }
    template<class T> vector<T> shift(vector<T> v, int x) { v.insert(v.begin(), x, 0); return v; }

    template<class T> vector<T>& operator+=(vector<T>& l, const vector<T>& r) {
        l.rSZ(max(SZ(l), SZ(r))); F0R(i, SZ(r)) l[i] += r[i]; return l;
    }
    template<class T> vector<T>& operator-=(vector<T>& l, const vector<T>& r) {
        l.rSZ(max(SZ(l), SZ(r))); F0R(i, SZ(r)) l[i] -= r[i]; return l;
    }
    template<class T> vector<T>& operator*=(vector<T>& l, const T& r) { trav(t, l) t *= r; return l; }
    template<class T> vector<T>& operator/=(vector<T>& l, const T& r) { trav(t, l) t /= r; return l; }

    template<class T> vector<T> operator+(vector<T> l, const vector<T>& r) { return l += r; }
    template<class T> vector<T> operator-(vector<T> l, const vector<T>& r) { return l -= r; }
    template<class T> vector<T> operator*(vector<T> l, const T& r) { return l *= r; }
    template<class T> vector<T> operator*(const T& r, const vector<T>& l) { return l * r; }
    template<class T> vector<T> operator/(vector<T> l, const T& r) { return l /= r; }

    template<class T> vector<T> operator*(const vector<T>& l, const vector<T>& r) {
        if (min(SZ(l), SZ(r)) == 0) return {};
        vector<T> x(SZ(l) + SZ(r) - 1); F0R(i, SZ(l)) F0R(j, SZ(r)) x[i + j] += l[i] * r[j];
        return x;
    }
    template<class T> vector<T>& operator*=(vector<T>& l, const vector<T>& r) { return l = l * r; }

    template<class T> vector<T> rem(vector<T> a, vector<T> b) {
        while (SZ(b) && b.back() == 0) b.pop_back();
        assert(SZ(b)); b /= b.back();
        while (SZ(a) >= SZ(b)) {
            a -= a.back() * shift(b, SZ(a) - SZ(b));
            while (SZ(a) && a.back() == 0) a.pop_back();
        }
        return a;
    }
    template<class T> vector<T> interpolate(vector<pair<T, T>> v) {
        vector<T> ret;
        F0R(i, SZ(v)) {
            vector<T> prod = { 1 };
            T todiv = 1;
            F0R(j, SZ(v)) if (i != j) {
                todiv *= v[i].f - v[j].f;
                vector<T> tmp = { -v[j].f,1 }; prod *= tmp;
            }
            ret += prod * (v[i].s / todiv);
        }
        return ret;
    }
}

using namespace vecOp;

class factorial {
public:
    LL MAXX, MOD;
    VL f, ff;

    factorial(LL maxx = 200010, LL mod = 998244353) {
        MAXX = maxx;
        MOD = mod;

        f.RSZ(MAXX);
        ff.RSZ(MAXX);

        f[0] = 1;
        for (int i = 1; i < MAXX; i++) f[i] = (f[i - 1] * i) % MOD;
        for (int i = 0; i < MAXX; i++) ff[i] = mul_inv(f[i], MOD);
    }

    long long mul_inv(long long a, long long b)
    {
        long long b0 = b, t, q;
        long long x0 = 0, x1 = 1;
        if (b == 1) return 1;
        while (a > 1) {
            q = a / b;
            t = b, b = a % b, a = t;
            t = x0, x0 = x1 - q * x0, x1 = t;
        }
        if (x1 < 0) x1 += b0;
        return x1;
    }

    long long division(long long a, long long b) {		// (a / b) mod p = ((a mod p) * (b^(-1) mod p)) mod p
        long long ans, inv;
        inv = mul_inv(b, MOD);
        ans = ((a % MOD) * inv) % MOD;
        return ans;
    }

    LL calcc(LL n, LL a) {
        if (n == a) return 1;
        if (n == 0) return 0;
        if (n < a) return 0;
        LL ans = (f[n] * ff[a]) % MOD;
        ans = (ans * ff[n - a]) % MOD;
        return ans;
    }

    LL calcp(LL n, LL a) {
        LL ans = (f[n] * ff[n - a]) % MOD;
        return ans;
    }

    LL exp(LL base, LL n) {
        base %= MOD;
        LL ans = 1, x = base, MAXLEVEL = 60, i;

        for (i = 0; i < MAXLEVEL; i++) {
            if ((1LL << i) > n) break;
            if ((1LL << i) & n) ans = (ans * x) % MOD;
            x = (x * x) % MOD;
        }
        return ans;
    }
};

#ifdef _MSC_VER 
//#include <intrin.h>
#endif

namespace FFT {
#ifdef _MSC_VER 
    int size(int s) {
        if (s == 0) return 0;
        unsigned long index;
        _BitScanReverse(&index, s);
        return index + 1;
    }
#else
    constexpr int size(int s) { return s > 1 ? 32 - __builtin_clz(s - 1) : 0; }
#endif

    template<class T> bool small(const vector<T>& a, const vector<T>& b) {
        return (LL)SZ(a) * SZ(b) <= 500000;
    }

    void genRoots(vmi& roots) { // primitive n-th roots of unity
        int n = SZ(roots); mi r = pow(mi(root), (MOD - 1) / n);
        roots[0] = 1; FOR(i, 1, n) roots[i] = roots[i - 1] * r;
    }
    void genRoots(VCD& roots) { // change cd to complex<double> instead?
        int n = SZ(roots); LD ang = 2 * PAI / n;
        F0R(i, n) roots[i] = CD(cos(ang * i), sin(ang * i)); // is there a way to do this more quickly?
    }

    template<class T> void fft(vector<T>& a, vector<T>& roots) {
        int n = SZ(a);
        for (int i = 1, j = 0; i < n; i++) { // sort by reverse bit representation
            int bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit; if (i < j) swap(a[i], a[j]);
        }
        for (int len = 2; len <= n; len <<= 1)
            for (int i = 0; i < n; i += len)
                F0R(j, len / 2) {
                auto u = a[i + j], v = a[i + j + len / 2] * roots[n / len * j];
                a[i + j] = u + v, a[i + j + len / 2] = u - v;
            }
    }

    template<class T> vector<T> conv(vector<T> a, vector<T> b) {
        //if (small(a, b)) return a * b;
        int s = SZ(a) + SZ(b) - 1, n = 1 << size(s);
        vector<T> roots(n); genRoots(roots);

        a.RSZ(n), fft(a, roots); b.RSZ(n), fft(b, roots);
        F0R(i, n) a[i] *= b[i];
        reverse(begin(roots) + 1, end(roots)); fft(a, roots); // inverse FFT

        T in = T(1) / T(n); trav(x, a) x *= in;
        a.RSZ(s); return a;
    }

    VL conv(const VL& a, const VL& b) {
        //if (small(a, b)) return a * b;
        VCD X = conv(VCD(ALL(a)), VCD(ALL(b)));
        VL x(SZ(X)); F0R(i, SZ(X)) x[i] = round(X[i].real());
        return x;
    } // ~0.55s when SZ(a)=SZ(b)=1<<19

    VL conv(const VL& a, const VL& b, LL mod) { // http://codeforces.com/contest/960/submission/37085144
        //if (small(a, b)) return a * b;
        int s = SZ(a) + SZ(b) - 1, n = 1 << size(s);

        VCD v1(n), v2(n), r1(n), r2(n);
        F0R(i, SZ(a)) v1[i] = CD(a[i] >> 15, a[i] & 32767); // v1(x)=a0(x)+i*a1(x)
        F0R(i, SZ(b)) v2[i] = CD(b[i] >> 15, b[i] & 32767); // v2(x)=b0(x)+i*b1(x)

        VCD roots(n); genRoots(roots);
        fft(v1, roots), fft(v2, roots);
        F0R(i, n) {
            int j = (i ? (n - i) : i);
            CD ans1 = (v1[i] + conj(v1[j])) * CD(0.5, 0); // a0(x)
            CD ans2 = (v1[i] - conj(v1[j])) * CD(0, -0.5); // a1(x)
            CD ans3 = (v2[i] + conj(v2[j])) * CD(0.5, 0); // b0(x)
            CD ans4 = (v2[i] - conj(v2[j])) * CD(0, -0.5); // b1(x)
            r1[i] = (ans1 * ans3) + (ans1 * ans4) * CD(0, 1); // a0(x)*v2(x)
            r2[i] = (ans2 * ans3) + (ans2 * ans4) * CD(0, 1); // a1(x)*v2(x)
        }
        reverse(begin(roots) + 1, end(roots));
        fft(r1, roots), fft(r2, roots); F0R(i, n) r1[i] /= n, r2[i] /= n;

        VL ret(n);
        F0R(i, n) {
            LL av = (LL)round(r1[i].real()); // a0*b0
            LL bv = (LL)round(r1[i].imag()) + (LL)round(r2[i].real()); // a0*b1+a1*b0
            LL cv = (LL)round(r2[i].imag()); // a1*b1
            av %= mod, bv %= mod, cv %= mod;
            ret[i] = (av << 30) + (bv << 15) + cv;
            ret[i] = (ret[i] % mod + mod) % mod;
        }
        ret.resize(s);
        return ret;
    } // ~0.8s when SZ(a)=SZ(b)=1<<19
}
using namespace FFT;

long long gcd(long long a, long long b)
{
    while (b != 0) {
        long long t = b;
        b = a % b;
        a = t;
    }
    return a;
}



class tree {		// implementation of recurvie programming
    int ct;
public:
    int nn, root;				// # of nodes, id of root
    vector<int> parent;			// parent of each node; -1 if unassigned
    vector<int> depth;			// depth of each node
    vector<int> sz;				// subtree size of each node 
    vector<vector<int>> adj;	// adjacency list from each node
    vector<vector<int>> sons;	// sons list from each node

    // for cartesian_decomposition
    vector<int> in, out;		// starting and ending position of a subtree
    vector<int> pos;			// inorder of DFS

    // for LCA sparse table
    vector<vector<int>> pred;
    int MAXLEVEL;

    tree(int n) {
        nn = n;
        adj.clear();
        adj.resize(n);
    }

    void add_path(int a, int b) {
        adj[a].push_back(b);
        adj[b].push_back(a);
    }

    void add_directed_path(int a, int b) {
        adj[a].push_back(b);
    }

    void dfs_set_root(int id, bool cartesian_decomposition = false) {	// internal
        if (cartesian_decomposition) {
            in[id] = ct;
            pos[ct] = id;
            ct++;
        }

        sz[id]++;

        for (auto p : adj[id]) {
            if (parent[p] == -1) {
                parent[p] = id;
                depth[p] = depth[id] + 1;
                dfs_set_root(p, cartesian_decomposition);
                sz[id] += sz[p];

                sons[id].push_back(p);
            }
        }

        if (cartesian_decomposition) out[id] = ct - 1;
    }

    void set_root(int id, bool cartesian_decomposition = true) {		// set root of the tree and calculate necessary info
        if (cartesian_decomposition) {
            in.resize(nn);
            out.resize(nn);
            pos.resize(nn);
            ct = 0;
        }

        parent.assign(nn, -1);
        depth.assign(nn, -1);
        sz.assign(nn, 0);
        sons.clear();
        sons.resize(nn);

        // dfs_set_root(id, cartesian_decomposition);


        // set root using stack
        stack<pair<int, int>> st;		// id, # of sons processes
        st.push({ id, 0 });
        parent[id] = 0;
        depth[id] = 0;

        int ct = 0;

        while (!st.empty()) {
            int id = st.top().first, x = st.top().second;

            if (x == 0) {
                in[id] = ct;
                pos[ct] = id;
                sz[id] = 1;
                ct++;
            }

            if (x >= adj[id].size()) {
                out[id] = ct - 1;
                if (parent[id] != -1) {
                    sz[parent[id]] += sz[id];
                }
                st.pop();
            }
            else {

                st.top().second++;

                int p = adj[id][x];
                if (parent[p] == -1) {
                    parent[p] = id;
                    depth[p] = depth[id] + 1;
                    sons[id].push_back(p);
                    st.push({ p, 0 });
                }
            }
        }

        int i = 0;
    }

    void eulerian_tour_dfs(int root, vector<int>& ans) {
        ans.push_back(root);
        for (auto p : sons[root]) {
            eulerian_tour_dfs(p, ans);
            ans.push_back(root);
        }
    }

    vector<int> eulerian_tour(int root) {
        vector<int> ans;

        eulerian_tour_dfs(root, ans);

        return ans;
    }


    void prep_LCA() {		// prepare the sparse table for LCA calculation
        MAXLEVEL = 1;
        while ((1 << MAXLEVEL) < nn) MAXLEVEL++;
        MAXLEVEL++;

        pred.assign(MAXLEVEL, vector<int>(nn, 0));
        pred[0] = parent;

        int i, j, k;
        for (i = 1; i < MAXLEVEL; i++) {
            for (j = 0; j < nn; j++) {
                if (pred[i - 1][j] != -1) pred[i][j] = pred[i - 1][pred[i - 1][j]];
            }
        }
    }

    int get_p_ancestor(int a, int p) {		// get p-ancestor of node a;  need to call set_root() and prep_LCA() first
        int i;
        for (i = MAXLEVEL - 1; (i >= 0) && (p > 0) && (a != -1); i--) {
            if ((1 << i) & p) {
                p -= (1 << i);
                a = pred[i][a];
            }
        }
        return a;
    }

    int LCA(int a, int b) {		// get the LCA of a and b, need to call set_root() and prep_LCA() first
        int da = depth[a], db = depth[b];

        if (da > db) {
            swap(da, db);
            swap(a, b);
        }

        int i, j, k;
        for (i = MAXLEVEL - 1; i >= 0; i--) {
            if (db - (1 << i) >= da) {
                db -= (1 << i);
                b = pred[i][b];
            }
        }

        if (a == b) return a;

        for (i = MAXLEVEL - 1; i >= 0; i--) {
            if (pred[i][a] != pred[i][b]) {
                a = pred[i][a];
                b = pred[i][b];
            }
        }

        return parent[a];
    }

    int get_distance(int a, int b) {	// get distance between a and b, need to call set_root() and prep_LCA() first
        int c = LCA(a, b);
        int ans = depth[a] + depth[b] - 2 * depth[c];
        return ans;
    }

    int get_diameter() {
        int a, b, c, i, j, k, id, INF = nn + 100, ans;
        vector<int> dist(nn), last(nn);
        queue<int> q;

        if (nn == 1) return 0;

        // first pass, start with 1 -- any node
        a = 1;
        dist.assign(nn, INF);
        dist[a] = 0;
        q.push(a);

        while (!q.empty()) {
            id = q.front();
            q.pop();

            for (auto p : adj[id]) {
                if (dist[p] == INF) {
                    dist[p] = dist[id] + 1;
                    q.push(p);
                }
            }
        }

        // second pass, start from the most remote node id, collect last to get ID
        a = id;
        dist.assign(nn, INF);
        last.assign(nn, -1);
        dist[a] = 0;
        q.push(a);

        while (!q.empty()) {
            id = q.front();
            q.pop();

            for (auto p : adj[id]) {
                if (dist[p] == INF) {
                    dist[p] = dist[id] + 1;
                    last[p] = id;
                    q.push(p);
                }
            }
        }

        // a and id forms the diameter
        ans = dist[id];

        return ans;

        // construct the path of diamter in path
        vector<int> path;
        b = id;
        c = id;
        do {
            path.push_back(b);
            b = last[b];
        } while (b != -1);

        return ans;
    }
};


// Union-Find Disjoint Sets Library written in OOP manner, using both path compression and union by rank heuristics
// initialize: UnionFind UF(N)

class UnionFind {                                              // OOP style
private:
    vector<int> p, rank, setSize;
    // p = path toward the root of disjoint set; p[i] = i means it is root
    // rank = upper bound of the actual height of the tree; not reliable as accurate measure
    // setSize = size of each disjoint set

    int numSets;
public:
    UnionFind(int N) {
        setSize.assign(N, 1);
        numSets = N;
        rank.assign(N, 0);
        p.assign(N, 0);
        for (int i = 0; i < N; i++) p[i] = i;	// each belongs to its own set
    }

    int findSet(int i) {
        return (p[i] == i) ? i : (p[i] = findSet(p[i]));		// path compression: cut short of the path if possible
    }

    bool isSameSet(int i, int j) {
        return findSet(i) == findSet(j);
    }

    void unionSet(int i, int j) {
        if (!isSameSet(i, j)) {
            numSets--;
            int x = findSet(i), y = findSet(j);
            // rank is used to keep the tree short
            if (rank[x] > rank[y]) { p[y] = x; setSize[x] += setSize[y]; }
            else {
                p[x] = y; setSize[y] += setSize[x];
                if (rank[x] == rank[y]) rank[y]++;
            }
        }
    }

    int numDisjointSets() {		// # of disjoint sets
        return numSets;
    }

    int sizeOfSet(int i) {		// size of set
        return setSize[findSet(i)];
    }
};


#define MAXN 2050000			// total # of prime numbers
#define MAXP 1001000		// highest number to test prime

int prime[MAXN];		// prime numbers: 2, 3, 5 ...
int lp[MAXP];		// lp[n] = n if n is prime; otherwise smallest prime factor of the number
int phi[MAXP];			// phii function

class prime_class {
public:
    long top;

    prime_class() {			// generate all prime under MAXP
        int i, i2, j;

        top = 0;
        lp[0] = 0;
        lp[1] = 1;
        for (i = 2; i < MAXP; i++) lp[i] = 0;

        top = 0;
        for (i = 2; i < MAXP; ++i) {
            if (lp[i] == 0) {
                lp[i] = i;
                prime[top++] = i;
            }
            for (j = 0; (j < top) && (prime[j] <= lp[i]) && (i * prime[j] < MAXP); ++j)
                lp[i * prime[j]] = prime[j];
        }
    }

    bool isprime(long long key)
    {
        if (key < MAXP)	return (lp[key] == key) && (key >= 2);
        else {
            int i;
            for (i = 0; (i < top) && (prime[i] * prime[i] <= key); i++)
                if (key % prime[i] == 0) return false;
            return true;
        }
    }

    unordered_map<int, int> factorize(int key) {
        unordered_map<int, int> ans;

        while (lp[key] != key) {
            ans[lp[key]]++;
            key /= lp[key];
        }
        if (key > 1) ans[key]++;

        return ans;
    }

    int get_phi(int key) {	// calculate Euler's totient function, also known as phi-function
        int ans = key, last = 0;

        while (lp[key] != key) {
            if (lp[key] != last) {
                last = lp[key];
                ans -= ans / last;
            }
            key /= lp[key];
        }
        if ((key > 1) && (key != last)) ans -= ans / key;

        return ans;
    }

    void calc_all_phi(int n) {
        int i, j, k;
        for (int i = 1; i < n; i++) phi[i] = i;
        for (int i = 2; i < n; i++) {
            if (phi[i] == i) {
                for (int j = i; j < n; j += i) {
                    phi[j] /= i;
                    phi[j] *= i - 1;
                }
            }
        }
    }

    vector<pair<long long, long long>> factorize_full(long long key) {		// can be used to factorize numbers >= MAXP
        vector<pair<long long, long long>> ans;

        long i, ct, sq = sqrt(key) + 10;

        for (i = 0; (i < top) && (prime[i] <= sq); i++)
            if (key % prime[i] == 0) {
                ct = 0;
                while (key % prime[i] == 0) {
                    ct++;
                    key /= prime[i];
                }
                ans.push_back({ prime[i], ct });
            }
        if (key > 1) {
            ans.push_back({ key, 1 });
        }
        return ans;
    }

    void generate_divisors(int step, int v, vector<pair<int, int>>& fp, vector<int>& ans) {
        if (step < fp.size()) {
            generate_divisors(step + 1, v, fp, ans);
            for (int i = 1; i <= fp[step].second; i++) {
                v *= fp[step].first;
                generate_divisors(step + 1, v, fp, ans);
            }
        }
        else ans.push_back(v);
    }

    void generate_divisors_full(long long step, long long v, vector<pair<long long, long long>>& fp, vector<long long>& ans) {
        if (step < fp.size()) {
            generate_divisors_full(step + 1, v, fp, ans);
            for (int i = 1; i <= fp[step].second; i++) {
                v *= fp[step].first;
                generate_divisors_full(step + 1, v, fp, ans);
            }
        }
        else ans.push_back(v);
    }

    vector<int> get_divisors(int key) {
        unordered_map<int, int> f = factorize(key);
        int n = f.size();
        vector<pair<int, int>> fp;
        for (auto p : f) fp.push_back(p);
        vector<int> ans;
        generate_divisors(0, 1, fp, ans);
        return ans;
    }

    vector<long long> get_divisors_full(long long key) {
        vector<pair<long long, long long>> f = factorize_full(key);
        int n = f.size();
        vector<pair<long long, long long>> fp;
        for (auto p : f) fp.push_back(p);
        vector<long long> ans;
        generate_divisors_full(0, 1, fp, ans);
        return ans;
    }


    long long get_divisors_count(long long key) {
        vector<pair<long long, long long>> f = factorize_full(key);
        long long ans = 1;
        for (auto p : f) ans *= (p.second + 1);
        return ans;
    }

};


#define MP make_pair
#define PB push_back
#define F first
#define S second
#define LB lower_bound
#define UB upper_bound

#define SZ(x) ((int)x.size())
#define LEN(x) ((int)x.length())
#define ALL(x) begin(x), end(x)
#define RSZ resize
#define ASS assign
#define REV(x) reverse(x.begin(), x.end());

#define MAX(x) *max_element(ALL(x))
#define MIN(x) *min_element(ALL(x))
#define FOR(i, n) for (int i = 0; i < n; i++) 
#define FOR1(i, n) for (int i = 1; i <= n; i++) 
#define SORT(x) sort(x.begin(), x.end())
#define RSORT(x) sort(x.rbegin(), x.rend())
#define SUM(x) accumulate(x.begin(), x.end(), 0LL)


#define IN(x) cin >> x;
#define OUT(x) cout << x << "\n";
#define INV(x, n) FOR(iiii, n) { cin >> x[iiii]; }
#define INV1(x, n) FOR1(iiii, n) { cin >> x[iiii]; }
#define OUTV(x, n) { FOR(iiii, n) { cout << x[iiii] << " "; } cout << "\n"; }
#define OUTV1(x, n) { FOR1(iiii, n) { cout << x[iiii] << " "; } cout << "\n"; }
#define OUTYN(x) { if (x) cout << "YES\n"; else cout << "NO\n"; }
#define OUTyn(x) { if (x) cout << "Yes\n"; else cout << "No\n"; }


#define MOD7 1000000007
#define MOD9 1000000009
#define MOD3 998244353

#define MAXN 300010
#define SQ 70
vector<fenwick_tree<int>> ft(SQ, fenwick_tree<int>(MAXN));

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);


    int n, q, i, j, k, x, xx, m, xid;

    cin >> n >> q;
    unordered_map<int, int> id;
    VI a(n + 1), tid(q), l(q), r(q), kk(q), pos(q), newv(q);

    FOR1(i, n) {
        cin >> a[i];
        id[a[i]] = 0;
    }

    srand(time(NULL));

    FOR(i, q) {
        cin >> tid[i];
        if (tid[i] == 1) {
            cin >> pos[i] >> newv[i];
            id[newv[i]] = 0;
        }
        else cin >> l[i] >> r[i] >> kk[i];
    }

    m = 0;
    for (auto p : id) id[p.first] = m++;

    VVL plist(m);
    FOR(i, m) {
        FOR(j, SQ) {
            if ((rand() % 100) < 20) {
                plist[i].push_back(j);
            }
        }
    }

    FOR1(i, n) {
        xid = id[a[i]];
        for (auto p : plist[xid]) ft[p].update(i, 1);
    }

    FOR(i, q) {
        if (tid[i] == 1) {
            xid = id[a[pos[i]]];
            for (auto p : plist[xid]) ft[p].update(pos[i], -1);
            a[pos[i]] = newv[i];
            xid = id[a[pos[i]]];
            for (auto p : plist[xid]) ft[p].update(pos[i], 1);
        }
        else {
            bool ans = true;
            if ((r[i] + 1 - l[i]) % kk[i] != 0) ans = false;
            else if (kk[i] == 1) ans = true;
            else if (r[i] - l[i] < 200) {
                map<LL, LL> ct;
                for (j = l[i]; j <= r[i]; j++) ct[a[j]]++;
                for (auto p : ct) {
                    if ((p.second % kk[i]) != 0) {
                        ans = false;
                        break;
                    }
                }
            }
            else {
                FOR(j, SQ) {
                    x = ft[j].query_full(l[i], r[i]);
                    if ((x % kk[i]) != 0) {
                        ans = false;
                        break;
                    }
                }
            }
            OUTYN(ans);
        }
    }



    return 0;
}

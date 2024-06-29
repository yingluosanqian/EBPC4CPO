#define REP(x,n) for(int x=0;x<n;++x)
#define REP2(x,a,b) for(int x=a;x<b;++x)
#define IT(x) (x).begin(), (x).end()
#include <cstdio>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <cstring>
#include <string>
#include <tuple>
#include <optional>
#include <utility>
#include <cassert>
#include <string>
#include <map>

using namespace std;

typedef long long ll;
// typedef pair<int, int> pi;

typedef unsigned int ui;

typedef vector<int> vi;
typedef vector<ui> vu;
typedef vector<ll> vl;

typedef vector<vi> vii;
typedef vector<vl> vll;

typedef std::string str;

template<typename T, typename ...Args>
void read(T &first, Args&&... rest);

void read(int &x) { scanf("%d",&x); }
void read(char &x) { scanf("%c", &x); }
void read(ui &x) { scanf("%u",&x); }
void read(ll &x) { scanf("%lld",&x); }

template<typename U, typename V>
void read(pair<U, V> &x) { read(x.first, x.second); }

template<typename T>
void read(pair<vector<T>&, int> x) {
    vector<T> &r = x.first;
    int n = x.second;
    r = std::decay_t<decltype(x.first)>(n);
    REP(i,n) read(r[i]);
}

void read(std::string &s) {
    constexpr int MAXN = 1e6 + 10;
    // constexpr int MAXN = 1e4;
    static char buf[MAXN];
    scanf("%s", buf);
    s = buf;
}

template<typename T>
void read(vector<T> &x) {
    int n;
    read(n);
    read<T>({x, n});
}

template<typename T, typename ...Args>
void read(T &first, Args&&... rest){
    read(first);
    read(rest...);
}

void _print() {}
void _print(int x) {
    printf("%d ", x);
}
void _print(char x) {
    printf("%c ", x);
}
void _print(ui x) {
    printf("%u ", x);
}
void _print(size_t x) {
    printf("%zu ", x);
}
void _print(ll x) {
    printf("%lld ", x);
}
void _print(bool x) {
    printf("%s ", x ? "YES" : "NO");
}
void _print(const char *x) {
    printf("%s ", x);
}
void _print(const std::string &s) {
    printf("%s ", s.c_str());
}
/*
void _print(const vi &xs) {
    for(auto x:xs) printf("%d ", x);
}
void _print(const vu &xs) {
    for(auto x:xs) printf("%u ", x);
}
void _print(const vl &xs) {
    for(auto x: xs) printf("%lld ", x);
}
*/
template<typename T>
void _print(const std::vector<T> &xs) {
    for(auto x: xs) _print(x);
}
template<typename T>
void _print(const std::optional<T> &xs) {
    if (xs == std::nullopt) printf("NO\n");
    else {
        printf("YES\n");
        _print(xs.value());
    }
}
template<typename ...Args>
void print(const Args&... xs) {
    (_print(xs), ...);
    printf("\n");
}



struct Prob {
    struct input_tag_t {};
    constexpr static input_tag_t input_tag{};

    struct Query {
    };

    ll t, x, y;

    Prob(input_tag_t _t) {
        read(t, x, y);
    }

    Prob() = default;
    Prob(const Prob&o) = default;
    Prob(Prob&&o) = default;
    Prob& operator=(const Prob&o) = default;
    Prob& operator=(Prob&&o) = default;

    auto solve() {
        if (t - x - y < 0) return false;
        constexpr int L = 120;
        ll dp[L + 1][L + 1];
        std::fill(*dp, dp[L + 1], 0LL);
        dp[0][0] = t - x - y;
        REP(i, L) {
            REP(j, L) {
                ll v = dp[i][j];
                dp[i + 1][j] += v / 2;
                dp[i][j + 1] += v - (v / 2);
            }
        }
        int u = 0, v = 0;
        while (u < L && v < L) {
            if (u == x && v == y) return true;
            if (dp[u][v] % 2 == 1) ++u;
            else ++v;
        }
        return false;
    }

    static Prob input() {
        return Prob(input_tag);
    }

    void solvef() {
        _solvef(&Prob::solve);
    }

    void _solvef(void(Prob::*f)(void)) {
        (this->*f)();
    }

    template<typename T>
    void _solvef(T(Prob::*f)(void)) {
        print((this->*f)());
    }
};



int main(void) {
    //print(Prob::input().solve());

    int T=1;
    read(T);
    while (T--) {
        Prob sol = Prob::input();
        sol.solvef();
    }
    return 0;
}


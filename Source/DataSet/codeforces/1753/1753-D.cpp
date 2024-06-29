#ifndef _GLIBCXX_NO_ASSERT
#include <cassert>
#endif
#include <cctype>
#include <cerrno>
#include <cfloat>
#include <ciso646>
#include <climits>
#include <clocale>
#include <cmath>
#include <csetjmp>
#include <csignal>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
 
#if __cplusplus >= 201103L
#include <ccomplex>
#include <cfenv>
#include <cinttypes>
//#include <cstdalign>
#include <cstdbool>
#include <cstdint>
#include <ctgmath>
#include <cwchar>
#include <cwctype>
#endif
 
// C++
#include <algorithm>
#include <bitset>
#include <complex>
#include <deque>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <ios>
#include <iosfwd>
#include <iostream>
#include <istream>
#include <iterator>
#include <limits>
#include <list>
#include <locale>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <ostream>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <typeinfo>
#include <utility>
#include <valarray>
#include <vector>
 
#if __cplusplus >= 201103L
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <forward_list>
#include <future>
#include <initializer_list>
#include <mutex>
#include <random>
#include <ratio>
#include <regex>
#include <scoped_allocator>
#include <system_error>
#include <thread>
#include <tuple>
#include <typeindex>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#endif
using namespace std;
using i64 = long long;
//using i128 = __int128;
#define MAXN 1000000
#define MAXM 200050
#define M 1000000
#define K 17
#define MAXP 25
#define MAXK 55
//#define MAXC 255
#define MAXERR 105
#define MAXLEN 105
#define MDIR 10
#define MAXR 705
#define BASE 102240
#define MAXA 28
#define MAXT 100005
#define LIMIT 86400
#define MAXV 305
#define LEQ 1
#define EQ 0
#define OP 0
#define CLO 1
#define DIG 1
#define C1 0
#define C2 1
#define PLUS 0
#define MINUS 1
#define MUL 2
#define CLO 1
#define VERT 1
//#define B 31
#define B2 1007
#define W 1
#define H 20
#define SPEC 1
#define MUL 2
#define CNT 3
#define ITER 1000
#define INF 1e18
#define EPS 1e-7
#define MOD 998244353
#define FACT 100000000000000
#define SRC 0
#define PI acos(-1)
#define PUSH 0
#define POP 1
typedef long long ll;
typedef ll hash_t;
//typedef __int128_t lint;
typedef long double ld;
typedef pair<int,int> ii;
typedef pair<int,ll> ilp;
typedef pair<ll,ii> pl;
typedef pair<ll, ll> pll;
typedef pair<ll,int> ppll;
typedef pair<ll,int> li;
typedef pair<ll,ll> iv;
typedef pair<double,int> ip;
typedef tuple<int,int,int> iii;
typedef tuple<ll, ll, ll> tll;
typedef tuple<ld, int, int> iit;
typedef tuple<ll,int,int> i3;
typedef vector<vector<int>> vv;
typedef vector<int> vi;
typedef vector<ll> vl;
typedef vector<ii> vii;
typedef vector<vector<ll>> vll;
typedef complex<ld> cd;
typedef tuple<int,ll,ll> tpl;
#define pb push_back
#define eb emplace_back
#define ppb pop_back
#define ppf pop_front
#define pf push_front
#define bk back()
#define frt front()
#define ins insert
#define er erase
#define sc second
#define fr first
#define mp make_pair
#define mt make_tuple
#define repl(i,x,y) for (int i = (x); i <= (y); ++i)
#define rep(i,x,y) for (int i = (x); i < (y); ++i)
#define rev(i,x,y) for (int i = (x); i >= (y); --i)
#define repd(i,x,y,d) for (int i = (x); i < (y); i += (d))
#define LSOne(S) (S & (-S))
#define trav(i,v) for (auto &i : v)
#define foreach(it,v) for (auto it = begin(v); it != end(v); ++it)
#define bend(v) (v).begin(), (v).end()
#define rbend(v) (v).rbegin(), (v).rend()
#define sortarr(v) sort(bend(v))
#define rsortarr(v) sort(rbend(v))
#define unique(v) v.er(unique(bend(v)), end(v))
#define extend(A,B) A.insert(end(A), bend(B))
#define sz(A) (int)(A.size())
#define fill(V) iota(bend(V), (0))
#define vfill(V, st) iota(bend(V), st)
#define sum(V) accumulate(bend(V), 0LL)
#define vsum(V, st) accumulate(bend(V), (ll)(st))
template<class T> bool ckmin(T &a, T b) { return a > b ? a = b, 1 : 0; };
template<class T> bool ckmax(T &a, T b) { return a < b ? a = b, 1 : 0; };
template<class T> void amax(T &a, T b, T c) { a = max(b, c); };
template<class T> void amin(T &a, T b, T c) { a = min(b, c); };
template<class T> T getmax(vector<T> &v) { return *max_element(bend(v)); };
template<class T> T getmin(vector<T> &v) { return *min_element(bend(v)); };
template<class T> int compress(vector<T> &v, T &val) { return (int)(lower_bound(bend(v), val) - begin(v)); };
template<class T> auto vfind(vector<T> &v, T val) {
    return find(bend(v), val);
}
template<class T> auto verase(vector<T> &v, T val) {
    return v.er(vfind(v, val));
}
template<class T> void revarr(vector<T> &v) { reverse(bend(v)); };
template<class T> void print(vector<T> &v, char end) { trav(i,v) cout << i << end;
}
mt19937 gen((int)(chrono::steady_clock::now().time_since_epoch().count()));
int rng(int l,int r) {return uniform_int_distribution<int>(l,r)(gen);}

void fast_io() {
    ios::sync_with_stdio(false);
    cin.tie(NULL); cout.tie(NULL);
}

int r,c;
vector<vector<char>> a;
vector<vector<ll>> d;
bool ib(int i, int j) {
    return i >= 0 && j >= 0 && i < r && j < c;
}

vector<ii> mv2 = {{-2,0}, {2,0}, {0,-2}, {0,2}}, mv1 = {{-1,0}, {1,0}, {0,-1}, {0,1}};
const int UP = 0, DOWN = 1, LFT = 2, RGT = 3;
const int INF2 = 1e9;
//(ix, iy) from mv1, (nx,ny) from mv2
//check if we can slide the sunbed here
bool comp(int ix, int iy, int nx, int ny, int command) {
    if (a[ix][iy] == '.' || a[nx][ny] == '.') return false;
    if (a[ix][iy] == '#' || a[nx][ny] == '#') return false;
    if (command == UP) return (a[ix][iy] == 'D' || a[nx][ny] == 'U');
    if (command == DOWN) return (a[ix][iy] == 'U' || a[nx][ny] == 'D');
    if (command == LFT) return (a[ix][iy] == 'R' || a[nx][ny] == 'L');
    if (command == RGT) return (a[ix][iy] == 'L' || a[nx][ny] == 'R');
    assert(false);
    return false;
}

//(ix, iy) from mv1
//check if we can rotate the sunbed the sunbed here
ii dir(int ix, int iy, int command) {
    ii ans = {INF2, INF2};
    if (a[ix][iy] == '.' || a[ix][iy] == '#') return ans;
    if (command == UP) {
        if (a[ix][iy] == 'R') {
            ans = {-1,-1};
        } else if (a[ix][iy] == 'L') {
            ans = {-1,1};
        }
    } else if (command == DOWN) {
        if (a[ix][iy] == 'R') {
            ans = {1,-1};
        } else if (a[ix][iy] == 'L') {
            ans = {1,1};
        }
    } else if (command == LFT) {
        if (a[ix][iy] == 'D') {
            ans = {-1,-1};
        } else if (a[ix][iy] == 'U') {
            ans = {1,-1};
        }
    } else if (command == RGT) {
        if (a[ix][iy] == 'D') {
            ans = {-1,1};
        } else if (a[ix][iy] == 'U') {
            ans = {1,1};
        }
    }
    return ans;
}

//idea is we perform SSSP to find the minimum cost to free up a cell
//then the answer is just the sum of the costs to free up 2 adjacent cells

//note that while performing SSSP, we need to move from one free cell to another that was originally occupied by a sunbed
//except for the starting cell which can be free originally
int main() {
    fast_io();
    cin >> r >> c;
    int p,q; cin >> p >> q;
    a.resize(r + 2), d.resize(r + 2);
    rep(i, 0, r + 2) a[i].resize(c + 2), d[i].resize(c + 2);
    priority_queue<i3, vector<i3>, greater<i3>> pq;
    rep(i,0,r) {
        rep(j,0,c) {
            cin >> a[i][j], d[i][j] = INF;
            if (a[i][j] == '.') {
                d[i][j] = 0;
                pq.push({d[i][j], i, j});
            }
        }
    }
    while (!pq.empty()) {
        auto [dist, x, y] = pq.top();
        pq.pop();
        if (dist != d[x][y]) continue;
        
        //handle sliding first
        rep(i,0,4) {
            int nx = x + mv2[i].fr, ny = y + mv2[i].sc, ix = x + mv1[i].fr, iy = y + mv1[i].sc;
            if (ib(nx,ny) && comp(ix,iy,nx,ny,i) && d[nx][ny] > d[x][y] + q) {
                d[nx][ny] = d[x][y] + q;
                pq.push({d[nx][ny], nx, ny});
            }
        }
        
        //now handle the rotation
        rep(i,0,4) {
            int ix = x + mv1[i].fr, iy = y + mv1[i].sc;
            if (!ib(ix,iy)) continue;
            ii rdir = dir(ix,iy,i);
            if (rdir.fr == INF2) continue;
            int nx = x + rdir.fr, ny = y + rdir.sc;
            if (d[nx][ny] > d[x][y] + p) {
                d[nx][ny] = d[x][y] + p;
                pq.push({d[nx][ny], nx, ny});
            }
        }
    }
    ll ans = INF;
    rep(i,0,r) {
        rep(j,0,c) {
            if (a[i][j] == '#') continue;
            rep(k,0,4) {
                int ix = i + mv1[k].fr, iy = j + mv1[k].sc;
                if (ib(ix,iy) && a[ix][iy] != INF && d[i][j] != INF && d[ix][iy] != INF) {
                    ckmin(ans, d[i][j] + d[ix][iy]);
                }
            }
        }
    }
    if (ans != INF) cout << ans;
    else cout << -1;
}
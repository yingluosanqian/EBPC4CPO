// LUOGU_RID: 140095778
#include <iostream>
#include <algorithm>
#include <cstring>
#include <vector>
#include <cstdio>
#include <queue>
#include <unordered_map>
#include <set>
#include <cmath>
#include <map>
#include <stack>
#include <sstream>
#include <random>
#include <ctime>
#include <chrono>
#include <bitset>
#include <iomanip>
#define ios ios::sync_with_stdio(false), cin.tie(0), cout.tie(0);
#define TIME cerr << 1e3 * clock() / CLOCKS_PER_SEC << " ms\n";
#define debug(x) cerr << #x << " : " << x << '\n'
#define all(v) v.begin() , v.end()
// std::mt19937 rnd(std::chrono::system_clock::now().time_since_epoch().count());

#define x first
#define y second
// #define int long long

using namespace std;

typedef long long LL;
typedef pair<int , int> PII;

const int N = 5005;
const int INF = 0x3f3f3f3f;
const int mod = 998244353;

int read(){
    int x = 0 , f = 1;
    char c = getchar();
    while(c < '0' || c > '9'){
        if(c == '-')f = -1;
        c = getchar();
    }
    while(c >= '0' && c <= '9'){
        x = x * 10 + c - '0';
        c = getchar();
    }
    return x * f;
}

template <typename T> inline void write(T x) {
    static int stk[100], top = 0;
    if (x == 0) return (void)putchar('0');
    if (x < 0) x = -x, putchar('-');
    while (x) stk[++top] = x % 10, x /= 10;
    while (top) putchar(stk[top--] + '0');
}

int a[N];
int p[N];
int f[N][N];
int s[N];

void solve(){
    int n , k;
    cin >> n >> k;
    k ++;
    for(int i = 1 ; i <= n ; i ++)cin >> a[i] , p[a[i]] = i;
    memset(f , 0x3f , sizeof f);
    f[0][0] = 0;
    int tot = (1 << k);
    for(int i = 0 ; i <= n ; i ++){
        if(i > 0){
            for(int j = 1 ; j <= p[i] ; j ++)s[j] ++;
        }
        for(int j = 0 ; j < tot ; j += 2){
            if(f[i][j] == INF)continue;
            for(int x = 1 ; x <= k && x + i <= n ; x ++){
                int cur = i , st = j | (1 << (x - 1)) , cnt = s[p[x + i] + 1];
                while(st & 1){
                    cur ++;
                    st >>= 1;
                }
                for(int y = 1 ; y <= k ; y ++)
                    if((j >> (y - 1) & 1) && p[i + y] > p[i + x])cnt ++;
                f[cur][st] = min(f[cur][st] , f[i][j] + cnt);
            }
        }
    }
    cout << f[n][0] << endl;
}

signed main(){
    // freopen("test.in","r",stdin);
    // freopen("test.out","w",stdout);
    int T = 1;
    // cin >> T;
    while(T --){
        solve();
    }
    return 0;
}
/* stuff you should look for
 * int overflow, array bounds
 * special cases (n=1?)
 * do smth instead of nothing and stay organized
 * WRITE STUFF DOWN
 * DON'T GET STUCK ON ONE APPROACH
 */
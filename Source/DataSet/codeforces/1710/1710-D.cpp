#include <cassert>
#include <cstdio>
using namespace std;

const int maxn = 2010;
int find(int);
int fa[maxn];

int main() {
    int t;
    scanf("%d", &t);
    while (t--) {
        int n;
        scanf("%d", &n);
        static bool mp[maxn][maxn];
        for (int i = 1; i <= n; i++) {
            static char str[maxn];
            scanf("%s", str);
            for (int j = 0; i + j <= n; j++) mp[i][j] = str[j] == '1';
        }
        static int L[maxn], R[maxn];
        for (int i = 1; i <= n; i++) fa[i] = L[i] = R[i] = i;
        static int Gx[maxn], Gy[maxn], cnt;
        cnt = 0;
        for (int j = 1; j < n; j++)
            for (int i = 1; i + j <= n; i++)
                if (mp[i][j]) {
                    int fx = find(i), fy = find(i + j);
                    if (fx == fy)
                        continue;
                    static int seg[maxn], c;
                    c = 0;
                    int p = i, fp = fx;
                    while (fp != fy) seg[++c] = L[fp], p = R[fp] + 1, fp = find(p);
                    seg[1] = i, seg[++c] = i + j;
                    if (c == 2)
                        cnt++, Gx[cnt] = seg[1], Gy[cnt] = seg[2];
                    else {
                        cnt++, Gx[cnt] = seg[1], Gy[cnt] = seg[c];
                        cnt++, Gx[cnt] = seg[c], Gy[cnt] = seg[2];
                        for (int x = 3; x < c; x++) cnt++, Gx[cnt] = seg[1], Gy[cnt] = seg[x];
                    }
                    for (int x = 2; x <= c; x++) fa[find(seg[x])] = fx;
                    R[fx] = R[fy];
                }
        for (int i = 1; i <= cnt; i++) printf("%d %d\n", Gx[i], Gy[i]);
    }
    return 0;
}

int find(const int u) { return u == fa[u] ? u : fa[u] = find(fa[u]); }

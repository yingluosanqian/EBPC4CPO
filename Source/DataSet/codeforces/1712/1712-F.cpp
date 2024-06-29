// LUOGU_RID: 102297525
/*
好像可以 dsu ，然后因为权值的范围是不超过子树大小的。
可以做到一只 log。haluan 
*/
#include <set>
#include <map>
#include <cmath>
#include <queue>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <iostream>
#include <algorithm>
#define pii pair <int , int>
#define pll pair <LL , LL>
#define mp make_pair
#define fs first
#define sc second
using namespace std;
typedef long long LL;
typedef unsigned long long ULL;

//const int Mxdt=100000;
//static char buf[Mxdt],*p1=buf,*p2=buf;
//#define getchar() p1==p2&&(p2=(p1=buf)+fread(buf,1,Mxdt,stdin),p1==p2)?EOF:*p1++;

template <typename T>
void read(T &x) {
	T f=1;x=0;char s=getchar();
	while(s<'0'||s>'9') {if(s=='-') f=-1;s=getchar();}
	while(s>='0'&&s<='9') {x=(x<<3)+(x<<1)+(s^'0');s=getchar();}
	x *= f;
}

template <typename T>
void write(T x , char s='\n') {
	if(!x) {putchar('0');putchar(s);return;}
	if(x<0) {putchar('-');x=-x;}
	T t = 0 , tmp[25] = {};
	while(x) tmp[t ++] = x % 10 , x /= 10;
	while(t -- > 0) putchar(tmp[t] + '0');
	putchar(s);
}

const int MAXN = 1e6 + 5;

int n , q;
vector <int> G[MAXN];

int dis[MAXN];
void dfs(int x , int fa) {
	for (auto v:G[x]) {
		if(v == fa) continue;
		dis[v] = dis[x] + 1;
		dfs(v , x);
	}
}

int mn[MAXN] , siz[MAXN] , hson[MAXN] , dep[MAXN] , mx[MAXN];
void pre(int x , int fa) {
	siz[x] = 1;
	dep[x] = dep[fa] + 1;
	if(G[x].size() == 1u) mn[x] = 0;
	for (auto v:G[x]) {
		if(v == fa) continue;
		mn[v] = min(mn[x] + 1 , mn[v]);
		mx[x] = max(mx[v] + 1 , mx[x]);
		pre(v , x);
		mn[x] = min(mn[v] + 1 , mn[x]);
		siz[x] += siz[v];
		if(siz[hson[x]] < siz[v]) hson[x] = v;
	}
}

//int calc(int x , int v) {
//	dis[x] = 0;
//	dfs(x , 0);
//	int res = 0;
//	for (int i = 1; i <= n; ++i) if(dis[i] >= mn[x] + mn[i] + v) res = max(res , mn[x] + v + mn[i]);
//	return res;
//}

//int val[MAXN << 1] , CLJ; 

int d[MAXN];
void solve(int x , int fa) {
	d[++d[0]] = x;
	for (auto v:G[x]) {
		if(v == fa) continue;
		solve(v , x);
	}
}

int L;
struct BIT {
	int tr[MAXN << 1];
	inline void update(int x , int y) {for (; x <= L * 2; x += (x & (-x))) tr[x] = max(tr[x] , y);}
	inline int find(int x) {if(x < 0) return -1e9; int res = -1e9;for (; x; x -= (x & (-x))) res = max(res , tr[x]);return res;}
	inline void clear(int x) {for (; x <= L * 2; x += (x & (-x))) tr[x] = -1e9;}
}T;

void clear(int x , int fa) {
	int v = L + dep[x] - mn[x];
	T.clear(L * 2 - v + 1);
	for (auto v:G[x]) {
		if(v == fa) continue;
		clear(v , x);
	}
}

pii qr[MAXN];
int Ans[MAXN] , cnt;
void calc(int x , int fa , int kp) {
	for (auto v:G[x]) {
		if(v == fa || v == hson[x]) continue;
		calc(v , x , 0);
	}
	if(hson[x]) calc(hson[x] , x , 1);
	for (auto v:G[x]) {
		if(v == fa || v == hson[x]) continue;
		d[0] = 0;
		solve(v , x);
		for (int i = 1; i <= d[0]; ++i) {
			for (int j = 1; j <= cnt; ++j) {
				int nd = qr[j].fs + 2 * dep[x] - dep[d[i]] + mn[d[i]] + L;
				Ans[qr[j].sc] = max(Ans[qr[j].sc] , T.find(L * 2 - nd + 1) + mn[d[i]] + qr[j].fs);
			}
			
//			for (int j = nd; j <= n * 2; ++j) Ans = max(Ans , val[j] + mn[d[i]] + CLJ);
			// !!!
		}
		for (int i = 1; i <= d[0]; ++i) {
			int v = L + dep[d[i]] - mn[d[i]];
			T.update(L * 2 - v + 1 , mn[d[i]]);
//			val[] = max(val[n + dep[d[i]] - mn[d[i]]] , mn[d[i]]);
		}
	}
	d[0] = 0;
	d[++d[0]] = x;
	for (int i = 1; i <= d[0]; ++i) {
		for (int j = 1; j <= cnt; ++j) {
			int nd = qr[j].fs + 2 * dep[x] - dep[d[i]] + mn[d[i]] + L;
			Ans[qr[j].sc] = max(Ans[qr[j].sc] , T.find(L * 2 - nd + 1) + mn[d[i]] + qr[j].fs);
		}
//			for (int j = nd; j <= n * 2; ++j) Ans = max(Ans , val[j] + mn[d[i]] + CLJ);
		// !!!
	}
	for (int i = 1; i <= d[0]; ++i) {
		int v = L + dep[d[i]] - mn[d[i]];
		T.update(L * 2 - v + 1 , mn[d[i]]);
//			val[] = max(val[n + dep[d[i]] - mn[d[i]]] , mn[d[i]]);
	}
	if(!kp) {
		clear(x , fa);
//		for (int i = 1; i <= n * 2; ++i) val[i] = 0;
	}
}

int main() {
//	freopen("plot.in" , "r" , stdin);
//	freopen("plot.out" , "w" , stdout);
	read(n);
	for (int i = 2; i <= n; ++i) {
		int p;read(p);
		G[p].push_back(i);
		G[i].push_back(p);
	}
	
	dfs(1 , 0);
	int x = 1;
	for (int i = 1; i <= n; ++i) if(dis[i] > dis[x]) x = i;
	dis[x] = 0;
	dfs(x , 0);
	int y = 1;
	for (int i = 1; i <= n; ++i) if(dis[i] > dis[y]) y = i;
	L = dis[y];
//	cerr << L << endl;
	
	for (int i = 1; i <= n; ++i) mn[i] = 1e9;
	pre(1 , 0) , pre(1 , 0);
	read(q);
	for (int i = 1; i <= q; ++i) {
		int v;read(v);//CLJ = v;
		if(L > v) {
			qr[++cnt] = mp(v , i);
//			Ans = 0;
//			calc(1 , 0 , 0);
//			write(Ans , ' ');
		}
		else {
			Ans[i] = L;
//			write(L , ' ');
		}
	}
	L ++;
	for (int i = 1; i <= L * 2; ++i) T.tr[i] = -1e9;
	calc(1 , 0 , 1);
	for (int i = 1; i <= q; ++i) write(Ans[i] , ' ');
	return 0;
}
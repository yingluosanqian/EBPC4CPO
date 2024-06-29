// LUOGU_RID: 149307091
#import <bits/stdc++.h>
#define ll long long
using namespace std;
const int maxn = 1e5 + 5;
int fa[maxn], sz[maxn];
int find(int x){
    return fa[x] == x ? x : fa[x] = find(fa[x]);
}
struct e{
    int a, b, c;
};
bool cmp(e a,e b){
	return a.c<b.c;
}
int main() {
    int n;
    cin >> n;
    for(int i = 1; i <= n; i++)fa[i] = i, sz[i] = 1;
    vector<e>vec(n - 1);
    for(int i = 0; i < n - 1; i++)
    cin >> vec[i].a >> vec[i].b >> vec[i].c;
    sort(vec.begin(), vec.end(), cmp);
    ll ans = 0;
    for(int i = 0; i < n - 1; i++){
        int a = find(vec[i].a);
        int b = find(vec[i].b);        
        if(a != b){
            ans += 1LL * vec[i].c * sz[a] * sz[b];
            sz[b] += sz[a];
            sz[a] = 0;
            fa[a] = b;
        }
    }
    cout << ans;
}
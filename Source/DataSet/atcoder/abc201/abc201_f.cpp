// LUOGU_RID: 148601465
//Ве№И ABC201F 
//https://www.luogu.com.cn/problem/AT_abc201_f
#include<bits/stdc++.h>
#define LL long long
#define fr(x) freopen(#x".in","r",stdin);freopen(#x".out","w",stdout);
using namespace std;
const int N=2e5+5;
int n,p[N];LL a[N],b[N],c[N],f[N],ans=1e18;
struct Bit
{
	LL a[N];
	inline int lb(int x){return x&-x;}
	inline void add(int wz,LL x){for(;wz<=n;wz+=lb(wz)) a[wz]=min(a[wz],x);}
	inline LL ask(int wz){LL ans=1e18;for(;wz;wz-=lb(wz)) ans=min(ans,a[wz]);return ans;}
}B;
int main()
{
	ios::sync_with_stdio(0);cin.tie(0);cout.tie(0);cin>>n;
	for(int i=1,x;i<=n;i++) cin>>x,p[x]=i,B.a[i]=1e18;
	for(int i=1;i<=n;i++) cin>>a[i]>>b[i]>>c[i],b[i]=min(b[i],a[i]),c[i]=min(c[i],a[i]);
	for(int i=1;i<=n;i++) a[i]+=a[i-1],b[i]+=b[i-1];
	for(int i=n;i;i--) c[i]+=c[i+1];
	for(int i=1;i<=n;i++) ans=min(ans,(f[p[i]]=min(b[i-1],B.ask(p[i]-1)+a[i-1]))+c[i+1]),B.add(p[i],f[p[i]]-a[i]);
	return cout<<ans,0;
}
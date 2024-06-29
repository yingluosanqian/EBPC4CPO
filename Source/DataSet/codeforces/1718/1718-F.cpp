#include<bits/stdc++.h>
namespace ifzw{
#define ll long long
#define dd double
#define ull unsigned ll 
#define LL __int128
#define siz(A) ((int)A.size())
using namespace std;
char gc(){static char buf[1<<16],*s,*t;if(s==t){t=(s=buf)+fread(buf,1,1<<16,stdin);if(s==t)return EOF;}return *s++;}
#define getchar gc
ll read()
{
	char c;
	ll w=1;
	while((c=getchar())>'9'||c<'0')if(c=='-')w=-1;
	ll ans=c-'0';
	while((c=getchar())>='0'&&c<='9')ans=(ans<<1)+(ans<<3)+c-'0';
	return ans*w;
}
void pc(char c,int op)
{
	static char buf[1<<16],*s=buf,*t=buf+(1<<16);
	(op||((*s++=c)&&(s==t)))&&(fwrite(buf,1,s-buf,stdout),s=buf);
}
void wt(int x)
{
	if(x>9)wt(x/10);
	pc('0'+x%10,0);
}
void wts(int x,char op)
{
	if(x<0)pc('-',0),x=-x;
	wt(x),pc(op,0);
}
char ST;
const int xx=1e5+5;
int mn[xx],prim[xx],cnt;
int mx[xx],a[xx];
LL bit[xx];
void pre(int N)
{
	for(int i=2;i<=N;i++)
	{
		if(!mn[i])
		{
			mn[i]=i,prim[++cnt]=i,mx[i]=i;
			if(i<=316)bit[i]=LL(1)<<(cnt-1);
		}
		for(int j=1;j<=cnt;j++)
		{
			if(i*prim[j]>=xx)break;
			mn[i*prim[j]]=prim[j];
			mx[i*prim[j]]=mx[i];
			bit[i*prim[j]]=bit[i]|bit[prim[j]];
			if(i%prim[j]==0)break;
		}
	}
}
LL s[xx<<2];
void build(int k,int l,int r)
{
	if(l==r)return s[k]=bit[a[l]],void();
	int mid=l+r>>1;
	build(k<<1,l,mid);
	build(k<<1|1,mid+1,r);
	s[k]=s[k<<1]|s[k<<1|1];
}
LL ask(int k,int l,int r,int x,int y)
{
	if(x<=l&&r<=y)return s[k];
	int mid=l+r>>1;
	if(y<=mid)return ask(k<<1,l,mid,x,y);
	if(mid<x)return ask(k<<1|1,mid+1,r,x,y);
	return ask(k<<1,l,mid,x,y)|ask(k<<1|1,mid+1,r,x,y);
}
#define ii array<int,2>
vector<ii>v[xx];
const int B=316;
int L[xx],R[xx],bel[xx],n,C,q;
//120Mb
int sf[xx][B+2],bk[B+5][B+2];
void inc(int id,int x,int op)
{
	for(int i=x;i>=L[bel[x]];i--)sf[i][id]+=op;
	for(int i=bel[x];i>=1;i--)bk[i][id]+=op;
}
int ask(int id,int x){return sf[x][id]+bk[bel[x]+1][id];}
int lst[xx],val[xx],ans[xx];
map<LL,int>mp;
//vector<int>P;
int re,P[xx],cts;
void dfs(int x,int y,int z)
{
	if(x==cts)
	{
		re+=(C/y)*z;
		return;
	}
	dfs(x+1,y,z);
	if(y*P[x]<=C)dfs(x+1,y*P[x],-z);
}
//void dfs(int x,int y,int z)
//{
//	if(!y)return;
//	if(x==siz(P))
//		return re+=y*z,void();
//	dfs(x+1,y,z);
//	dfs(x+1,y/P[x],-z);
//}
int get(LL B)
{
	if(mp.count(B))return mp[B];
//	P.clear();
	cts=0;
	for(int i=0;i<66;i++)
		if(B>>i&1)P[cts++]=(prim[i+1]);
	reverse(P,P+cts);
	re=0;
//	dfs(0,C,1);
	dfs(0,1,1);
	return (mp[B]=re);
}
char ED;
int main(){
	cerr<<abs(&ST-&ED)/1024.0/1024<<"\n";
	#ifdef AAA
	system("ulimit -s 524288");
	cerr<<" stack is on \n";
	#endif 
//	freopen("a.in","r",stdin);
//	freopen("a.out","w",stdout);
	pre(1e5);
	n=read(),read(),C=read(),q=read();
	for(int i=1;i<=n;i++)a[i]=read();
	build(1,1,n);
	
	for(int i=1;i<=q;i++)
	{
		int l=read(),r=read();
		v[r].push_back({l,i});
	}
	//Osqrt - O1 求后缀和。
	for(int i=1;i<=n;i++)bel[i]=(i-1)/B+1,R[bel[i]]=i,(!L[bel[i]]?L[bel[i]]=i:0);
	for(int i=1;i<=C;i++)val[mx[i]]++;
	for(int i=1;i<=n;i++)
	{
		if(mx[a[i]]>316)
		{
			int M=mx[a[i]];
			if(lst[M])
				inc(C/M,lst[M],-1),inc(317,lst[M],-val[M]);
			lst[M]=i;
			inc(C/M,lst[M],1),inc(317,lst[M],val[M]);
		}
		for(auto it:v[i])
		{
			int l=it[0],bl=bel[l]+1;
			LL B=ask(1,1,n,l,i);
			int an=get(B)-ask(317,it[0]);
			int cs=0;
			for(int j=1;j<=316;j++)
			{
				if(bit[j]&B)++cs;
				an+=cs*(sf[l][j]+bk[bl][j]);
			}
			ans[it[1]]=an;
		}
	}
	for(int i=1;i<=q;i++)cout<<ans[i]<<"\n";
	//~ cerr<<cnt<<"#\n";
	//~ cerr<<prim[cnt-1]<<"#\n";
	//7732，单次。
	//14920 单次，为什么这也敢上a?????
	pc('1',1);
	return 0;
}

}signed main(){return ifzw::main();}

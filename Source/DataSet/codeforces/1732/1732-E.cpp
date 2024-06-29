// LUOGU_RID: 138669937
#include<bits/stdc++.h>
#define ll unsigned int
#define F(i,l,r) for(int i=l,i##end=r;i<=i##end;++i)
#define G(i,l,r) for(int i=l,i##end=r;i>=i##end;--i)
using namespace std;
namespace Fread{const int SIZE=1<<16;char buf[SIZE],*S,*T;inline char getchar(){if(S==T){T=(S=buf)+fread(buf,1,SIZE,stdin);if(S==T)return'\n';}return *S++;}}namespace Fwrite{const int SIZE=1<<16;char buf[SIZE],*S=buf,*T=buf+SIZE;inline void flush(){fwrite(buf,1,S-buf,stdout);S=buf;}inline void putchar(char c){*S++=c;if(S==T)flush();}struct NTR{~NTR(){flush();}}ztr;}
#define getchar Fread::getchar
#define putchar Fwrite::putchar
#define Setprecision 10
#define between ' '
template<typename T>struct is_char{static constexpr bool value=(std::is_same<T,char>::value||std::is_same<T,signed char>::value||std::is_same<T,unsigned char>::value);};template<typename T>struct is_integral_ex{static constexpr bool value=(std::is_integral<T>::value||std::is_same<T,__int128>::value)&&!is_char<T>::value;};template<typename T>struct is_floating_point_ex{static constexpr bool value=std::is_floating_point<T>::value||std::is_same<T,__float128>::value;};namespace Fastio{struct Reader{template<typename T>typename std::enable_if_t<std::is_class<T>::value,Reader&>operator>>(T&x){for(auto &y:x)*this>>y;return *this;}template<typename T>typename std::enable_if_t<is_integral_ex<T>::value,Reader&>operator>>(T&x){char c=getchar();short f=1;while(c<'0'||c>'9'){if(c=='-')f*=-1;c=getchar();}x=0;while(c>='0'&&c<='9'){x=(x<<1)+(x<<3)+(c^48);c=getchar();}x*=f;return *this;}template<typename T>typename std::enable_if_t<is_floating_point_ex<T>::value,Reader&>operator>>(T&x){char c=getchar();short f=1,s=0;x=0;T t=0;while((c<'0'||c>'9')&&c!='.'){if(c=='-')f*=-1;c=getchar();}while(c>='0'&&c<='9'&&c!='.')x=x*10+(c^48),c=getchar();if(c=='.')c=getchar();else return x*=f,*this;while(c>='0'&&c<='9')t=t*10+(c^48),s++,c=getchar();while(s--)t/=10.0;x=(x+t)*f;return*this;}template<typename T>typename std::enable_if_t<is_char<T>::value,Reader&>operator>>(T&c){c=getchar();while(c=='\n'||c==' '||c=='\r')c=getchar();return *this;}Reader&operator>>(char*str){int len=0;char c=getchar();while(c=='\n'||c==' '||c=='\r')c=getchar();while(c!='\n'&&c!=' '&&c!='\r')str[len++]=c,c=getchar();str[len]='\0';return*this;}Reader&operator>>(std::string&str){str.clear();char c=getchar();while(c=='\n'||c==' '||c=='\r')c=getchar();while(c!='\n'&&c!=' '&&c!='\r')str.push_back(c),c=getchar();return*this;}Reader(){}}cin;const char endl='\n';struct Writer{typedef __int128 mxdouble;template<typename T>typename std::enable_if_t<std::is_class<T>::value,Writer&>operator<<(T x){for(auto &y:x)*this<<y<<between;*this<<'\n';return *this;}template<typename T>typename std::enable_if_t<is_integral_ex<T>::value,Writer&>operator<<(T x){if(x==0)return putchar('0'),*this;if(x<0)putchar('-'),x=-x;static int sta[45];int top=0;while(x)sta[++top]=x%10,x/=10;while(top)putchar(sta[top]+'0'),--top;return*this;}template<typename T>typename std::enable_if_t<is_floating_point_ex<T>::value,Writer&>operator<<(T x){if(x<0)putchar('-'),x=-x;mxdouble _=x;x-=(T)_;static int sta[45];int top=0;while(_)sta[++top]=_%10,_/=10;if(!top)putchar('0');while(top)putchar(sta[top]+'0'),--top;putchar('.');for(int i=0;i<Setprecision;i++)x*=10;_=x+0.5;while(_)sta[++top]=_%10,_/=10;for(int i=0;i<Setprecision-top;i++)putchar('0');while(top)putchar(sta[top]+'0'),--top;return*this;}template<typename T>typename std::enable_if_t<is_char<T>::value,Writer&>operator<<(T c){putchar(c);return*this;}Writer&operator<<(char*str){int cur=0;while(str[cur])putchar(str[cur++]);return *this;}Writer&operator<<(const char*str){int cur=0;while(str[cur])putchar(str[cur++]);return*this;}Writer&operator<<(std::string str){int st=0,ed=str.size();while(st<ed)putchar(str[st++]);return*this;}Writer(){}}cout;}
#define cin Fastio::cin
#define cout Fastio::cout
#define endl Fastio::endl
namespace GC
{
	const int N=5000;
	const int M=50000;
	const int T=1000;
	int pre[T+2][T+2];
	int a[N+2],b[N+2];
	int fac[M+2][3];
	bool isp[M+2];
	int pri[M/10],tot;
	int n;
	void work()
	{
		fac[1][0]=fac[1][1]=fac[1][2]=1;
		F(i,2,M)
		{
			if(!isp[i])fac[i][0]=fac[i][1]=1,fac[i][2]=i,pri[++tot]=i;
			for(int j=1;pri[j]*i<=M;++j)
			{
				int tmp=pri[j]*i;
				isp[tmp]=true;
				fac[tmp][0]=fac[i][0]*pri[j];
				fac[tmp][1]=fac[i][1];
				fac[tmp][2]=fac[i][2];
				if(fac[tmp][0]>fac[tmp][1])
					fac[tmp][0]^=fac[tmp][1]^=fac[tmp][0]^=fac[tmp][1];
				if(fac[tmp][1]>fac[tmp][2])fac[tmp][1]^=fac[tmp][2]^=fac[tmp][1]^=fac[tmp][2];
				if(i%pri[j]==0)break;
			}	
		}		
		F(i,0,T)pre[0][i]=pre[i][0]=i;
		F(i,1,T)F(j,1,i)pre[i][j]=pre[j][i]=pre[j][i%j];
	}
	int gcd(int a,int b)
	{
		int ans=1;
		F(i,0,2)
		{
			int tmp=(fac[a][i]>T)?(b%fac[a][i]?1:fac[a][i]):pre[fac[a][i]][b%fac[a][i]];
			b/=tmp;ans*=tmp;
		}
		return ans;
	}
}
constexpr int w=1000,MK=5e4;
int st[w+9][MK+9];
ll ans[w+9][MK+9];int L[w+9],R[w+9];
ll ca[MK+9];
int a[MK+9],b[MK+9];
int tag[w+9];
ll qq[w+9];
ll cal(int x,int y)
{
	ll g=GC::gcd(x,y);
   	return (ll)x/g*y/g;
}
void modify(int cc,int l,int r,int x)
{
   	qq[cc]=3e9;
	F(i,L[cc],R[cc])
	{
		if(tag[cc]) a[i]=tag[cc];
		if(i>=l && i<=r) a[i]=x;
		ca[i]=cal(a[i],b[i]);
	   	qq[cc]=min(qq[cc],ca[i]);
	}
	tag[cc]=0;
}
int main()
{
	GC::work();
	ll n,q;
	F(i,1,w)F(j,1,MK) ans[i][j]=3e9,st[i][j]=1e6;
	cin>>n>>q;
	F(i,1,w) L[i]=(i-1)*n/w+1,R[i]=i*n/w;
	F(i,1,n) cin>>a[i];
	F(i,1,n) cin>>b[i];
	F(cc,1,w)
	{
	   	F(j,L[cc],R[cc])st[cc][b[j]]=b[j];
		F(i,1,MK)F(j,2,MK/i){if(st[cc][i]<=MK)break;st[cc][i]=st[cc][i*j];}
		F(i,1,MK)if(st[cc][i]<=MK)F(j,1,MK/i){ans[cc][i*j]=min(ans[cc][i*j],(ll)st[cc][i]/i*j);}
	}
	F(cc,1,w)
	{
		qq[cc]=3e9;
	   	F(j,L[cc],R[cc])
		   	ca[j]=cal(a[j],b[j]),qq[cc]=min(qq[cc],ca[j]);
	}
	F(i,1,q)
	{
		int op,x,y,z;
		cin>>op>>x>>y;
		if(op==1)cin>>z;
		if(op==1)
		{
		   	F(cc,1,w)
			{
			   	if(x<=L[cc] && R[cc]<=y) {tag[cc]=z,qq[cc]=ans[cc][z];continue;}
				else if(y<L[cc] || x>R[cc]) continue;
				modify(cc,x,y,z);
			}
		}
		else
		{
			ll out=3e9;
		   	F(cc,1,w)
			{
			   	if(x<=L[cc] && R[cc]<=y){out=min(out,qq[cc]);continue;}
				else if(y<L[cc] || x>R[cc]) continue;
				F(j,max(L[cc],x),min(R[cc],y)) if(!tag[cc])out=min(out,ca[j]);else out=min(out,cal(tag[cc],b[j]));
			}
			cout<<out<<endl;
		}
	}
	return 0;
}

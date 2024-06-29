// LUOGU_RID: 149221953
#include<algorithm>
#include<iostream>
#include<cstring>
#include<cstdio> 
#include<cmath>
using namespace std;
#define int long long
const int mod=1e9+7;
const int N=105;
struct ss
{
	int x,y;
};
ss p[N];
int dp[N][N][2][2];
int dian[N][N][N];
int p2[N];
int cj(ss a,ss b,ss base)
{
	return (a.x-base.x)*(b.y-base.y)-(a.y-base.y)*(b.x-base.x);
}
bool cmp(ss a,ss b)
{
	if(a.x^b.x)
	return a.x<b.x;
	return a.y<b.y;
}
signed main()
{
	int n;
	cin>>n;
	p2[0]=1;
	for(int i=1;i<=n;i++)
	{
		p2[i]=p2[i-1]*2%mod;
	}
	for(int i=1;i<=n;i++)
	{
		cin>>p[i].x>>p[i].y;
	}
	sort(p+1,p+1+n,cmp);
	for(int i=1;i<=n;i++)
	{
		for(int j=1;j<=n;j++)
		{
			for(int k=1;k<=n;k++)
			{
				if(i==j || i==k || j==k)
				continue; 
				for(int l=1;l<=n;l++)
				{
					if(l!=i && l!=j && l!=k && abs(cj(p[k],p[j],p[i]))==abs(cj(p[j],p[l],p[i]))+abs(cj(p[l],p[k],p[i]))+abs(cj(p[k],p[j],p[l])))//三角形面积等于三个同顶点的三角形面积之和，说明这个点在三角形内部 
					dian[i][j][k]++; 
				}
			}
		}
	} 
	int ans=0;
	for(int P=n-1;P;P--) 
	{
		memset(dp,0,sizeof(dp));
		for(int i=P+1;i<=n;i++)
		{
			dp[P][i][0][0]=dp[P][i][0][1]=1;
		}
		for(int i=P;i<=n;i++)
		{
			for(int j=P;j<=n;j++)
			{
				for(int k=0;k<=1;k++)
				{
					for(int l=j+1;l<=n;l++)
					{ 
						int chaji=abs(cj(p[j],p[l],p[P]));
						if(cj(p[l],p[i],p[j])>0)
						{
							dp[j][l][(k+chaji)%2][0]+=dp[i][j][k][0]*p2[dian[P][j][l]]%mod,dp[j][l][(k+chaji)%2][0]%=mod;
//							cout<<dp[j][l][(k+chaji)%2][0]<<endl;
						}
						else
						{
							dp[j][l][(k+chaji)%2][1]+=dp[i][j][k][1]*p2[dian[P][j][l]]%mod,dp[j][l][(k+chaji)%2][1]%=mod;
//							cout<<dp[j][l][(k+chaji)%2][1]<<endl;
						}
					}
				}
			}
		}
		for(int j=P+1;j<=n;j++)
		{
			for(int k=0;k<=1;k++)
			{
				int ans1=0,ans2=0;
				for(int i=P;i<j;i++)
				{
					ans1+=dp[i][j][k][0],ans1%=mod;
					ans2+=dp[i][j][k][1],ans2%=mod;
				}
				ans+=ans1*ans2%mod;
				ans%=mod;
			}
		}
	} 
	int r=n*(n-1)/2;
	cout<<(ans-r+mod)%mod;
	return 0;
}
#include<cstdio>
#include<algorithm>
using namespace std;

#define ll long long int
int n,mod,cnt=-1;
int inc(int a,int b)
{
return (a+b<mod)?a+b:a+b-mod;
}
int dec(int a,int b)
{
return (a<b)?a-b+mod:a-b;
}
int mul(int a,int b)
{
return (int)((ll)a*b%mod);
}
int pow(int a,int b)
{
int res=1;
while(b>0){
if(b&1) res=mul(res,a);
a=mul(a,a); b>>=1;
}
return res;
}
int inv(int x)
{
return pow(x,mod-2);
}

int fir[2002],to[4004];
int deg[2002],nxt[4004];
int dp[2002][2002];
int leq[2002][2002];
int pre[2002][2002];
int suf[2002][2002];
void add(int a,int b)
{
to[++cnt]=b;
nxt[cnt]=fir[a];
fir[a]=cnt;
deg[b]++;
return;
}
int son[2002],val[2002];
int ans[2002],arr[2002];
int C[2002][2002];
void dfs(int i,int fa)
{
if(deg[i]==1&&i!=0){
for(int j=0;j<n;j++){
dp[i][j]=1;
leq[i][j]=j+1;
}
return;
}
for(int j=fir[i];j!=-1;j=nxt[j])
if(to[j]!=fa)
dfs(to[j],i);
int scnt=0;
for(int j=fir[i];j!=-1;j=nxt[j])
if(to[j]!=fa){
son[scnt]=to[j]; scnt++;
}
for(int k=0;k<n;k++){
int P=1,S=1;
for(int w=0;w<scnt;w++){
pre[son[w]][k]=P;
P=mul(P,leq[son[w]][k]);
}
for(int w=scnt-1;w>=0;w--){
suf[son[w]][k]=S;
S=mul(S,leq[son[w]][k]);
}
}
for(int w=0;w<scnt;w++){
for(int j=0;j<n;j++){
val[j]=mul(pre[son[w]][j],suf[son[w]][j]);
if(j>0) val[j]=inc(val[j],val[j-1]);
}
for(int j=1;j<n;j++)
dp[i][j]=inc(dp[i][j],mul(val[j-1],dp[son[w]][j]));
}
for(int j=0;j<n;j++){
int mult=1;
for(int w=0;w<scnt;w++)
mult=mul(mult,leq[son[w]][j]);
dp[i][j]=inc(dp[i][j],mult);
leq[i][j]=dp[i][j];
}
for(int j=1;j<n;j++)
leq[i][j]=inc(leq[i][j],leq[i][j-1]);
return;
}

int main(void)
{
scanf("%d%d",&n,&mod);
for(int i=0;i<n;i++)
fir[i]=-1;
for(int i=1;i<n;i++){
int a,b;
scanf("%d%d",&a,&b);
add(a-1,b-1); add(b-1,a-1);
}
dfs(0,-1);
for(int i=0;i<n;i++){
arr[i]=1;
for(int j=fir[0];j!=-1;j=nxt[j])
arr[i]=mul(arr[i],leq[to[j]][i]);
}
for(int i=n-1;i>0;i--)
arr[i]=dec(arr[i],arr[i-1]);
for(int i=0;i<n;i++){
C[i][0]=C[i][i]=1;
for(int j=1;j<i;j++)
C[i][j]=inc(C[i-1][j],C[i-1][j-1]);
}
for(int i=0;i<n;i++){
ans[i]=arr[i];
for(int j=0;j<i;j++)
ans[i]=dec(ans[i],mul(ans[j],C[i][j]));
}
for(int i=0;i<n-1;i++)
printf("%d%c",ans[i],i==n-2?'\n':' ');
return 0;
}
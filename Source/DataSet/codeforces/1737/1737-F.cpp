#include<bits/stdc++.h>
#define rp(i,a,b) for(int i=a,_=b;i<=_;++i)
#define pr(i,a,b) for(int i=a,_=b;i>=_;--i)
#define sz(x) int((x).size())
using namespace std;
typedef unsigned int ui;
int T,m,c[5],b[20],p[20],r[20];
int main(){
scanf("%d",&T);
while(T--){
scanf("%d",&m);
memset(c,0,sizeof(c));
rp(i,0,m-1)scanf("%d",&b[i]),++c[min(4,b[i])];
if(c[4]||c[3]>1||c[2]>2||(c[2]&&c[3])){
puts("-1");continue;
}
rp(i,0,m-1)p[i]=i;
sort(p,p+m,[&](int x,int y){return b[x]<b[y];});
vector<ui>v,u[2];
if(c[3]){
v=vector<ui>({2,1,3});
}else if(c[2]==1){
v=vector<ui>({1,2});
}else if(c[2]==2){
v=vector<ui>({10,4,6,8,5,2,9,1});
}else if(c[1]==1){
v=vector<ui>({1}),--c[1];
}else{
v=vector<ui>({1,5,4}),c[1]-=2;
}
rp(i,1,c[1]){
rp(k,0,1){
u[k]=v;
rp(j,0,sz(v)-1){
u[k][j]=(v[j]<<2)|(k^(j&1));
}
}
int k=u[1].back()&1;
u[1].insert(u[1].begin(),u[k].back());
u[k].pop_back();
u[1].insert(u[1].begin()+1,1);
reverse(u[0].begin(),u[0].end());
v=u[1];
for(ui x:u[0])v.push_back(x);
}
rp(i,0,sz(v)-1){
rp(j,0,m-1)r[p[j]]=v[i]>>(2*j)&3;
rp(j,0,m-1)printf("%d%c",r[j]," \n"[j==m-1]);
}
}
return 0;
}

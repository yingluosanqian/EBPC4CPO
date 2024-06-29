// LUOGU_RID: 149302484
#include <bits/stdc++.h>
const int N = 1e6 + 10;
const int mod = 1e9 + 7;
const int INF = 0x3f3f3f3f;
using namespace std;
int n,k;
char s[10][10];
int dx[4]={0,0,1,-1},dy[4]={-1,1,0,0};
int ans;
void dfs(int cnt){
    if(cnt==0){
        ans++;
        return;
    }
    vector<pair<int, int> > vis;
    for(int i=1;i<=n;++i)
        for(int j=1;j<=n;++j)
            if(s[i][j]=='.'){
                bool flag=false;
                for(int t=0;t<4;++t){
                    int tx=i+dx[t],ty=j+dy[t];
                    if(tx>=1 && tx<=n && ty>=1 && ty<=n && s[tx][ty]=='@')
                        flag=true;
                }
                if(flag){
                    s[i][j]='@';
                    dfs(cnt-1);
                    s[i][j]='#';
                    vis.push_back({i,j});
                }
            }
    for(auto it:vis) s[it.first][it.second] = '.';
}
int main() {
    scanf("%d %d",&n,&k);
    for(int i=1;i<=n;++i)scanf("%s", s[i] + 1);
    for(int i=1;i<=n;++i)
        for(int j=1;j<=n;++j)
            if(s[i][j]=='.')
                s[i][j]='@',
                dfs(k-1),
                s[i][j]='#';
    printf("%lld\n",ans);
    return 0;
}

// LUOGU_RID: 149211452
#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
typedef pair<LL, LL> PLL;
const LL INF=4e18;

int main(){
#if 1
   ios::sync_with_stdio(0);
   cin.tie(0);
   cout.tie(0);
#endif
   LL n; 
   cin >> n;
   vector<PLL> v(n);
   for (LL i = 0; i < n; i++) {
      cin >> v[i].first >> v[i].second;
   }
   sort(v.begin(), v.end());

   LL ok = 0, ng = INF;
   while(ng - ok > 1){
      LL md = (ok + ng)/2;
      queue<PLL> que;
      bool able = false;
      LL mi = INF, ma = 0;
      for (auto p:v){
         while(!que.empty()){
            if(que.front().first > p.first - md) {
               break;
            }
            mi = min(mi, que.front().second); 
            ma = max(ma, que.front().second);
            que.pop();
         }
         if(mi <= p.second - md || ma >= p.second + md) {
            able=true;
         }
         que.push(p);
      }
      if(able) {
         ok = md;
      } else {
         ng = md;
      }
   }
   cout << ok << endl;
   return 0;
}


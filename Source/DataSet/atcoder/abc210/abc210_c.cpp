// LUOGU_RID: 149295098
#include<bits/stdc++.h>
using namespace std;
int N, K, c[300005], tot;
map<int, int>mp;
int main(){
	cin >> N >> K;
	for(int i = 1; i <= K; i++){
		cin >> c[i];
		mp[c[i]]++;
	}
	int ans = mp.size();
	for(int i = 1; i + K <= N; i++){
		cin >> c[i + K];
		mp[c[i + K]]++;
		mp[c[i]]--;
		if(mp[c[i]] == 0)mp.erase(c[i]);
		ans = max(ans, (int)mp.size());
	}
	cout << ans;
}
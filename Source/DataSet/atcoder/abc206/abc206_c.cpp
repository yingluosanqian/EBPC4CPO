#include <bits/stdc++.h>
using namespace std;
#define ll long long
/* 1. Can I use Maths ? 
   2. Prefix Sum ? 
   3. Binary Search ? 
   4. Map ?
*/

ll Nc2(ll n){
	return (n*(n-1))/2;
}

void solve(){
	int n; cin >> n;
	vector<ll> arr(n);
	map<ll,ll> temp;
	for(int i = 0 ; i < n ; i++){
		cin >> arr[i];
	}
	for(int i = 0 ; i < n ; i++)
		temp[arr[i]]++;
	ll cnt = 0;
	for(auto x : temp){
		cnt += Nc2(x.second);
	}
	cout << Nc2(n) - cnt << endl;
}

int main(){
	ios_base::sync_with_stdio(false);
	cin.tie(NULL);
	ll q = 1; 
	// cin >> q;
	while(q--){
		solve();
	}
    return 0;
}

#include <bits/stdc++.h>
#define fer(i,a,b) for(re i = a ; i <= b ; ++ i)
#define der(i,a,b) for(re i = a ; i >= b ; -- i)
#define de(x) cout << x << "\n"
#define sf(x) scanf("%lld",&x)
#define pll pair<int,int>
#define re register int
#define int long long
#define pb push_back
#define y second
#define x first
using namespace std;
const int N = 1e6 + 10, M = 2010, inf = 0x3f3f3f3f, mod = 1e9 + 7 ;
map<string, int> q ;

signed main() {
	string x ;

	q["ABC"] ++ ;
	q["ARC"] ++ ;
	q["AGC"] ++ ;
	q["AHC"] ++ ;

	for (int i = 1 ; i <= 3 ; i ++) {
		cin >> x ;
		q[x] ++ ;
	}

	for (auto i : q) {
		if (i.y == 1) {
			cout << i.x << "\n" ;
			break ;
		}
	}
	return 0;
}
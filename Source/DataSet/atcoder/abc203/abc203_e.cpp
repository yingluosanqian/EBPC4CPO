// LUOGU_RID: 148789855
#include<bits/stdc++.h>
using namespace std;
map<int, vector<int>>a;
set<int>s;
int n, m;
int main() {
	ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);
	cin >> n >> m;
	for (int i = 1, x, y; i <= m; i++)cin >> x >> y, a[x].push_back(y);
	s.insert(n);
	for (auto& i : a) {
		vector<int>A, B;
		for (int j : i.second)
			if (s.count(j))A.push_back(j);
		for (int j : i.second)
			if (s.count(j - 1) || s.count(j + 1))B.push_back(j);
		for (int j : A)s.erase(j);
		for (int j : B)s.insert(j);
	}
	cout << s.size() << '\n';
	return 0;
}
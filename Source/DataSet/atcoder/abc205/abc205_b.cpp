#include <bits/stdc++.h>
using namespace std;
#define endl '\n'
#define sz(x) ((int)x.size())
#define all(x) (x).begin(), (x).end()
#define ll long long
#define pi pair<int, int>
#define pll pair<ll, ll>
#define Fi first
#define Se second

const int N = 1e3 + 5;
int a[N];

void answer() {
    int n;
    cin >> n;
    for (int i = 1; i <= n; ++i) {
        cin >> a[i];
    }
    sort(a + 1, a + n + 1);
    for (int i = 1; i <= n; ++i) {
        if (a[i] != i) {
            cout << "No" << endl;
            return;
        }
    }
    cout << "Yes" << endl;
}   

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    int tt = 1;
    // cin >> tt;
    for (int i = 1; i <= tt; ++i) {
        answer();   
    }
    return 0;
}
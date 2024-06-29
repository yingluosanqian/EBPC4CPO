#include <bits/stdc++.h>
using namespace std;
using i64 = long long;

constexpr int N = 2e5 + 5;

void solve() {
    string s;
    cin >> s;

    if (s == string(4, s[0])) { // "7777"
        cout << "Weak\n";
    } else {
        if (s[1] == s[0] + 1 and s[2] == s[1] + 1 and s[3] == s[2] + 1) { // "1234"
            cout << "Weak\n";
        } else {
            if (s == "9012" or s == "8901" or s == "7890") {
                cout << "Weak\n";
            } else {
                cout << "Strong\n";
            }
        }
    }
}

signed main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int T = 1;
    // cin >> T;
    
    while (T--) {
        solve();
    }

    return 0;
}
#include <chrono>
#include <vector>
#include <iostream>
#include <set>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <bitset>
#include <iomanip>
#include <stack>

using namespace std;

#define ll long long
#define all(x) (x).begin(), (x).end()
#define mp make_pair
#define pb emplace_back
#define fi first
#define se second

const int INF = 1e9 + 10;
const ll INFLL = 1e18;

int main() {
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    string s;
    cin >> s;
    int ans = 0;
    for (int a = 0; a < 10; a++) {
        for (int b = 0; b < 10; b++) {
            for (int c = 0; c < 10; c++) {
                for (int d = 0; d < 10; d++) {
                    bool ok = true;
                    for (int i = 0; i < 10; i++) {
                        if (s[i] == 'o') {
                            if (a != i && b != i && c != i && d != i) {
                                ok = false;
                            }
                        } else if (s[i] == 'x') {
                            if (a == i || b == i || c == i || d == i) {
                                ok = false;
                            }
                        }
                    }
                    if (ok) {
                        ans++;
                    }
                }
            }
        }
    }
    cout << ans << "\n";
    return 0;
}

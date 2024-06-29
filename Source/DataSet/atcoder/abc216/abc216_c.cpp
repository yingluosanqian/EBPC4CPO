#include <bits/stdc++.h>
#define rep(i, n) for (int i = 0; i < (n); i++)
using namespace std;
using ll = long long;
using P = pair<int, int>;

int main()
{
    ll n, x = 0;
    cin >> n;
    string ans;
    while (n > 0)
    {
        if (n % 2)
        {
            ans += 'A';
            n--;
        }
        else
        {
            ans += 'B';
            n /= 2;
        }
    }
    reverse(ans.begin(), ans.end());
    cout << ans << endl;
    return 0;
}
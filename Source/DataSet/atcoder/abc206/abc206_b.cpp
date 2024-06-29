#include <bits/stdc++.h>
using namespace std;
#define endl '\n'
typedef long long lln;

int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cin.tie(0);
    lln n;
    cin >> n;
    lln savings = 0;
    for (lln i = 1;; i++)
    {
        savings += i;
        if(savings>=n){
            cout << i << endl;
            return 0;
        }
    }
    return 0;
}
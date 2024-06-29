// LUOGU_RID: 149301725
#include <bits/stdc++.h>
using namespace std;
int main() {
    string a;
    cin >> a;
    reverse(a.begin(), a.end());
    for (int i = 0; i < a.size(); i++)
    {
        if (a[i] == '6')
            putchar('9');
        else if (a[i] == '9')
            putchar('6');
        else
            putchar(a[i]);
    }
    return 0;
}
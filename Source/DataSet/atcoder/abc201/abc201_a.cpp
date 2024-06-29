#include <bits/stdc++.h>
using namespace std;
 
int main(){
    ios_base::sync_with_stdio(0); cin.tie(0);
    int a,b,c; cin >>a>>b>>c;
    int minn = min(c,(min(a,b)));
    int makss = max(c,(max(a,b)));
    int tengah = (a+b+c)-(minn+makss);

    if((tengah - minn == makss - tengah)){
        cout << "Yes";
    } 
    else if(minn==makss && minn==tengah){
        cout << "Yes";
    }
    else {
        cout << "No";
    }
    return 0;
}
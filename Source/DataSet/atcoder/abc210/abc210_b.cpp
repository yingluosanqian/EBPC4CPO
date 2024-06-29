#include <bits/stdc++.h>
using namespace std;
int main(){
    int a; cin >> a;
    string b;
    cin >> b;
    for (int i=0; i<b.size() ; i++){
        if ( b[i]== '1'){
            if (i%2 == 0){
                cout << "Takahashi";
                break;
            }
            else{
                cout << "Aoki";
                break;
            }
        }
    }
}

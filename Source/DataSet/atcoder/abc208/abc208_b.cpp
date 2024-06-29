#include <bits/stdc++.h>
#define rep(i,a,b) for(int i=a; i<b; i++)
using namespace std;
typedef long long ll;
typedef unsigned long long ull;

int main(void){
    int P;
    cin >> P;
    
    int c=0;
    for(int i=10; i>=1; i--) {
        int N = 1;
        for(int j=i; j>=1; j--) {
            N*=j;
        }
        
        if(P%N==0) {
            c+=P/N;
            break;
        } else {
            c+=P/N;
            P=P%N;
        }
    }
    cout << c << endl;
}
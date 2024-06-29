#include <iostream>
#include<cmath>
using namespace std;
int main() {
  long long x,y,z,res1,res2; 
  cin>>x>>y>>z;
  if(z%2==0){
      x=abs(x);
      y=abs(y);
  }
//  res1=pow(x,z),res2=pow(y,z);
  if(x>y)cout<<">"<<"\n";
  else if (x<y)cout<<"<"<<"\n";
  else cout<<"="<<"\n";
    return 0;
}

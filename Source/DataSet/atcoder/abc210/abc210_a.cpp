// LUOGU_RID: 149207384
#include<bits/stdc++.h>
using namespace std;
int main()
{
	int n,x,a,b;
	cin>>n>>x>>a>>b;
	cout<<min(n,x)*a+max(n-x,0)*b;
	cout<<endl;
    return 0;
}
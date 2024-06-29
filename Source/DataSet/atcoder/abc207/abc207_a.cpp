// LUOGU_RID: 149308335
#include<bits/stdc++.h>
using namespace std;
int main()
{
	int a,b,c;
	cin>>a>>b>>c;
	cout<<a+b+c-min(min(a,b),c);
	cout<<"\n";
	return 0;
}
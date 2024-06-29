// LUOGU_RID: 149308043
#include<bits/stdc++.h>
using namespace std;
int main()
{
	int a;
	cin>>a;
	if(int(1.08*a)<206)
		cout<<"Yay!";
	else if(int(1.08*a)==206)
		cout<<"so-so";
	else
		cout<<":(";
	cout<<"\n";
	return 0;
}
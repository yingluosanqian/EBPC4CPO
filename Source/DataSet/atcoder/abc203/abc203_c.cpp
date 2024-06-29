#include <bits/stdc++.h>
using namespace std;

const int maxn=2e5+1;
long n, k;
struct FRIENDS
{
	long a, b;
	FRIENDS(long a=0,long b=0):a(a),b(b){}
}friends[maxn]={};
bool comp(FRIENDS lhs, FRIENDS rhs)
{
	return lhs.a<rhs.a;
}
int main()
{
	cin>>n>>k;
	for (int i=1;i<=n;i++)
		cin>>friends[i].a>>friends[i].b;
	sort(friends+1,friends+n+1,comp);
	long pos=0;
	for (int i=1;i<=n and friends[i].a<=pos+k;i++)
	{
		k=k-(friends[i].a-pos)+friends[i].b;
		pos=friends[i].a;
	}
	cout<<pos+k;
	return 0;
}
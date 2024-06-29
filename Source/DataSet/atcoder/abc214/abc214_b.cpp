// Bismillahir Rahmanir Raheem
/* contest time
    1. No use of mobile (must);
    2. Think about problem not solution.
    3. Find full solution
    4. try to proof or find cases
    5. write  down solution
    6. then code

    practice time
    1. think for straigt half an hour .
    2. try to learn
    3. dont make the solve count
    4. make the learn count
    */
// READ PROBLEM AGAIN Andddd AGAIN.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
#define nl "\n"

#define faster ios_base::sync_with_stdio(false), cin.tie(0), cout.tie(0);

void solve(int t_case)
{
	int s,t;
	cin>>s>>t;
	int x=0;
	for(int i=0; i<=s; i++){
		for(int j=0; j<=s; j++){
			for(int k=0; k<=s; k++){
				{
					if(i+j+k<=s and i*j*k<=t)
						x++;
				}
			}
		}
	}
	cout<<x<<nl;
}
int32_t main()
{
    faster;
    int t=1;
    //cin >> t;
    for (int i = 1; i <= t; i++)
        solve(i);
    return 0;
}

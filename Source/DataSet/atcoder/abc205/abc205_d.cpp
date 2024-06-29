
#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define ld long double
#define intmax INT_MAX
#define intmin INT_MIN
#define PI 3.1415926535897932384626433832795
#define ed "\n"
#define F first
#define S second
#define MP make_pair
#define PB push_back
#define endl "\n"
const long long MOD = 1000000007;

const ll inf = 1e15;
const ll N = 200005;


void print(vector<ll>&arr){
  for(auto &v:arr){
    cout<<v<<" ";
  }
  cout<<endl;
}

int gcd(int a, int b)
{
    if (b == 0)
        return a;
    return gcd(b, a % b);
}
inline ll modadd(ll a, ll b, ll mod = MOD)
{
    return ((a % mod) +(b % mod)) % mod;
}
/*
ll n,m;
ll interactive(ll a,ll b) {
    //cout<<"->"<<a<<",b->"<<b<<endl;
    if(a<1||b<1)return -1;
    if(a>n||b>m)return -1;
    cout << "? " << a << " " << b<<endl;
    cout.flush();
    ll res;
    cin >> res;
    //cout<<"---->"<<res<<endl;
    return res;
}
void printinterative(ll ans1,ll ans2){
  cout<<"! "<<ans1<<" "<<ans2<<endl;
  cout.flush();
}
ll nsum(ll x){
    ll ans=(x*(x+1))/2;
    return ans;
}
*/
void solve(){
    ll n,q;
    cin>>n>>q;
    vector<ll>arr(n);
    for(ll i=0;i<n;i++)cin>>arr[i];
    vector<ll>prefix(n);
    ll current=0;
    vector<ll>last(n);
    for(ll i=0;i<n;i++){
        last[i]=arr[i]-i-1;
    }
    //print(last);
    //print(arr);
    while(q--){
        ll target;
        cin>>target;
        ll index=lower_bound(last.begin(),last.end(),target)-last.begin();
        if(index==n){
            cout<<arr[n-1]+target-last[n-1]<<endl;
        }
        else{
            cout<<arr[index]-(last[index]-target)-1<<endl;
        }
    }
}
//467138328
signed int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    ll t = 1;
    //cin >> t;
    // string temp;getline(cin,temp);
    // fact[0]=1;
    // for(ll i=1;i<100001;i++)fact[i]=fact[i-1]*i;
    for(ll i=1;i<=t;i++)
    {
        //cout<<"Test case #"<<i<<endl;
        solve();
    }
    return 0;
}
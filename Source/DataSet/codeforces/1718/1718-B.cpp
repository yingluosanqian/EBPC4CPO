
///* dont stick to an approach */
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <algorithm>
#include <map>
#include <queue>
#include <set>
#include <array>
#include <unordered_set>
#include <unordered_map>
#include <stack>
#include <cassert>
#include <vector>
#include <numeric>
#include <cstdlib>
#include <bitset>
#include <chrono>
#include <random>
#include <fstream>
#include <iterator>
#include <complex>
#include <cstring>
#include <list>
 //  c++20
 // #include <bit>

using namespace std;

 typedef long long ll;
//typedef int ll;
//typedef unsigned long long int ull;
typedef long double ldb;

#define PB push_back

#define For(i, n) for (ll i = 0; i < n; i++)

#define trav(a,x) for (auto& a : x)

#define PYES cout<<"YES\n"
#define PNO cout<<"NO\n"
#define PYes cout<<"Yes\n"
#define PNo cout<<"No\n"
//#define endl '\n'

#define sq(x)           (x*x)

#define vll             vector<ll>
#define pll             pair<ll,ll>
#define ff              first
#define ss              second
#define rev(v)          reverse(v.begin(),v.end())
#define srt(v)          sort(v.begin(),v.end())
#define grtsrt(v)       sort(v.begin(),v.end(),greater<ll>())
#define mnv(v)          *min_element(v.begin(),v.end())
#define mxv(v)          *max_element(v.begin(),v.end())
#define all(v)          v.begin(),v.end()
#define ACC(v)          accumulate(v.begin(),v.end(),0ll)

#define eps             1e-9

const double PI=acos(-1.0);

#define topi ios::sync_with_stdio(0), cin.tie(0), cout.tie(0)

mt19937 rng((unsigned int)chrono::steady_clock::now().time_since_epoch().count());

//--------------------------------------------functions-------------------------------------------------//

ll power(ll a,ll b){ll result=1;while(b>0){if(b%2 == 1){result *= a;} a *= a;b /= 2;}return result;}
ll gcd(ll x,ll y){
    ll r;
    while(y!=0&&(r=x%y)!=0){
        x=y;y=r;
    }
    return y==0?x:y;
}
ll countSetBits(ll x){ll Count=0;while(x>0){if(x&1) Count++;x=x>>1;}return Count;}
ll mod(ll x,ll M){return ((x%M + M)%M);}
ll add(ll a, ll b,ll M){return mod(mod(a,M)+mod(b,M),M);}
ll mul(ll a, ll b,ll M){return mod(mod(a,M)*mod(b,M),M);}
//ll powerM(ll a,ll b,ll M){
//    ll res=1ll;
//    while(b){
//        if(b%2ll==1ll){
//            res=(a*res)%M;
//        }
//        a=(a*a)%M;b/=2ll;
//    }
//    return res;
//}
ll mod_inv(ll a, ll m) {
    ll g = m, r = a, x = 0, y = 1;
    while (r != 0) {
        ll q = g / r;
        g %= r; swap(g, r);
        x -= q * y; swap(x, y);
    }
    return mod(x,m);
}


////------------------------------------sieve of Eratosthenes-----------------------------------------------//
//ll MAXN=1e7+1000;
//vector<ll> fact(MAXN+1);
//void sieve_of_erantothenes()
//{
//    fact[1] = 1;
//    for (ll i = 2; i <= MAXN; i++)
//        fact[i] = i;
//    for (ll i = 4; i <= MAXN; i += 2)
//        fact[i] = 2;
//    for (ll i = 3; i * i <= MAXN; i++){
//        if (fact[i] == i){
//            for (ll j = i * i; j <= MAXN; j += i)
//                if (fact[j] == j)
//                    fact[j] = i;
//        }
//    }
//}
//----------------------------------------nCr mod------------------------------------------------------//
ll nCr(ll n, ll k){
    if(n<k) return 0;
    if(k==0) return 1;
    ll res = 1;
    if (k > n - k) k = n - k;
    for(ll i = 0; i < k; ++i){
        res *= (n - i);
        res /= (i + 1);
    }
    return res;
}

vector<ll> FA,INV;
void fcalc(ll n,ll M){
    FA.resize(n+1);
    FA[0]=1;
    for(ll i=1;i<=n;i++){
        FA[i]=(FA[i-1]*i)%M;
    }
}

void icalc(ll n,ll M){
    INV.resize(n+1);
    INV[n]=mod_inv(FA[n],M);
    for(ll i=n-1;i>=0;i--){
        INV[i]=(INV[i+1]*(i+1))%M;
    }
}
ll nCrM(ll n,ll r,ll M){
    if(r<0) return 0;
    if(n<r) return 0;
    if(r==0) return 1;
    ll a=FA[n];
    a*=INV[r];
    a%=M;
    a*=INV[n-r];
    a%=M;
    return a;
}
ll rand_int(ll l, ll r){
    return uniform_int_distribution<ll>(l, r)(rng);
}

ifstream fin;
ofstream fout;

template <typename T>
ostream& operator <<(ostream& output, const vector<T>& data)
{
    for (const T& x : data)
        output << x <<" ";
    return output;
}

template<typename T>
istream& operator>>(istream& input,vector<T>& data){
    for (auto& item : data) {
        input >> item;
    }
    return input;
}
//--------------------------------------------solve----------------------------------------------//
void solve(){
//    ll n;cin>>n;
//    vll vv(3,1);
//    vv[0]=0;
//    for(ll i=3;i<=45;i++){
//        vv.PB(vv[i-1]+vv[i-2]);
//    }
//    cout<<vv[45]<<endl;
    ll n;cin>>n;
    vector<ll> v(n);
    cin>>v;
    if(n>45){
        PNO;
        return;
    }else{
        ll sum=ACC(v);
        vll vv(3,1);
        vv[0]=0;
        ll ind=-1;
        ll sx=0;
        for(ll i=3;i<=45;i++){
            vv.PB(vv[i-1]+vv[i-2]);
        }
        for(ll i=1;i<=45;i++){
            sx+=vv[i];
            if(sx==sum){
                ind=i;
            }
        }
        if(ind==-1){
            PNO;
            return;
        }
        ll f=0;
        function<void(vll,ll,ll)> rec=[&](vll me,ll jh,ll pos){
            if(jh==0){
                f=1;
                return ;
            }else if(f==0){
                for(ll i=0;i<me.size();i++){
                    if(me[i]>=vv[jh] && pos!=i){
//                        vll me2=me;
//                        me2[i]-=vv[jh];
//                        grtsrt(me2);
//                        rec(me2,jh-1);
                        me[i]-=vv[jh];
                        rec(me,jh-1,i);
                        me[i]+=vv[jh];
                    }
                }
            }
        };
        rec(v,ind,-1);
        if(f){
            PYES;
        }else{
            PNO;
        }
    }
}



int main(){

    topi;

//    freopen("calc.in","r",stdin);
//    freopen("calc.out","w",stdout);

    cout << fixed << setprecision(10);

    ll t=1;

    cin >> t;

    for(ll i=1;i<=t;i++){

//        cout <<"Case #"<<i<<": ";

        solve();

    }

    return 0;
}

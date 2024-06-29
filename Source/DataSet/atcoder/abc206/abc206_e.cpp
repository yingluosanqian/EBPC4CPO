#include <bits/stdc++.h>
using namespace std;
#define overload1(a, ...) a
#define overload2(a, b, ...) b
#define overload3(a, b, c, ...) c
#define overload4(a, b, c, d, ...) d
#define overload5(a, b, c, d, e ...) e
#define overload6(a, b, c, d, e, f ...) f
#define overload7(a, b, c, d, e, f, g ...) g
#define fast_io ios::sync_with_stdio(false); cin.tie(nullptr);
#pragma GCC optimize("Ofast,no-stack-protector,unroll-loops,fast-math")
typedef __int128_t int128;
typedef long long ll;
typedef long double ld;
#define chmin(a,b) a = min(a,b);
#define chmax(a,b) a = max(a,b);
#define bit_count(x) __builtin_popcountll(x)
#define leading_zero_count(x) __builtin_clz(x)
#define trailing_zero_count(x) __builtin_ctz(x)
#define gcd(a,b) __gcd(a,b)
#define lcm(a,b) a / gcd(a,b) * b
#define rep(...) overload4(__VA_ARGS__, rrep, rep1, rep0)(__VA_ARGS__)
#define rep0(n) for(int _ = 0 ; _ < n ; _++)
#define rep1(i,n) for(int i = 0 ; i < n ; i++)
#define rrep(i,a,b) for(int i = a ; i < b ; i++)
#define repi(it,S) for(auto it = S.begin() ; it != S.end() ; it++)
#define pt(a) cout << a << endl;
#define print(...) printall(__VA_ARGS__);
#define debug(a) cout << #a << " " << a << endl;
#define all(a) a.begin(), a.end()
#define endl "\n";
#define v1(T,n,a) vector<T>(n,a)
#define v2(T,n,m,a) vector<vector<T>>(n,v1(T,m,a))
#define v3(T,n,m,k,a) vector<vector<vector<T>>>(n,v2(T,m,k,a))
#define v4(T,n,m,k,l,a) vector<vector<vector<vector<T>>>>(n,v3(T,m,k,l,a))
template<typename T,typename U>istream &operator>>(istream&is,pair<T,U>&p){is>>p.first>>p.second;return is;}
template<typename T,typename U>ostream &operator<<(ostream&os,const pair<T,U>&p){os<<p.first<<" "<<p.second;return os;}
template<typename T>istream &operator>>(istream&is,vector<T>&v){for(T &in:v){is>>in;}return is;}
template<typename T>ostream &operator<<(ostream&os,const vector<T>&v){for(auto it=v.begin();it!=v.end();){os<<*it<<((++it)!=v.end()?" ":"");}return os;}
template<typename T>istream &operator>>(istream&is,vector<vector<T>>&v){for(T &in:v){is>>in;}return is;}
template<typename T>ostream &operator<<(ostream&os,const vector<vector<T>>&v){for(auto it=v.begin();it!=v.end();){os<<*it<<((++it)!=v.end()?"\n":"");}return os;}
template<typename T>ostream &operator<<(ostream&os,const set<T>&v){for(auto it=v.begin();it!=v.end();){os<<*it<<((++it)!=v.end()?" ":"");}return os;}
template<typename T>ostream &operator<<(ostream&os,const multiset<T>&v){for(auto it=v.begin();it!=v.end();){os<<*it<<((++it)!=v.end()?" ":"");}return os;}
template<class... Args> void printall(Args... args){for(auto i:initializer_list<common_type_t<Args...>>{args...}) cout<<i<<" ";cout<<endl;}

ll f(ll l, ll r){
    vector<ll> C(2*r+1,0);
    ll res = 0;
    for(int i = r; i > 1; i--){
        ll y = r / i - (l - 1) / i;
        C[i] = y * y;
        for(int j = 2 * i; j <= r; j += i){
            C[i] -= C[j];
        }
        res += C[i];
        if(i >= l) res -= y * 2 - 1;
    }
    return res;
}

ll g(ll l, ll r){
    ll res = 0;
    rep(i,l,r+1){
        rep(j,l,r+1){
            ll g = gcd(i,j);
            if(g == 1) continue;
            if(g == i) continue;
            if(g == j) continue;
            res++;
        }
    }
    return res;
}

void solve(){
    ll l, r;
    cin >> l >> r;
    ll res = f(l,r);
    print(res)
}

int main(){
    fast_io
    int t = 1;
    // cin >> t;
    rep(i,t) solve();
}
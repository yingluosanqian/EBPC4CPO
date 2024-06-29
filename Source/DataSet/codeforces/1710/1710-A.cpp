	
//HEADER FILES AND NAMESPACES
 
#include<bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>  
#include <ext/pb_ds/tree_policy.hpp>  
#pragma GCC target("popcnt") 
using namespace std;
using namespace __gnu_pbds;
 
template <typename T>
using ordered_set = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;
 
template <typename T>
using ordered_multiset = tree<T, null_type, less_equal<T>, rb_tree_tag, tree_order_statistics_node_update>; 

using cd = complex<double>;
// DEFINE STATEMENTS
const long long infty = 1e18;
#define num1 1000000007
#define num2 998244353
#define REP(i,a,n) for(ll i=a;i<n;i++)
#define REPd(i,a,n) for(ll i=a; i>=n; i--)
#define pb push_back
#define pob pop_back
#define fr first
#define sc second
#define fix(f,n) std::fixed<<std::setprecision(n)<<f
#define all(x) x.begin(), x.end()
#define M_PI 3.14159265358979323846
#define epsilon (double)(0.000000001)
#define popcount __builtin_popcountll
#define fileio(x) freopen("input.txt", "r", stdin); freopen(x, "w", stdout);
#define out(x) cout << ((x) ? "Yes\n" : "No\n")
#define sz(x) x.size()
 
 
typedef long long ll;
typedef long long unsigned int llu;
typedef vector<long long> vll;
typedef pair<long long, long long> pll;
typedef vector<pair<long long, long long>> vpll;
typedef vector<int> vii;
 
// DEBUG FUNCTIONS 
#ifndef ONLINE_JUDGE
 
template<typename T>
void __p(T a) {
    cout<<a;
}
template<typename T, typename F>
void __p(pair<T, F> a) {
    cout<<"{";
    __p(a.first);
    cout<<",";
    __p(a.second);
    cout<<"}";
}
template<typename T>
void __p(std::vector<T> a) {
    cout<<"{";
    for(auto it=a.begin(); it<a.end(); it++)
        __p(*it),cout<<",}"[it+1==a.end()];
}
template<typename T>
void __p(std::set<T> a) {
    cout<<"{";
    for(auto it=a.begin(); it!=a.end();){
        __p(*it); 
        cout<<",}"[++it==a.end()];
    }
 
}
template<typename T>
void __p(std::multiset<T> a) {
    cout<<"{";
    for(auto it=a.begin(); it!=a.end();){
        __p(*it); 
        cout<<",}"[++it==a.end()];
    }
}
template<typename T, typename F>
void __p(std::map<T,F> a) {
    cout<<"{\n";
    for(auto it=a.begin(); it!=a.end();++it)
    {
        __p(it->first);
        cout << ": ";
        __p(it->second);
        cout<<"\n";
    }
    cout << "}\n";
}
 
template<typename T, typename ...Arg>
void __p(T a1, Arg ...a) {
    __p(a1);
    __p(a...);
}
template<typename Arg1>
void __f(const char *name, Arg1 &&arg1) {
    cout<<name<<" : ";
    __p(arg1);
    cout<<endl;
}
template<typename Arg1, typename ... Args>
void __f(const char *names, Arg1 &&arg1, Args &&... args) {
    int bracket=0,i=0;
    for(;; i++)
        if(names[i]==','&&bracket==0)
            break;
        else if(names[i]=='(')
            bracket++;
        else if(names[i]==')')
            bracket--;
    const char *comma=names+i;
    cout.write(names,comma-names)<<" : ";
    __p(arg1);
    cout<<" | ";
    __f(comma+1,args...);
}
#define trace(...) cout<<"Line:"<<__LINE__<<" ", __f(#__VA_ARGS__, __VA_ARGS__)
#else
#define trace(...)
#define error(...)
#endif
 
 
// DEBUG FUNCTIONS END 
 
//modinverse gand maraye
template <int MOD_> struct modnum {
    static constexpr int MOD = MOD_;
    static_assert(MOD_ > 0, "MOD must be positive");
 
private:
    using ll = long long;
 
    int v;
 
    static int minv(int a, int m) {
        a %= m;
        assert(a);
        return a == 1 ? 1 : int(m - ll(minv(m, a)) * ll(m) / a);
    }
 
public:
 
    modnum() : v(0) {}
    modnum(ll v_) : v(int(v_ % MOD)) { if (v < 0) v += MOD; }
    explicit operator int() const { return v; }
    friend std::ostream& operator << (std::ostream& out, const modnum& n) { return out << int(n); }
    friend std::istream& operator >> (std::istream& in, modnum& n) { ll v_; in >> v_; n = modnum(v_); return in; }
 
    friend bool operator == (const modnum& a, const modnum& b) { return a.v == b.v; }
    friend bool operator != (const modnum& a, const modnum& b) { return a.v != b.v; }
 
    modnum inv() const {
        modnum res;
        res.v = minv(v, MOD);
        return res;
    }
    friend modnum inv(const modnum& m) { return m.inv(); }
    modnum neg() const {
        modnum res;
        res.v = v ? MOD-v : 0;
        return res;
    }
    friend modnum neg(const modnum& m) { return m.neg(); }
 
    modnum operator- () const {
        return neg();
    }
    modnum operator+ () const {
        return modnum(*this);
    }
 
    modnum& operator ++ () {
        v ++;
        if (v == MOD) v = 0;
        return *this;
    }
    modnum& operator -- () {
        if (v == 0) v = MOD;
        v --;
        return *this;
    }
    modnum& operator += (const modnum& o) {
        v += o.v;
        if (v >= MOD) v -= MOD;
        return *this;
    }
    modnum& operator -= (const modnum& o) {
        v -= o.v;
        if (v < 0) v += MOD;
        return *this;
    }
    modnum& operator *= (const modnum& o) {
        v = int(ll(v) * ll(o.v) % MOD);
        return *this;
    }
    modnum& operator /= (const modnum& o) {
        return *this *= o.inv();
    }
 
    friend modnum operator ++ (modnum& a, int) { modnum r = a; ++a; return r; }
    friend modnum operator -- (modnum& a, int) { modnum r = a; --a; return r; }
    friend modnum operator + (const modnum& a, const modnum& b) { return modnum(a) += b; }
    friend modnum operator - (const modnum& a, const modnum& b) { return modnum(a) -= b; }
    friend modnum operator * (const modnum& a, const modnum& b) { return modnum(a) *= b; }
    friend modnum operator / (const modnum& a, const modnum& b) { return modnum(a) /= b; }
};
 
template <typename T> T pow(T a, long long b) {
    assert(b >= 0);
    T r = 1; while (b) { if (b & 1) r *= a; b >>= 1; a *= a; } return r;
}
// khatam
using num = modnum<998244353>;
//ll a[100006];
// ll mod = 1000000007;
ll mod = 998244353;
// //ll mod = 1000000009;
const int N = 2e5+5;
ll fact[N] ,inv_fact[N];
 
long long powm(long long a, long long b) {
    long long res = 1;
    while (b > 0) {
        if (b & 1)
            res = res * a %mod;
        a = a * a %mod;
        b >>= 1;
    }
    return res;
}
 
void pre_calculate(){
    fact[0] = 1;
    for(int i = 1;i<N;i++){
        fact[i] = fact[i-1]*i % mod;
    }
 
    inv_fact[N-1] = powm(fact[N-1],mod-2);
 
    for(int i = N-2;i>=0;i--){
        inv_fact[i] = inv_fact[i+1]*(i+1) % mod;
        //cout << inv_fact[i]<<"\n";
    }
}
 
 
// #define N 1000005
 
// ll lpf[N];
// ll mobius[N];
// void least_prime_factor()
// {
//     for (ll i = 2; i < N; i++)
//         if (!lpf[i])
//             for (ll j = i; j < N; j += i)
//                 if (!lpf[j])
//                     lpf[j] = i;
// }
 
// void Mobius(){
//     for (ll i = 1; i < N; i++) {
//         if (i == 1)
//             mobius[i] = 1;
//         else {
//             if (lpf[i / lpf[i]] == lpf[i])
//                 mobius[i] = 0;
//             else
//                 mobius[i] = -1 * mobius[i / lpf[i]];
//         }
//         //cout << mobius[i] << "\n";
//     }
// }
ll modInverse(ll a, ll m){
    ll m0 = m;
    ll y = 0, x = 1;
    if (m == 1)
        return 0;
    while (a > 1) {
        ll q = a / m;
        ll t = m;
        m = a % m, a = t;
        t = y;
 
        // Update y and x
        y = x - q * y;
        x = t;
    }
    if (x < 0)
        x += m0;
    return x;
}
ll gcd (ll a, ll b) {
    while (b) {
        a %= b;
        swap(a, b);
    }
    return a;
}
// vector<vector<ll>>tre;
// vector<ll>endofstring;
// ll cnt = 0;
// void built(string s){
//     ll node = 0;
//     for(ll i = 0;i<s.length();i++){
//         if(tre[node][s[i]- 'a'] == 0)tre[node][s[i]-'a'] = ++cnt;
//         node = tre[node][s[i]-'a'];
//     }
//     endofstring[node] = 1;
// }
// vector<ll>pi;
// void prefix_function(string s){
//     ll n = s.length();
//     pi.resize(n,0);
//     for(ll i = 1;i<n;i++){
//         ll j = pi[i-1];
//         while(j>0 && s[i]!=s[j])j = pi[j-1];
//         if(s[i] == s[j])j++;
//         pi[i] = j;
//     }
// }
// vector<ll>z;
// void z_function(string s) {
//     ll n =  s.length();
//     z.resize(n,0);
//     for (ll i = 1, l = 0, r = 0; i < n; ++i) {
//         if (i <= r)
//             z[i] = min (r - i + 1, z[i - l]);
//         while (i + z[i] < n && s[z[i]] == s[i + z[i]])
//             ++z[i];
//         if (i + z[i] - 1 > r)
//             l = i, r = i + z[i] - 1;
//     }
// }
// vector<ll>p;
// void sort_cyclic_shifts(string const& s){
//     ll n = s.size();
//     const ll alphabet = 256;
//     vector<ll>c(n), cnt(max(alphabet,n),0);
//     p.resize(n);
//     for(ll i = 0;i<n;i++)cnt[s[i]]++;
//     for(ll i = 1;i<alphabet;i++)cnt[i] += cnt[i-1];
//     for(ll i = 0;i<n;i++)p[--cnt[s[i]]] = i;
//     c[p[0]] = 0;
//     ll classes = 1;
//     for(ll i = 1;i<n;i++){
//         if(s[p[i]] != s[p[i-1]])classes++;
//         c[p[i]] = classes-1;
//     }
//     vector<ll>pn(n), cn(n);
//     //trace(p);
//     for(ll h = 0;(1<<h) < n;++h){
//         for(ll i = 0;i<n;i++){
//             pn[i] = p[i] - (1<<h);
//             if(pn[i] < 0)pn[i] += n;
//         }
//         fill(cnt.begin(),cnt.begin()+classes,0);
//         //trace(p);
//         for(ll i = 0;i<n;i++)cnt[c[pn[i]]]++;
//         for(ll i = 1;i<classes;i++)cnt[i] += cnt[i-1];
//         for(ll i = n-1;i>=0;i--)p[--cnt[c[pn[i]]]] = pn[i];
//         cn[p[0]] = 0;
//         classes =1;
//         for(ll i = 1;i<n;i++){
//             pair<ll,ll>cur = {c[p[i]],c[(p[i] + (1<<h))%n]};
//             pair<ll,ll>prev = {c[p[i-1]], c[(p[i-1] + (1<<h))%n]};
//             if(cur!=prev) ++classes;
//             cn[p[i]] = classes-1;
//         }
//         c.swap(cn);
//     }
// }

// vector<vector<ll>>imp(3,vll(3));
struct Matrix{
    vector<vector<ll>> a; // = imp;
    ll n = 0;
    Matrix operator *(Matrix &other){
        Matrix product;
        product.n = n;
        for(ll i = 0;i<n;i++){
            for(ll j = 0;j<n;j++){
                for(ll k = 0;k<n;k++){
                    product.a[i][k] += a[i][j]*other.a[j][k];
                    product.a[i][k] %= (mod-1);
                }
            }
        }
        return product;
    }
};

void solve(){
    ll n,m,k; cin >> n >> m >> k;
    vector<ll>a(k);
    for(ll i = 0;i<k;i++)cin >> a[i];
    ll x = 0, y = 0;
    sort(a.begin(),a.end());
    ll x1 = 0, x2 = 0;
    vector<ll>b(k),c(k);
    ll ma1 = 0, ma2 = 0;
    set<ll>S1,S2;
    for(ll i = 0;i<k;i++){
        ll temp = a[i]/m;
        ll temp2 = a[i]/n;
        if(temp>=2){
            b[i] = temp;
            x1+=b[i];
            ma1 = max(ma1,b[i]);
            S1.insert(b[i]);
        }
        if(temp2>=2){
            c[i] = temp2;
            x2+=c[i];
            ma2 = max(ma2,c[i]);
            S2.insert(c[i]);
        }
    }
    sort(b.begin(),b.end());
    sort(c.begin(),c.end());
    for(ll i = 0;i<k;i++){
        if(n == b[i]+1){
            if(b[i]>=3)n = 2;
        }
        else n = max(0ll, n - b[i]);
        if(m == c[i]+1){
            if(c[i]>=3)m= 2;
        }
        else m = max(0ll, m - c[i]);
    }
    if(n==0||m==0)cout << "Yes\n";
    else cout << "No\n";
    // // trace(x1,x2,ma1,ma2);
    // ll temp = 0;
    // if(x1>n){
    //     if(S1.find(x1-n)!=S1.end() || S1.lower_bound(x1-n+2)!=S1.end() || ma1>=n)temp=1;
    // }
    // if(x2>m){
    //     if(S2.find(x2-m)!=S2.end() || S2.lower_bound(x2-m+2)!=S2.end() || ma2>=m)temp=1;
    // }
    
    // // trace(temp);
    // if(x1==n || x2==m){
    //     temp = 1;
    // }   
    // else if(x1<n && x2<m){
    //     temp = 0;
    // }
    // if(temp)cout << "Yes\n";
    // else cout << "No\n";
}   
int main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    // cout.tie(NULL);
 
    #ifdef LOCALFLAG 
        freopen("Input.txt", "r", stdin);
        freopen("Output2.txt", "w", stdout);
    #endif
    
    ll t = 1;
    cin >> t;
    while(t--){
        solve();
    }
 
} 
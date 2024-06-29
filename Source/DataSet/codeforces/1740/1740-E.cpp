#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
#define int                                                             long long
#define ll                                                              long long
#define ld                                                              long double
#define ull                                                             unsigned long long
#define itn                                                             int
#define sqrt                                                            sqrtl
#define cbrt                                                            cbrtl
#define pow                                                             powl
#define floor                                                           floorl
#define ceil                                                            ceill
#define fi                                                              first
#define se                                                              second
#define range(a, b, c)                                                  a <= b && b <= c
#define ckmin(a, b)                                                     a = min(a, b)
#define ckmax(a, b)                                                     a = max(a, b)
#define all(x)                                                          (x).begin(), (x).end()
#define allr(x)                                                         (x).rbegin(), (x).rend()
#define mp(a, b)                                                        make_pair(a, b)
#define N                                                               1000005
#define xyz                                                             1000005
#define debug(x)                                                        cout << #x << " => " << x << "\n";
#define tc                                                              int t; cin >> t; while(t--) {solve();}
#define slv                                                             solve();
#define ordered_multiset                                                tree<int, null_type, less_equal<int>, rb_tree_tag, tree_order_statistics_node_update>
#define ordered_set                                                     tree<int, null_type, less<int>, rb_tree_tag, tree_order_statistics_node_update>
#define Aufan                                                           ios_base::sync_with_stdio(false);cin.tie(NULL);cout.precision(numeric_limits<double>::max_digits10);
using namespace std;
using namespace __gnu_pbds;
bitset<1000007> isPrime;
const int MOD=1e9+7;
const int mod=998244353;
const int INF=2e9;
const int INFF=4e18;
ull fac[xyz];
ull modin[xyz];
struct Point                                                            {int x;int y;};
template <typename T>T modx(T a,T b)                                    {return (a-1)%b+1;}
template <typename T>T modp(T a,T b)                                    {return (a%b+b)%b;}
template <typename T> void print(T v)                                   {for(auto x:v)cout<<x<<' ';cout<<'\n';}
template <typename A, typename B> void print(map<A,B> m)                {for(auto x:m)cout<<x.fi<<' '<<x.se<<'\n';}
template <typename A, typename B> void print(vector<pair<A,B>> v)       {for(auto x:v)cout<<x.fi<<' '<<x.se<<'\n';}
template <typename T> map<T,int> freq_count(vector<T> v)                {int n=v.size();map<T,int> m;for(T x:v)m[x]++;return m;}
template <typename T> vector<pair<T,int>> group_count(vector<T> v)      {int n=v.size();vector<pair<T,int>> r;if(!n)return r;int c=1;for(int i=1;i<n;i++){if(v[i]==v[i-1])c++;else{r.push_back(make_pair(v[i-1],c));c=1;}}r.push_back(make_pair(v[n-1],c));return r;}
template <typename T> vector<T> prefix_sum(vector<T> v)                 {int n=v.size();vector<T> r(n+1,0);for(int i=0;i<n;i++) {r[i+1]=r[i]+v[i];}return r;}
template <typename A, typename B> vector<A> keys(map<A,B> m)            {vector<A> v;for(auto x:m) v.push_back(x.fi);return v;}
template <typename A, typename B> vector<B> values(map<A,B> m)          {vector<B> v;for(auto x:m) v.push_back(x.se);return v;}
vector<pair<char,int>> group_count(string s)                            {return group_count(vector(all(s)));}
map<char,int> freq_count(string s)                                      {return freq_count(vector(all(s)));}
int orientation(Point p,Point q,Point r)                                {int val=(q.y-p.y)*(r.x-q.x)-(q.x-p.x)*(r.y-q.y);if(val==0)return 0;return (val>0)?1:2;}
bool onSegment(Point p,Point q,Point r)                                 {if(q.x<=max(p.x,r.x)&&q.x>=min(p.x,r.x)&&q.y<=max(p.y,r.y)&&q.y>=min(p.y,r.y))return true;return false;}
bool doIntersect(Point p1,Point q1,Point p2,Point q2)                   {int o1=orientation(p1,q1,p2);int o2=orientation(p1,q1,q2);int o3=orientation(p2,q2,p1);int o4=orientation(p2,q2,q1);if(o1!=o2&&o3!=o4)return true;if(o1==0&&onSegment(p1,p2,q1))return true;if(o2==0&&onSegment(p1,q2,q1))return true;if(o3==0&&onSegment(p2,p1,q2))return true;if(o4==0&&onSegment(p2,q1,q2))return true;return false;}
bool difdigit(int a)                                                    {int b=floor(log10(a)+1);int arr[10];memset(arr,0,sizeof(arr));while(a>0){arr[a%10]++;a/=10;}if(count(arr,arr+10,1)==b)return true;return false;}
bool isPerfectSquare(int x)                                             {if(x>=0){int sr=sqrt(x);return (sr*sr==x);}return false;}
bool isPal(int x)                                                       {string s="";while(x){s+=x%10;x/=10;}string t=s;reverse(all(s));return s==t;}
bool checkPrime(int x)                                                  {for(int i=2;i*i<=x;i++){if(x%i==0){return false;}}return true;}
char letter(int n)                                                      {string alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";return alphabet[n];}
char number(int n)                                                      {string bruh="0123456789"; return bruh[n];}
string balik(string s)                                                  {reverse(s.begin(),s.end());return s;}
string base10_2(int x)                                                  {if(x == 1)return "1";if(x%2)return base10_2(x/2)+"1";return base10_2(x/2)+"0";}
string to_upper(string s)                                               {string ret="";for(auto c:s){ret.push_back(toupper(c));}return ret;}
string to_lower(string s)                                               {string ret="";for(auto c:s){ret.push_back(tolower(c));}return ret;}
int xorNih(int n)                                                       {if(n%4==0)return n;if(n%4==1)return 1;if(n%4==2)return n+1;return 0;}
int base2_10(int x)                                                     {int output;for(int i=0;x>0;i++){if(x%10==1)output+=pow(2,i);x/=10;}return output;}
int digitsum(int a)                                                     {int b=0;while(a>0){b+=a%10;a/=10;}return b;}
int digitmul(int a)                                                     {int b=1;while(a>0){if(a%10)b*=a%10;a/=10;}return b;}
int factorial(int a)                                                    {int b=1;while(a>0){b*=a;a--;}return b;}
int ceildiv(int a, int b)                                               {int q;q=a/b+(a%b!=0);return q;}
int binpow(int a, int b, int c)                                         {if(b==0)return 1;int d=binpow(a,b/2,c);return ((b%2?a%c:1)*((d*d)%c))%c;}
ull power(ull x,int y)                                                  {ull res=1;x=x%MOD;while(y>0){if(y&1)res=(res*x)%MOD;y=y>>1;x=(x*x)%MOD;}return res;}
ull modInverse(ull n)                                                   {return power(n,MOD-2);}
ull nCrModPFermat(ull n,int r)                                          {if(n<r)return 0;if(r==0)return 1;return (fac[n]*modin[r]%MOD*modin[n-r]%MOD)%MOD;}
void generatefac()                                                      {fac[0]=1;modin[0]=modInverse(1);for(int i=1;i<=xyz;i++){fac[i]=(fac[i-1]*i)%MOD;modin[i]=modInverse(fac[i]);}}
void callprime()                                                        {for(int i=2;i<=1000000;i++){if(!isPrime[i]){int j=i*i;while(j<=1000000){isPrime[j]=true;j+=i;}}}isPrime.flip();isPrime[1]=false;}
vector<vector<int>> v(N);
vector<int> dpt(N);
int dfs(int x) {
        int sum = 0;
        dpt[x] = 1;
        for (auto z : v[x]) {
                sum += dfs(z);
                ckmax(dpt[x], dpt[z] + 1);
        }
        ckmax(sum, dpt[x]);
        return sum;
}
void solve() {
        int n;
        cin >> n;
        for (int i = 2; i <= n; i++) {
                int x;
                cin >> x;
                v[x].push_back(i);
        }
        cout << dfs(1) << '\n';
        return;
}

int32_t main()
{
        Aufan
        slv
        return 0;
}
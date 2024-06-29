// HEADER FILES ARE GIVEN BELOW
#include "bits/stdc++.h"
#include "iostream"
#include "cstdio"
#include "cstring"
#include "cmath"
#include "chrono"
#include "complex"
#include "unordered_map"
#include "unordered_set"
#include "iomanip"
#include "map"
#include "iterator"
#include "deque"
#include "queue"
#include "sstream"
#include "string"
#include "set"
#include "vector"
#include "algorithm"
#include "bitset"

// MACROS ARE GIVEN BELOW
#define PI              3.141592653589793238462
#define mod             998244353
#define inf             1000000000000000001;
#define li              long int
#define lli             long long int
#define vi              vector<int>
#define vs              vector<string>
#define vll             vector<ll>
#define vvi             vector<vi>
#define ii              pair<int, int>
#define mp(x,y)         make_pair(x,y)
#define mem(a,val)      memset(a,val,sizeof(a))
#define eb              emplace_back
#define vpll            vector<pair<ll,ll>>
#define pii             pair<int,int>
#define mii             map<int,int>
#define pll             pair<ll, ll>
#define F               first
#define S               second
#define all(C)          (C).begin(), (C).end()
#define ss              second
#define ff              first
#define pb              push_back
#define ppb             pop_back
#define pf              push_front
#define ppf             pop_front
#define mp              make_pair
#define pii             pair<int,int>
#define mii             map<int,int>
#define si              set<int>
#define debug(x)        cout << '>' << #x << ':' << x << endl
#define rep(i,a,n)      for(int i=a;i<n;i++)
#define ninf            LLONG_MIN
#define inf             LLONG_MAX
#define sz              size()
#define endl            '\n'
#define el              cout<<endl
#define end_line        << endl;
#define endll           '\n'
#define riter(it, ds)   for (it = ds.rbegin(); it != ds.rend(); it--)
#define iter(it, ds)    for (it = ds.begin(); it != ds.end(); it++)
#define go(i,n)         for(ll i=0;i<n;i++)
#define go1(i,n)        for(int i=1;i<n;i++)
#define rgo(i, n)       for(int i = n - 1; i >= 0; i--)
#define rgo1(i, n)      for(int i = n; i >= 1; i--)
#define go(i, n)        for (int i = 0; i < n; i++)
#define go1(i, n)       for (int i = 1; i <= n; i++)
#define fl(i,n)         for(int i=0;i<n;i++)
#define rl(i,m,n)       for(int i=n;i>=m;i--)
#define rep0(i,n)       for(int i=0;i<n;i++)
#define rep1(i,n)       for(int i=1;i<=n;i++)
#define repl(i,a,b)     for(int i=a; i<b; i++)
#define rep11(i,a,b)    for(int i=a; i<=b; i++)
#define f(i,a,b)        for(ll i=a;i<b;i++)
#define repe(i,a,b)     for(ll i=a;i<=b;i++)
#define dep(i,a,b)      for(ll i=a; i>=b; i--)
#define vi              vector<int>
#define vl              vector<ll>
#define vs              vector<string>
#define mll             map<ll,ll>
#define unmap           unordered_map<int,int> m
#define umll            unordered_map<ll,ll>
#define pf              push_front
#define pb              push_back
#define popb            pop_back
#define mp              make_pair
#define all(c)          c.begin(),c.end()
#define vr(v)           v.begin(),v.end():
#define rv(v)           v.end(),v.begin()
#define fast            ios_base::sync_with_stdio(false);cin.tie(0); cout.tie(0);
#define tc              ll tc;  cin>>tc;  while(tc--)
#define YES             cout<<"YES"<<endl
#define NO              cout<<"NO"<<endl
#define fixed           cout << fixed
#define dne             cout<<-1<<endl
#define set_bits __builtin_popcount


//SPEED
#define fio             ios_base::sync_with_stdio(false), cout.tie(),cin.tie()
#define fast            ios_base::sync_with_stdio(0), cin.tie(0)


using namespace std;

//ALIASES
using ll= long long;
using lld= long double;
using ull= unsigned long long;

//CONSTANTS
const int N = 2e5 + 11;
const lld pi= 3.141592653589793238;
const ll INF= 1e18;
const ll modu=1e9+7;


//FUNCTIONS GIVEN BELOW
int pow(int a,int b){if(!b){return 1;}int ans=pow(a,b/2); ans*=ans; if(b%2){return ans*=a;}return ans;}
int gcd(int a,int b){return b?gcd(b,a%b):a;}int lcm(int a,int b){return a/gcd(a,b)*b;}
int add(int a,int b,int m){return (a+b)%m;}
int sub(int a,int b,int m){return ((a-b)%m+m)%m;}
int mul(int a,int b,int m){return ((a%m)*(b%m))%m;}

//EXTRA FUNCTIONS
void sort_dec(int &a ,int &b , int &c) //to sort a,b,c in decreasing order
{
    if (a<b) swap(a,b);
    if (a<c) swap(a,c);
    if (b<c) swap(b,c);
}


bool is_prime(int n)
{// This will check if given n is prime or not
    if(n<=1)
        return false;
    if(n<=3)
        return true;
    if(n%2==0 || n%3==0)
        return false;
    for(int i=5; i*i<=n;i=i+6)
    {
        if (n%i==0 || n%(i+2)==0)
            return false;
    }
    return true;
}



// This will give prime factorization of 'n'
// 0(sqrt(n)/log(n))
// count of prime numbers is from 1 to n => (n/log(n))
vector<int> prime_fact(int n)
{
    vector<int> p,unique_p;
    if(n%2==0)
    {
        unique_p.pb(2);
        while(n%2==0)
        {
            p.pb(2),n/=2;
        }
    }
    for(int i=3;i*i<=n;i+=2){
        if(n%i==0){
            unique_p.pb(i);
            while(n%i==0){
                n/=i,p.pb(i);
            }
        }
    }
    if(n>1) p.pb(n),unique_p.pb(n);
//     return unique_p;
    return p;
}



ll findGCD(ll arr[], ll n)
{
    ll result = arr[0];
    for (ll i = 1; i<=n; i++)
    {
        result = gcd(arr[i], result);

        if(result == 1)
        {
           return 1;
        }
    }
    return result;
}

ll countDivisors(ll n)
{
    ll cnt = 0;
    for (ll i = 1; i <= sqrt(n); i++) {
        if (n % i == 0) {
            // If divisors are equal,
            // count only one
            if (n / i == i)
                cnt++;

            else // Otherwise count both
                cnt = cnt + 2;
        }
    }
    return cnt;
}



ll rootSqrt(ll x)
{
    // Base cases
    if (x == 0 || x == 1)
    return x;

    // Starting from 1, try all numbers until
    // i*i is greater than or equal to x.
    ll i = 1, result = 1;
    while (result <= x)
    {
      i++;
      result = i * i;
    }
    return i - 1;
}



ll binpow(ll a, ll x)
{
    if (x == 0)
    {
        return 1;
    }
    if (x % 2 == 1)
    {
        return (a * binpow(a, x - 1));
    }
    ll p = binpow(a, x / 2);
    return (p * p);
}


ll numberOfactors(unordered_set<int> &vc, ll n)
{
    ll ans = 0;
    for (ll i = 1; i <= sqrt(n); i++)
    {
        if ((n % i) != 0)
        {
            continue;
        }
        if (n / i == i)
        {
            ans++;
        }
        else
        {
            ans += 2;
            vc.insert(n / i);
        }
        vc.insert(i);
    }
    return ans;
}


ll numberOfactors(ll n)
{
    ll ans = 0;
    for (ll i = 1; i <= sqrt(n); i++)
    {
        if ((n % i) != 0)
        {
            continue;
        }
        if (n / i == i)
        {
            ans++;
        }
        else
        {
            ans += 2;
        }
    }
    return ans;
}



ll gcd(ll a, ll b)
{
    if (b == 0)
        return a;
    return gcd(b, a % b);
}




ll lcm(ll a, ll b)
{
    return (a / gcd(a, b)) * b;
}


ll fact(ll n)
{
    ll res = 1;
    go1(i, n)
    {
        res *= i;
        res %= (1000000007);
    }

    return res;
}



ll _stoi(string s)
{
    ll n = 0, id = 1;

    while (!s.empty())
    {
        n += (id * (s.back() - 48));
        id *= 10;
        s.pop_back();
    }
return n;
}



string _itos(ll n)
{
    string s;

    while (n > 0)
    {
        s.push_back(char((n % 10) + 48));
        n /= 10;
    }

    reverse(s.begin(), s.end());

    return s;
}



li accurateFloor(li a, li b)
{
    li val = a / b;
    while (val * b > a)
        val--;
    return val;
}





//white function lele


void white()
{

    //declaree krr le
  string xamper;

int number;
int paste;



//input lele

  cin >> number;
  cin >> xamper;



  //paste ko zero se declaree krr le
  paste=0;


  //loop chala de
  for(int index=0;index<xamper.size()-1;++index)
    //condtion laga de
   if(xamper[index]=='('&&xamper[index+1]==')')
   //paste ko update krr de
     paste++;



//output me ye lele

  cout << number-paste+1;

//ye new line ke liyeee
  el;



  //function end krr de
}








char ask_character(int i)
{
    cout << "? 1 " << i << endll;
    string s;
    cin >> s;
    return s[0];
}



int ask_cnt(int l, int r)
{
    cout << "? 2 " << l << " " << r << endll;
    int x;
    cin >> x;
    return x;
}



char anti(char a)
{
    if (a == '1')
    {
        return '0';
    }
    return '1';
}



int num(char a)
{
    return int(a) - 48;
}


bool type(int a)
{
    if (a < 0)
    {
        return 0;
    }
    return 1;
}

bool tip(char a)
{
    if (int(a) >= 65 and int(a) <= 90)
    {
        return 1;
    }
    return 0;
}

bool isP(int x)
{
    long long int s = 0, y = x;
    if (x < 0)
        return false;
    while (y)
    {
        s = s * 10 + y % 10;
        y = y / 10;
    }
    if (s == x)
        return true;
    else
        return false;
}


int opfor(vector<int> &vc, ll trg)
{
    ll l = 0, r = vc.size() - 1, m, ans = -1;

    while (l <= r)
    {
        m = (l + r) / 2;
        if (vc[m] <= trg)
        {
            ans = m;
            l = m + 1;
        }
        else
        {
            r = m - 1;
        }
    }

    return ans;
}


set<vector<int> > com;
set<vector<int> >::iterator it;

void dfs(vector<int> a)
{
    if (a.size() == 4)
    {
        sort(a.begin(), a.end());
        com.insert(a);
        return;
    }
    for (int i = 1; i <= 8; i++)
    {
        bool found = 0;
        for (int j = 0; j < a.size(); j++)
        {
            if (a[j] == i)
            {
                found = 1;
                break;
            }
        }
        if (!found)
        {
            a.push_back(i);
            dfs(a);
            a.pop_back();
        }
    }
}

ll printDivisors(ll n)
{
    ll ans = 0;
    for (ll i = 1; i <= sqrt(n); i++)
    {
        if (n % i != 0)
        {
            continue;
        }
        if (n / i == i)
        {
            ans++;
            continue;
        }
        ans += 2;
    }
    return ans;
}

bool is_pal(vector<int> vc)
{
    // vector<int>::iterator itr;
    for(int i=0; i<vc.size()/2; i++)
    {
        if(vc[i]!=vc[vc.size()-1-i])
        {
            return 0;
        }
    }

    return 1;
}


//TEMPLATES GIVEN BELOW
template<class Read>void in(Read &x){
     x=0;
     int f=0;
     char ch=getchar();
     while(ch<'0'||ch>'9'){
          f|=(ch=='-');
          ch=getchar();
       }
      while(ch>='0'&&ch<='9'){
          x=(x<<1)+(x<<3)+(ch^48);
          ch=getchar();
       }
       x=f?-x:x;
       return;
 }




//MAIN FUNCTION START FROM HERE




 signed main()
 {

     //fast krr de
     fast;

     //fixed krr de
     fixed;


     //tester lele
     int tester;


     //input lele
     cin>>tester;

//loop chala de
     while(tester--)
     {

        white();

     }
     return 0;
 }

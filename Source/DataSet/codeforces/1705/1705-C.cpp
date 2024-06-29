/*                                                   The internationalVillager                                                */



/******************************************************************************************************************************/
/******************************************************************************************************************************/
/******************************************************************************************************************************/
/******************************************************************************************************************************/
/****                                                                                                                      ****/
/****                                                                                                                      ****/
/****                                                                                                                      ****/
/****                                                                                                                      ****/
/**** //                //    // // // // // //              // // // //      // // // //           //         // // // // ****/
/**** // //          // //    //                             //                       //          ////         //          ****/
/**** //    //    //    //    //                             //                      //         //  //         //          ****/
/**** //       //       //    // // // // // //              // // // //         // // //     // // // //      // // // // ****/
/**** //                //                   //                       //           //               //         //       // ****/
/**** //                //                   //                       //          //                //         //       // ****/
/**** //                //    // // // // // // ............ // // // //         //                 //         // // // // ****/
/****                                                                                                                      ****/
/****                                                                                                                      ****/
/****                                                                                                                      ****/
/****                                                                                                                      ****/
/******************************************************************************************************************************/
/******************************************************************************************************************************/
/******************************************************************************************************************************/
/******************************************************************************************************************************/

/// ORDERED SET OPENED

/// Below are the header files and functions specifically for ordered set
/// ordered_set s;        :::: Declaration of ordered set
/// s.find_by_order(k)    :::: It returns to an iterator to the kth element (counting from zero)
/// s.order_of_key(k)     :::: It returns to the number of items that are strictly smaller than our item k
/// #include <ext/pb_ds/assoc_container.hpp>
/// #include <ext/pb_ds/tree_policy.hpp>
/// using namespace __gnu_pbds;
/// #define ordered_set tree<int, null_type,less_equal<int>, rb_tree_tag,tree_order_statistics_node_update>
/// #define ordered_set tree<int, null_type,less<int>, rb_tree_tag,tree_order_statistics_node_update>
/// #define ordered_set_pair tree<pair<ll, ll>, null_type, less<pair<ll, ll>>, rb_tree_tag, tree_order_statistics_node_update >

/// ORDERED SET CLOSED
#include <bits/stdc++.h>
using namespace std;
#define ll long long int
#define ld long double
#define MOD 1000000007
#define MOD1 998244353
#define INF 1000000007
#define pb push_back
#define pf push_front
#define ppb pop_back
#define ff first
#define ss second
#define lb lower_bound
#define ub upper_bound
#define pii pair<int,int>
#define pll pair<ll,ll>
#define vi vector<int>
#define vll vector<ll>
#define set_bits __builtin_popcountll
#define sz(x) ((int)(x).size())
#define all(x) (x).begin(), (x).end()
#define no cout<<"NO\n";
#define yes cout<<"YES\n";
#define bob cout<<"Bob\n"
#define alice cout<<"Alice\n"

#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

// bool prime[1000006];
/*ll spf[MAXN];
void sieve()
{
    spf[1] = 1;
    for (ll i=2; i<MAXN; i++)
        spf[i] = i;
    for (ll i=4; i<MAXN; i+=2)
        spf[i] = 2;

    for (ll i=3; i*i<MAXN; i++)
    {
        if (spf[i] == i)
        {
            for (ll j=i*i; j<MAXN; j+=i)
                if (spf[j]==j)
                    spf[j] = i;
        }
    }
}

   */                                               /// To use nCr uncomment from here

/*const int N = 1000001;                                      -
ll factorialNumInverse[N + 1];

// array to precompute inverse of 1! to N!                 -
ll naturalNumInverse[N + 1];

// array to store factorial of first N numbers              -
ll fact[N + 1];

// Function to precompute inverse of numbers
void InverseofNumber(ll p)
{                                                                -
    naturalNumInverse[0] = naturalNumInverse[1] = 1;
    for (int i = 2; i <= N; i++)
        naturalNumInverse[i] = naturalNumInverse[p % i] * (p - p / i) % p;
}
// Function to precompute inverse of factorials
void InverseofFactorial(ll p)                                  -                      -
{
    factorialNumInverse[0] = factorialNumInverse[1] = 1;

    // precompute inverse of natural numbers
    for (int i = 2; i <= N; i++)                                        -
        factorialNumInverse[i] = (naturalNumInverse[i] * factorialNumInverse[i - 1]) % p;
}

// Function to calculate factorial of 1 to N                               -
void factorial(ll p)
{
    fact[0] = 1;

    // precompute factorials                                                -
    for (int i = 1; i <= N; i++) {
        fact[i] = (fact[i - 1] * i) % p;
    }
}                                                                           -

// Function to return nCr % p in O(1) time                                       -
ll Binomial(ll N, ll R, ll p)
{
    // n C r = n!*inverse(r!)*inverse((n-r)!)
    ll ans = ((fact[N] * factorialNumInverse[R])                             -
              % p * factorialNumInverse[N - R])
             % p;
    return ans;                                                  -
}*/
                                                                  ///Till here for nCr
bool sortbysec(const pair<int,int> &a,
              const pair<int,int> &b)
{
    return (a.second < b.second);
}
bool my_sort(const pair<int,int> &a,
              const pair<int,int> &b)
{
    if(a.first<b.first)
        return true;
    else if(a.first>b.first)
        return false;
    else return(a.second > b.second);
}
ll mod_add(ll a, ll b, ll m) {a = a % m; b = b % m; return (((a + b) % m) + m) % m;}
ll mod_mul(ll a, ll b, ll m) {a = a % m; b = b % m; return (((a * b) % m) + m) % m;}
ll mod_sub(ll a, ll b, ll m) {a = a % m; b = b % m; return (((a - b) % m) + m) % m;}



/*void divisors(int n)
{
    for (int i=1; i<=sqrt(n); i++)
    {
        if (n%i == 0)
        {
            if (n/i == i)
                cout <<" "<< i;

            else
                cout << " "<< i << " " << n/i;
        }
    }
}*/
int divisors(int n)
{
    int ans=0;
    for (int i=1; i<=sqrt(n); i++)
    {
        if (n%i == 0)
        {
            if (n/i == i)
                ans+=i;

            else
                ans+=(i+n/i);
        }
    }
    return ans;
}

void primeFactors(int n)
{
    while (n % 2 == 0)
    {
        cout << 2 << " ";
        n = n/2;
    }
    for (int i = 3; i <= sqrt(n); i = i + 2)
    {
        while (n % i == 0)
        {
            cout << i << " ";
            n = n/i;
        }
    }
    if (n > 2)
        cout << n << " ";
}
ll factorial(ll n)
{
     ll M = 998244353;

     ll f = 1;
    for (ll i = 1; i <= n; i++)
        f = (f*i) % M;  // Now f never can
                        // exceed 10^9+7
    return f;
}
bool isPrime(ll n)
{
    // Corner cases
    if (n <= 1)
        return false;
    if (n <= 3)
        return true;

    // This is checked so that we can skip
    // middle five numbers in below loop
    if (n % 2 == 0 || n % 3 == 0)
        return false;

    for (ll i = 5; i * i <= n; i = i + 6)
        if (n % i == 0 || n % (i + 2) == 0)
            return false;

    return true;
}
/*void SieveOfEratosthenes(int n)
{
    memset(prime, true, sizeof(prime));

    for (int p = 2; p * p <= n; p++)
    {
        if (prime[p] == true)
        {
            for (int i = p * p; i <= n; i += p)
                prime[i] = false;
        }
    }
}*/
void printS (multiset<int>myset)
{
    for (auto it = myset.begin(); it !=
                             myset.end(); ++it)
        cout << *it<< ' ' ;
    cout<<endl;
}
void printS (set<pair<int,int>>myset)
{
    for (auto it = myset.begin(); it !=
                             myset.end(); ++it)
        cout << it->first<< " : " <<it->second<<" ; " ;
    cout<<endl;
}
void printS (set<int>myset)
{
    for (auto it = myset.begin(); it !=
                             myset.end(); ++it)
        cout << *it<< ' ' ;
    cout<<endl;
}
void printS (set<char>myset)
{
    for (auto it = myset.begin(); it !=
                             myset.end(); ++it)
        cout << *it<< ' ' ;
    cout<<endl;
}
void printS (set<ll>myset)
{
    for (auto it = myset.begin(); it !=
                             myset.end(); ++it)
        cout << *it<< ' ' ;
    cout<<endl;
}
void printSS (set<string>myset)
{
    for (auto it = myset.begin(); it !=
                             myset.end(); ++it)
        cout << *it<< ' ' ;
    cout<<endl;
}
void printV (vector<int>v)
{
    for(int i=0;i<v.size();i++)
        cout<<v[i]<<" ";
    cout<<endl;
}
void printV (vector<char>v)
{
    for(int i=0;i<v.size();i++)
        cout<<v[i]<<" ";
    cout<<endl;
}
void printV (vector<ll>v)
{
    for(int i=0;i<v.size();i++)
        cout<<v[i]<<" ";
    cout<<endl;
}
void printV (vector<long int>v)
{
    for(int i=0;i<v.size();i++)
        cout<<v[i]<<" ";
    cout<<endl;
}
void printMii (map<ll,ll>mp)
{
    for(auto it:mp)
        cout<<it.first<<" "<<it.second<<endl;
}
void printMii (map<int,int>mp)
{
    for(auto it:mp)
        cout<<it.first<<" "<<it.second<<endl;
}
void printVP (vector<pair<ll,ll>>v)
{
    for(ll i=0;i<v.size();i++)
        cout<<v[i].first<<" "<<v[i].second<<endl;
}
void printVP (vector<pair<int,int>>v)
{
    for(int i=0;i<v.size();i++)
        cout<<v[i].first<<" "<<v[i].second<<endl;
}
void printA (int v[],int n)
{
    for(int i=0;i<n;i++)
        cout<<v[i]<<" ";
    cout<<endl;
}
void printA (ll v[],ll n)
{
    for(ll i=0;i<n;i++)
        cout<<v[i]<<" ";
    cout<<endl;
}
mt19937 rng;
ll getRandomNumber(ll l, ll r)
{
    uniform_int_distribution<ll> dist(l, r);
    return dist(rng);
}
ll power(ll x, ll y, ll p)
{
    ll res = 1;     // Initialize result

    x = x % p; // Update x if it is more than or
                // equal to p

    if (x == 0) return 0; // In case x is divisible by p;

    while (y > 0)
    {
        // If y is odd, multiply x with result
        if (y & 1)
            res = (res*x) % p;

        // y must be even now
        y = y>>1; // y = y/2
        x = (x*x) % p;
    }
    return res;
}
ll power(ll x, ll y)
{
    ll res = 1;     // Initialize result

     // Update x if it is more than or
                // equal to p

    if (x == 0) return 0; // In case x is divisible by p;

    while (y > 0)
    {
        // If y is odd, multiply x with result
        if (y & 1)
            res = (res*x) ;

        // y must be even now
        y = y>>1; // y = y/2
        x = (x*x) ;
    }
    return res;
}
bool isPal(string s)
{
    int i,n=s.size();
    for(i=0;i<n/2;i++)
        if(s[i]!=s[n-1-i])
            return false;
    return true;
}
/*void sieve(vector<ll>&v)       /// returns number of divisors of all numbers from 1 to N in O(NlogN)
{
    for (int i = 1; i < v.size(); ++i) {
        for (int j = i; j < v.size(); j += i)
            v[j]++;
    }
}*/
void sieve(vector<ll>&v)       /// returns sum of divisors of all numbers from 1 to N in O(NlogN)
{
    for (int i = 1; i < v.size(); ++i) {
        for (int j = i; j < v.size(); j += i)
            v[j]+=i;
    }
}
/*int dfs(int root,vector<bool>&vis,vector<int>adj[],vector<int>&val,int ev)
{
    /// find cnt of connected components satisfying specific property
    /// refer to https://codeforces.com/problemset/problem/1592/C
    vis[root]=true;
    int temp=val[root];
    for(auto it:adj[root])
    {
        if(!vis[it])
        {
            temp^=dfs(it,vis,adj,val,ev);
        }
    }
    if(temp==ev)
    {
        //cnt++;
        temp=0;
    }
    return temp;
}*/
bool isValid()
{
    return true;
}
bool comp(pair<ll,ll> &a,pair<ll,ll> &b){
    if(a.second==b.second){
        return a.first<b.first;
    }
    return a.second<b.second;
}
bool isBipartite(int V, vector<int> adj[])
{
    vector<int> col(V, -1);
    queue<pair<int, int> > q;
    for (int i = 0; i < V; i++) {
        if (col[i] == -1) {
            q.push({ i, 0 });
            col[i] = 0;
            while (!q.empty()) {
                pair<int, int> p = q.front();
                q.pop();
                int v = p.first;
                int c = p.second;
                for (int j : adj[v]) {
                    if (col[j] == c)
                        return 0;
                    if (col[j] == -1) {
                        col[j] = (c) ? 0 : 1;
                        q.push({ j, col[j] });
                    }
                }
            }
        }
    }
    return 1;
}
ll dfsc(ll curr,vector<bool>&vis,vector<pll>&r,vector<ll>adj[],ll *ans)
{
    vis[curr]=true;
    ll x=0;
    for(auto it:adj[curr])
    {
        if(!vis[it])
            x+=dfsc(it,vis,r,adj,ans);
    }
    if(x<r[curr].ff)
    {
        *ans=*ans+1;
        return r[curr].ss;
    }
    return min(x,r[curr].ss);
}
void solve(int t)
{
    // For floating precision use :-
    // cout<<setprecision(10);
    // cout<<fixed;
    // Closed
    // DSU(n,p);
    /// p in vector<vector<int>> ///
    /// And to check if two nodes belong to same component
    /// findPar(u) == findPar(v) yes else no
    ll n,c,q,i,j;
    cin>>n>>c>>q;
    string s;cin>>s;
    vector<pll>len(c),query;
    ll curr=n;
    for(i=0;i<c;i++)
    {
        ll l,r;
        cin>>l>>r;
        query.pb({l,r});
        len[i].ff=curr+1;
        len[i].ss=(curr+(r-l+1));
        curr=len[i].ss;
    }
    for(i=0;i<q;i++)
    {
        ll ind;
        cin>>ind;
        for(j=c-1;j>-1;j--)
        {
            if(len[j].ff<=ind && len[j].ss>=ind)
            {
                ll gap=ind-len[j].ff;
                ind=gap+query[j].ff;
            }
        }
        cout<<s[ind-1]<<endl;
    }
    return ;
}
int main()
{
    ios::sync_with_stdio(false);
    //cin.tie(NULL);
    //sieve();
    /// For nCr uncomment from here
    //ll p = 1000000007;
    //InverseofNumber(p);
    //InverseofFactorial(p);
    //factorial(p);
    /// Till here
    /// And use it as Binomial(n,r,p)
    //printS(s);
    //SieveOfEratosthenes(1000005);
    int t;
    t=1;
    cin>>t;
    for(int i=1;i<=t;i++)
    {
        //cout<<"Case #"<<i<<": ";
        solve(i);
        //cout<<endl;
        //cout<<endl;
    }

    //solve();

    return 0;
}
/// The basic syntax of BINARY SEARCH OPENED

/// assign l to be minimum value which satisfies the condition
/// assign ans = l
/// assign h to be maximum value which can never be answer
/// define a function isValid with required arguments which checks can any value x satisfy the condition
/// now the code begins
/// while(l<=h)
/// {
///     ll m=(l+h)/2;
///     if(isValid(m,other args))
///     {
///         ans=m;
///         l=m+1;
///     }
///     else h=m-1;
/// }
/// cout<<ans<<endl;

/// BINARY SEARCH CLOSED


#include<bits/stdc++.h>
 /*
 |o              \o__ __o         o__ __o      o__ _o__/_   o__ __o/         \o__ __o/    o     o   
  |>             /       \       /       \    <|           <|      \         /       \   <|>   <|>  
 / \            //       <\     />       \\    |\          / \     <\       //       <\  / >   < \  
 \o/          o/           \o   \o____         |           \o/     o/      <o>           \o__ __o/  
  |          <|             |>      \_\__o__   o__/_        |__  _<|        |__  _ __o__  |__ __|   
  |\          \\           //             \\   |            |       \       |     /   \   /     |   
  o/            \         /     \         /   <o>          <o>       \\o    \         /        <o>  
  |               o      o       o       o     |            |          \     o       o          |   
 / \ _\o__/_     /\__ __/|       <\__ __/>    / \_\o__/_   / \         <\    <\__ __/>         / \  
                                                                                                    
  */                                                                                                  
                                                                                                    

// language c++
#include <map>
//define
#define fo(i,n) for(int i=0; i<n; i++)
#define ar array
#define Endl endl
#define ll long long
#define ld long double
#define ull unsigned long long
#define lb lower_bound
#define ub upper_bound
#define PI 3.1415927
#define pb push_back
#define pii pair<ll,ll>
#define po pop_back
#define yeah cout<<"YES"<<endl;
#define nope cout<<"NO"<<endl;
#define all(x) x.begin(),x.end()
#define sz(x) ((int)x.size())
#define sqr(x) ((x)*(x))
#define getunique(v) {sort(all(v)); v.erase(unique(all(v)), v.end());}
#define unique(v) { v.erase(unique(all(v)), v.end());}
#define X first
#define Y second
#define mp make_pair
#define debug(a) for(ll i=0;i<n;i++) cout<<a[i]<<" ";
#define print(a) for(auto it:a) cout<<it<<" ";
using namespace std;
#define ci(a,n) for(ll i=0;i<n;i++){cin>>a[i];}
#define VEC(n) ll int x; vector<ll int>v;for(int i=0;i<n;i++){cin>>x;v.pb(x);}
#define sm 998244353
#define pq priority_queue 

//functions///////////////
const int MAX_NM = 1e5;
int nums[MAX_NM];
 bool isPrime(int num){
    bool flag=true;
    for(int i = 2; i * i <= num; i++) {
       if(num % i == 0) {
          flag = false;
          break;
       }
    }
    return flag;}
    bool isVowel(char ch) {

    if(ch=='a' || ch=='e' || ch=='i' || ch=='o' || ch=='u')

         return true;

    else return false;

}

const int M= 1e9 +7 ;
        long long mod(long long x){
            return ((x%M + M)%M);
        }
        long long add(long long a, long long b){
            return mod(mod(a)+mod(b));
        }
        long long mul(long long a, long long b){
            return mod(mod(a)*mod(b));
        }
    
        ll modPow(ll a, ll b){
            if(b==0)
                return 1LL;
            if(b==1)
                return a%M;
            ll res=1;
            while(b){
                if(b%2==1)
                    res=mul(res,a);
                a=mul(a,a);
                b=b/2;
            }
            return res;
        }
///////////
bool sortbysec(const pair<int,int> &a,
              const pair<int,int> &b)
{
    return (a.second < b.second);
}
////////////////
ll getProduct(ll n){

    if(n == 0){
        return 1 ;
    }

    return (n%10) * getProduct(n/10) ;
}
/////////*////////
ll Digits(ll n)
{
  ll largest = -1;
    while (n) {
        ll r = n % 10;
 
        largest = max(r, largest);
 
       
 
        n /=10;
    }
    return largest;
}
/**/    /**/        /**/         /**/         /**/
ll digitSum(ll n){
    ll total=0;
    while(n){
        total+=n%10;
        n=n/10;
    }
    return total;
}
ll ceil(ll n)
{
    if (n % 2 == 0)
    {
        return n / 2;
    }
    return n / 2 + 1;
}
/**/ /**/ /**/ /**/ /**/ /**/ /**/
void SieveOfEratosthenes(int n)
{
    bool prime[n+1];
    memset(prime, true, sizeof(prime));
  
    for (int p=2; p*p<=n; p++)
    {
      
        if (prime[p] == true)
        { 
            
            for (int i=p*2; i<=n; i += p)
                prime[i] = false;
        }
    }
    for (int p=2; p<=n; p++)
       if (prime[p])
          cout << p << " ";
}
/**/ /**/ /**/ /**/ 
ll dif(long long int number)
{
    char seen[10] = {0};

    while (number) {
        int digit = number % 10;

        number /= 10;
        if (digit < 0) {
         
            digit = -digit;
            number = -number;
        }
        if (seen[digit]++)
            return 0; 
     }
     return 1;
 }
 /**/ /**/ /**/ /**/ /**/
bool isPalindrome(ll n)
{
   string s= to_string(n);

    string t= s;
    reverse(all(t));
    return s==t;

}
/**/ /**/ /**/ /**/
string intobinary(int n) {
    return bitset<8>(n).to_string();
}
/**/ /**/ /**/ /**/
bool isPowerOfTwo (ll x)
{
    
    return x && (!(x&(x-1)));
}
/**/ /*G*/ /*M*/ /**/
ll factorial(ll n)
{
    
  return (n==1 || n==0) ? 1:( n *((ll) factorial(n - 1)));
} 
/**/ /**/ /**/ /**/
ll gcd(ll a, ll b)
{
  if(b==0)
    return a;
  return gcd(b,a%b);
}
/**/ /**/ /**/ /**/
ll highestPowerOf2(ll n)
{
    return ((n & (~(n - 1))));
}
/**/ /**/ /**/ /**/
ll lcm(ll a, ll b)
{
  return (a/gcd(a,b))*b;
}
/**/ /**/ /**/ /**/
char to_upper (char x){
    if( 97 <= int(x) && int(x) <= 122) return char(x-32);
    if( 65 <= int(x) && int(x) <= 90) return x;
    return -1;
}
/**/ /**/ /**/ /**/
char to_lower (char x){
    if( 97 <= int(x) && int(x) <= 122) return x;
    if( 65 <= int(x) && int(x) <= 90) return char(x+32);
    return -1;
}
/**/ /**/ /**/ /**/
int numerize (char x){
    if(48 <= int(x) && int(x) <= 57) return int(x-'0');
    if(97 <= int(x) && int(x) <= 122) return int(x-96);
    if(65 <= int(x) && int(x) <= 90) return int(x-64);
    return -1;
}
/**/ /**/ /**/ /**/
bool arraySortedOrNot(ll arr[], ll n) {
 
    if (n == 0 || n == 1)
        return true;
 
    for (ll i = 1; i < n; i++) {
        if (arr[i - 1] > arr[i]) {
            return false;
        }
    }
    return true;
}
/**/ /**/ /**/ /**/
bool check(const ll array[], ll n)
{   
    const ll a0 = array[0];

    for (int i = 1; i < n; i++)      
    {         
        if (array[i] != a0)
            return true;
    }
    return false;
}
/**/ /**/ /**/ /**/
ll reverse(ll n) {

    ll r, rev = 0;

    while (n > 0) {
        r = n % 10;
        rev = rev * 10 + r;
        n = n / 10;
    }
    return rev;
}
bool isPal(string S)
{
  
    for (int i = 0; i < S.length() / 2; i++) {

        if (S[i] != S[S.length() - i - 1]) {
           
            return false;
        }
    }
  
    return true;
}
ll fact(ll n);
 
ll nCr(ll n, ll r)
{
    return fact(n) / (fact(r) * fact(n - r));
}
 
// Returns factorial of n
ll fact(ll n)
{
      if(n==0)
      return 1;
    ll res = 1;
    for (ll i = 2; i <= n; i++)
        res = res * i;
    return res;
}
/*GLOBAL*/    /**/     /**/    /*VARIABLES*/
vector<int> adj[110000];
ll a[300005],b[300005];
map<ll,set<ll>>bi;
vector<bool> vis(100005,0);
ll n,m,k,i,j,t,x,y,z,q,res;
string s,str;
ll cnt=0,ans=0;
ll inf = pow(10,18);
vector < pair < int , int > > g[100100];
ll d[100100] , p[100100];
ll pre[300005];
ll post[300005];
set<ll>st;
ll sol=0;
ll sum=0,sum1=0,sum2=0,diff=0;
ll increment=0;
/**/ /**/ /**/ /**/
void run(){
    #ifndef ONLIINE_JUDGE
    freopen("input.txt","r",stdin);
    freopen("output.txt","w",stdout);
    #endif
}
/**/ /**/ /**/ /**/

/**/ /*SOME*/  /*IMPORTANT*/ /*FUNCTIONS*/ /**/
ll lastdigit(ll n){
    ll rem=n%10;
    return rem;
}
bool isSquare(int x){
  int y=sqrt(x);
  return y*y==x;
}
vector<ll>mark;
/**/ /**/ /**/ /**/
void dfs(ll node,ll parent){
    if(mark[node]) return; //if visited
   increment++;
    mark[node]=1;
    for(auto &child:bi[node]){
        if(mark[child]||child==parent) continue; //parent and child not same
      
         dfs(child,node);
    }
 
}
ll countDigit(ll n) {

    string num = to_string(n);
 return num.size();
}
/**/ /**/ /**/ /**/
ll arraySum(ll a[], ll n) 
{
   ll initial_sum  = 0; 
    return accumulate(a, a+n, initial_sum);
}
ll cnttwo(ll n){
    ll i=n;
    ll counter=0;
    while (i>0)
    {
        i=i/2;
        counter++;
    }
    return counter;
}

ll nextPerfectSquare(ll n)
{   
    if (ceil((double)sqrt(n)) == floor((double)sqrt(n))){
        return n*n;
    }
    ll nextN = floor(sqrt(n)) + 1;
 
    return nextN * nextN;
}
/**/ /**/ /**/ /**/
void dijkstra(vector<vector<pair<ll, ll>>> & adj, set<pair<long long, ll>> &q, vector<long long> &dist, vector<ll> &p) {
    while (!q.empty()) {
      auto pp = q.begin();
      int node = pp -> second;
      long long value = (pp -> first);
      q.erase(q.begin());
      for (auto i : adj[node]) {
        ll w = i.second;
        int to = i.first;
        if (value + (ll)w < dist[to]) {
          q.erase({dist[to], to});
          dist[to] = value + (ll)w;
          q.insert({dist[to], to});
          p[to] = node;
        }
      }
    }
  }
  // int cost(string& a, string& b) {
//     int val = 0;
//     for(int i = 0; i < a.size(); ++i) {
//         val += abs(a[i] - b[i]);
//     }
//     return val;
// }
ll f[100005][4];

/**/ /*MAIN*/ /**/ /*THING*/
bool findPartiion(int arr[], int n)
{
    int sum = 0;
    int i, j;
 
    // Calculate sum of all elements
    for (i = 0; i < n; i++)
        sum += arr[i];
 
    if (sum % 2 != 0)
        return false;
 
    bool part[sum / 2 + 1];
 
    // Initialize the part array
    // as 0
    for (i = 0; i <= sum / 2; i++) {
        part[i] = 0;
    }
 
    // Fill the partition table in bottom up manner
 
    for (i = 0; i < n; i++) {
        // the element to be included
        // in the sum cannot be
        // greater than the sum
        for (j = sum / 2; j >= arr[i];
             j--) { // check if sum - arr[i]
            // could be formed
            // from a subset
            // using elements
            // before index i
            if (part[j - arr[i]] == 1 || j == arr[i])
                part[j] = 1;
        }
    }
 
    return part[sum / 2];
}  
int segregate0and1(int arr[], int size)
{
    int cnt=0;
    int type0 = 0;
    int type1 = size - 1;
 
    while (type0 < type1) {
        if (arr[type0] == 1) {
            if (arr[type1] != 1) {
                swap(arr[type0], arr[type1]);
                cnt++;
            }
            type1--;
        }
        else
            type0++;
    }
    return cnt;

}
ll CountSteps(string s1, string s2, ll size)
{
    ll i = 0, j = 0;
    ll result = 0;
 
    
    while (i < size) {
        j = i;
 
       
        while (s1[j] != s2[i]) {
            j += 1;
        }
 
        while (i < j) {
 
            char temp = s1[j];
            s1[j] = s1[j - 1];
            s1[j - 1] = temp;
            j -= 1;
            result += 1;
        }
        i += 1;
    }
    return result;
}
void answer(){
 ll a, b, c, d;
    cin >> a >> b >> c >> d;
    ll x = a * d, y = b * c;
    if (x == y)
        cout << "0"<<endl;
    else if ((y != 0 && x % y == 0) ||( x != 0 && y % x == 0))
        cout << "1"<<endl;
    else
        cout << "2" <<Endl;

}
/*
 2 4 16 32
 64 16 32 8
64=2^6==32*2=16*4+8*8
*/
int main()
{

ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  //bitset<8>(n).to_string();
 ll T = 1;
    cin >> T;
   
     for (int testcase = 1; testcase <= T; testcase++) {
        //cout << "Case #" << testcase << ": ";
        answer();
        
      }
    //dfs(i);
  //ans();
     
}
/*
  ->Things to be careful for-
     & floating point exception and with loop limits
     & corner cases
     & try to think of brute force approach
     & try different approaches if one doesn't work
     & be careful about RE
                    
*/

 








#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
 
using namespace std;
using namespace __gnu_pbds;
 
typedef pair<int, int> node;
typedef tree<int, null_type, less<int>, 
            rb_tree_tag, tree_order_statistics_node_update> ordered_set;
          
 
 
 
 
 

using ll = long long int;
ll mod = 998244353;

            /************************************ Useful Function *********************************************/
ll goodCeil(ll n, ll k){


    if(n%k == 0){
    return n/k;
    }

  return n/k + 1;

 
  
}

ll getMod(ll n){
  return (n + abs(n)*mod)%mod;
}
bool isPrime(ll n){
  if(n == 1){
    return false;
  }
  for(ll i = 2; i*i<=n; i++){
    if(n%i == 0){
      return false;
    }
  }

  return true;
}



ll sumOfArr(vector<ll> A){
  ll n = A.size();
  ll s = 0;
  for(int i=0; i<n; i++){
     s += A[i];
  }

  return s;
}
ll countbits(ll n){
  return __builtin_popcountll(n);
}
ll gcd(ll a, ll b){

 
  ll u = max(a, b);
  ll v = min(a, b);
  ll s;
  while(u&&v){
    s = u;
    u = v;
    v = s%v; 
 
  }
 
  return u;
 
}
 
ll lcm(ll a, ll b){
  return (a*b)/gcd(a,b);
}

ll numPrimeDivisors(ll n){

  if(n == 1){
    return 0;
  }
  ll l = 2;
  ll count = 0;
  while(n > 1){
    while(n%l != 0){
      l++;
    }
    
    n /= l;
    l =2;
    count++;
  }

  return count;

}
 
ll powRmod (ll l, ll n, ll mod){
  ll res = 1;
  while(n){
    if( (n&1) >= 1 ){
      res = (res*l)%mod;
    }
 
    l = (l*l)%mod;
    n >>=1;
  }
 
  return res;
}
 
ll highest_two_divide(ll n){
  ll pow = 1;
  while(n%pow == 0){
    pow = pow*2;
  }
 
  return pow/2;
}
 
ll powR(ll l, ll n){ //using binary exponentiation
  ll res = 1;
  while(n){
    if( (n&1) >= 1 ){
      res = res*l;
    }
 
    l = l*l;
    n >>=1;
  }
 
  return res;
    
}
vector<ll> primeList(ll n){
  vector<bool> S(n+1,false );
  vector<ll> L;
  for(ll i=2; i*i<=n; i++){  
    if(!S[i]){
     
      ll j = 2;
      while(i*j <= n){
        S[i*j] = true;
        j++;
      }
    }
  }
 
 
  for(ll i=2; i<=n; i++){
    if(!S[i]){
        L.push_back(i);
    }
  
  }
 
  return L;
}
 
ll fact_mod(ll n, ll mod){
  ll t = 1;
 
  for(int i=1; i<=n; i++){
    t = (t*i)%mod;
  }
 
  return t;
}
 
 

set<ll> getDivisors(ll n){
  set<ll> ans;
  for(int i=1; i*i<=n; i++){
    if(n%i == 0){
      ans.insert(i);
      ans.insert(n/i);
    }
  }

  return ans;
}

 
string StrRep(string a, ll n){
 string t = "";
 
 while(n){
  t = t + a;
 
  n--;
 }
  return t;
}
 
 
void printArr(vector<ll> &A){
    for(int i=0; i<A.size(); i++){
    cout << A[i] << " ";
  }
}
 
void printPair(pair<ll, ll> A){
  cout << A.first << " " << A.second << endl;
}

void printArrPair(vector<pair<ll, ll>> A){
  for(int i=0; i<A.size(); i++){
    cout << A[i].first << " " << A[i].second << endl;
  }
}
void printmat(vector<vector<ll>> A){
  for(int i=0; i<A.size(); i++){
    for(int j=0; j<A[i].size(); j++){
      cout <<  A[i][j] << " ";
    }
    cout << endl;
  }
}
 
bool isPar(string t){
  ll l =0;
  ll r = t.length()-1;
 
  while(l < r){
    if(t[l] != t[r]){
      return false;
    }
 
    l++;
    r--;
  }
 
 
  return true;
}
 
ll goodSqrt(ll n){
  ll s = sqrt(n);

  if(s*s == n){
    return s;
  }else if(s*s > n){
    return s -1;
  }

  return s;
}

vector<ll> KMP(string text, string pat){
  ll n = text.length();
  ll m  = pat.length();


  vector<ll> ans;
  vector<ll> LPS(m, 0);


  for(int i=1; i<m; i++){
    ll j = i-1;
    while(j >=0 && pat[LPS[j]] != pat[i]){
      j--;
    }

    if(j >=0 ){
       LPS[i] = LPS[j] + 1;
    }
  }

  ll i = 0;
  ll j = 0;

  while(i < n){
    if(text[i] == pat[j]){
        if(j == m - 1){

          ans.push_back(i - m + 1);

          j = 0;
          i++;
        }else{
          i++;
          j++;
        }
    }else{
        if(j == 0){
          i++;
        }else{
          j = LPS[j-1];
          
        }
    }
  }

  return ans;
}
 
vector<ll> factors(ll x){
  ll k = 1;
  vector<ll> T;
  for(;k*k < x; k++){
    if(x%k == 0){
      T.push_back(k);
      T.push_back(x/k);
    }
  }

  if(k*k == x){
    T.push_back(k);
  }

  return T;
}
/******************************************************* Definitions *******************************************************/
 
#define fo(i, a, n) for(ll i=a; i<=n; i++)
#define V vector<ll>
#define sorta(A) sort(A.begin(), A.end())
#define sortd(A) sort(A.rbegin(), A.rend())
#define VP vector<pair<ll, ll>>
#define VV vector<vector<ll>>
#define vector_stuff ll n; cin >> n; vector<ll> A(n); for(int i=0; i<n; i++)cin >> A[i];
#define Min_Priority priority_queue<ll, vector<ll>, greater<ll>>
#define msb(x) 31 - __builtin_clz(x)
#define matrix_stuff ll n, m; cin >> n >> m; vector<vector<ll>> mat(n, vector<ll> (m)); for(int i=0; i<n; i++){for(int j=0; j<m; j++){cin >> mat[i][j];}}
#define graph_stuff_un ll n; cin >> n; adj.assign(n+1, vector<ll> ()); ll u, v; fo(i, 1, n-1){cin >> u >> v;adj[u].push_back(v);adj[v].push_back(u);}
#define graph_stuff_dir ll n; cin >> n; adj.assign(n+1, vector<ll> ()); ll u, v; fo(i, 1, n-1){cin >> u >> v;adj[u].push_back(v);}
#define vector_stuff_k ll n, k; cin >> n >> k; vector<ll> A(n); for(int i=0; i<n; i++)cin >> A[i];
#define YES cout << "YES" << endl; return;
#define NO cout << "NO" << endl;return;
#define Tstart1 int t;cin>>t;while(t){solve();t--;}
#define Tstart2 solve();
#define take(A) vector<ll> A(n); for(int i=0; i<n; i++)cin >>A[i];
#define takem(A) vector<vector<ll>> A(n, vector<ll> (m)); for(int i=0; i<n; i++){for(int j=0; j<m; j++){cin>>A[i][j];}}
 
 
/*************************************************** Cool Data Structures ***************************************************/
 
 
                  /********************************* Binary Indexed Tree *******************************/
 
 
class Btree{
 
  private:
  vector<ll> B;
  ll n;
 
  public:
 
  void update(ll k, ll val){   
                              
    while(k <= n){
      B[k] += val;
      k += (k&-k);
    }
  }
 
  Btree(vector<ll> A){
    n = A.size();
    for(int i=1; i<=n; i++){
    B.push_back(0);
    }
 
    for(int i=0; i<n; i++){
      update(i+1, A[i]);
    }
  }
 
  ll sum(ll k){
    ll s = 0;
 
    while(k >= 1){
      s += B[k];
      k -= (k&-k);
    }
 
    return s;
  }
 
 
 
 
 
 
};
 
/************************************ Dijkstra ********************************************************/
 
/*************************************End of Dijkstra *************************************************/
 
 
                      /********************************* Segment Tree *******************************/
 
 
class segT_sum{
 
private:
vector<ll> T;
ll n;
 
public:  
 
segT_sum(vector<ll> A){
 
  n = A.size();
  for(int i=0; i<2*n; i++){
    T.push_back(0);
  }
 
  for(int i=0; i<n; i++){
    T[i + n] = A[i];
  }
 
  for(int i=n-1; i>0; i--){
    T[i] = T[i<<1] + T[i<<1|1];   
  }
 
}
 
void update(ll p, ll val){
  p = p + n;
  T[p] = val;


  while(p > 1){
    T[p>>1] = T[p] + T[p^1];
    p /= 2;
  }
 
}
 
vector<ll> Tr(){
  return T;
}
 
ll sum(ll l, ll r){    
 
 
    l += n;
    r += n;
    ll s = 0;
    for(; l < r; l = l>>1, r = r>>1){
      if(l&1){
       
         s += T[l];
         l++;
         
      }
      if(r&1){
        r--;
        s += T[r];
        
      
      }
    }
 
    return s;
}
 
 
};
 
 
class segT_min{
 
private:
vector<ll> T;
ll n;
ll min_start;
 
public:  
 
segT_min(vector<ll> A){
 
  n = A.size();
  for(int i=0; i<2*n; i++){
    T.push_back(0);
  }
 
  for(int i=0; i<n; i++){
    T[i + n] = A[i];
    min_start = max(min_start, A[i]);
  }
 
  for(int i=n-1; i>0; i--){
    T[i] = min(T[i<<1] , T[i<<1|1]);   
  }
 
}
 
void update(ll p, ll val){
 
   p = p + n;
  T[p] = val;


  while(p > 1){
    T[p>>1] = min(T[p], T[p^1]);
    p /= 2;
  }
 
}
 
vector<ll> Tr(){
  return T;
}
 
ll query(ll l, ll r){    
 
 
    l += n;
    r += n;
    ll s = 1e18;
    for(; l < r; l = l>>1, r = r>>1){
      if(l&1){
       
         s = min(s, T[l]);
         l++;
         
      }
      if(r&1){
        r--;
        s = min(s, T[r]);
        
      
      }
    }
 
    return s;
}
 
 
};
 
 
class segT_max{
 
private:
vector<ll> T;
ll n;
ll maxi_start;
 
public:  
 
segT_max(vector<ll> A){
 
  n = A.size();
  for(int i=0; i<2*n; i++){
    T.push_back(0);
  }
 
  for(int i=0; i<n; i++){
    T[i + n] = A[i];
    maxi_start = min(A[i], maxi_start);
  }
 
  for(int i=n-1; i>0; i--){
    T[i] = max(T[i<<1] , T[i<<1|1]);  
  }
 
}
 
void update(ll p, ll val){
 
   p = p + n;
  T[p] = val;


  while(p > 1){
    T[p>>1] = max(T[p], T[p^1]);
    p /= 2;
  }
 
}
 
vector<ll> Tr(){
  return T;
}
 
ll query(ll l, ll r){   
 
 
    l += n;
    r += n;
    ll s = maxi_start - 1;
    for(; l < r; l = l>>1, r = r>>1){
      if(l&1){
       
         s = max(s, T[l]);
         l++;
         
      }
      if(r&1){
        r--;
        s = max(s, T[r]);
        
      
      }
    }
 
    return s;
}
 
 
};

class segT_gcd{
 
private:
vector<ll> T;
ll n;

 
public:  
 
segT_gcd(vector<ll> A){
 
  n = A.size();
  for(int i=0; i<2*n; i++){
    T.push_back(0);
  }
 
  for(int i=0; i<n; i++){
    T[i + n] = A[i];
  }
 
  for(int i=n-1; i>0; i--){
    T[i] = gcd(T[i<<1] , T[i<<1|1]);  
  }
 
}
 
void update(ll p, ll val){
 
   p = p + n;
  T[p] = val;


  while(p > 1){
    T[p>>1] = gcd(T[p], T[p^1]);
    p /= 2;
  }
 
}
 
vector<ll> Tr(){
  return T;
}
 
ll query(ll l, ll r){   
 
 
    l += n;
    r += n;
    ll s =0;
    for(; l < r; l = l>>1, r = r>>1){
      if(l&1){
       
         s = gcd(s, T[l]);
         l++;
         
      }
      if(r&1){
        r--;
        s = gcd(s, T[r]);
        
      
      }
    }
 
    return s;
}
 
 
};





            /*********************************  Union Disjoint Set *******************************/

class DisjointSet{
  map<ll, ll>  parent;
  map<ll, ll> rank;
  map<ll, ll> countInSet;

  public:

  void Makeset(vector<ll> elements){
    for(int i=0; i<elements.size(); i++){
      parent[elements[i]] = elements[i];
      rank[elements[i]] = 0;
      countInSet[elements[i]] = 1;
    }
  }

  ll Find(ll x){
      if(x != parent[x]){
        return Find(parent[x]);
      }
        return x;
      
  }


  void Union(ll x, ll y){
    ll xParent = Find(x);
    ll yParent = Find(y);

    if(xParent == yParent){
      return;
    }


    if(rank[yParent] > rank[xParent]){
      countInSet[yParent] += countInSet[xParent];
      parent[xParent] = yParent;
      
      
    }else if(rank[yParent] < rank[xParent]){
          countInSet[xParent] += countInSet[yParent];
           parent[yParent] = xParent;
          
    }else{
          countInSet[yParent] += countInSet[xParent];
           parent[xParent] = yParent;
           rank[yParent]++;
           
    }
  }

  ll setCount(ll x){
    return countInSet[Find(x)];
  }




};


 /*********************************  Functions and Variable Specific to the Solution *******************************/
void solve(){


ll n, k;
cin >> n >> k;
take(A)


ll cnt = 0;
for(int i=0; i<n; i++){
  if(A[i] == 1){
    YES
  }
}


NO

}
int main(){





ios_base::sync_with_stdio(false), cin.tie(nullptr);




Tstart1

    return 0;
}



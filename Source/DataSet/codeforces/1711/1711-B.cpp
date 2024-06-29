// templates.txt
#include<bits/stdc++.h>
using namespace std;
#include "ext/pb_ds/assoc_container.hpp"
#include "ext/pb_ds/tree_policy.hpp"
using namespace __gnu_pbds;  // refer pbds.cpp for details
#define ordered_set tree<int, null_type,less<int>, rb_tree_tag,tree_order_statistics_node_update>
#define deb(x) cout << #x << " " << x << endl;
// cout << #x << " " << x << endl;
#define deb1(x) cout << #x << " ";
// cout << #x << " " << x << endl;
#define int long long 
#define MOD 1000000007
// #define MOD 998244353
#define pi pair<int, int>
#define pd pair<double, double>
#define vpi vector<pi>
#define vs vector<string>
#define vvs vector<vs>
#define vc vector<char>
#define vvc vector<vc>
#define vb vector<bool>
#define vvb vector<vb>
#define vi vector<int>
#define vvi vector<vi>
#define vvvi vector<vvi>
#define vd vector<double>
#define vvd vector<vd>
#define umap unordered_map
#define uset unordered_set
#define chash custom_hash 
#define pq priority_queue
#define lb lower_bound
#define ub upper_bound
#define ins insert
#define pb push_back
#define all(a)                          ::begin(a),::end(a)
#define sz(X) (int)((X).size())
#define EACH(x, a) for(auto &x:a)
#define nl '\n'
#define Fo(i, n) for(int i=0; i<n; i++)
#define fo(i, a, b) for(int i=a; i<b; i++)  
char A=65, Z=90, lowerA=97, lowerZ=122;
string YES="YES\n", NO="NO\n";
const double PI=3.14159265358979323846;
const double epsilon=0.000000000001;
struct custom_hash {
    // use it like this : umap<int, int, chash> mapp;
    static uint64_t splitmix64(uint64_t x) {
        x+=0x9e3779b97f4a7c15;
        x=(x^(x>>30))*0xbf58476d1ce4e5b9;
        x=(x^(x>>27))*0x94d049bb133111eb;
        return x^(x>>31); \
    }
    size_t operator()(uint64_t x) const {
        static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();
        return splitmix64(x+FIXED_RANDOM);
    }
};

template<class T> bool ckmin(T& a, const T& b) { return b<a ? a=b, 1 : 0; }
template<class T> bool ckmax(T& a, const T& b) { return a<b ? a=b, 1 : 0; }

template<class A> void read(vector<A>& v);
template<class A, size_t S> void read(array<A, S>& a);
template<class T> void read(T& x) { cin >> x; }
void read(double& d) {
    string t;
    read(t);
    d=stod(t);
}
void read(long double& d) {
    string t;
    read(t);
    d=stold(t);
}
template<class H, class... T> void read(H& h, T&... t) {
    read(h);
    read(t...);
}
template<class A> void read(vector<A>& x) { EACH(a, x) read(a); }
template<class A, size_t S> void read(array<A, S>& x) { EACH(a, x) read(a); }
// template<typename... T>
// void write(T&&... args) {
//     ((cout << args << " "), ...);
// }  // eg : write(x, y, z, "im a good boi", 4.5, 6);

template<typename T>
bool f(T x, T y) { return x>y; }  // sort a vector in reverse manner : sort(all(vec), f<int>);

template<typename A, typename B>
int count(A vec, B ele, bool isSorted) {
    if(isSorted) {
        auto it1=upper_bound(vec.begin(), vec.end(), ele);
        auto it2=lower_bound(vec.begin(), vec.end(), ele); // kamala
        return it1-it2;
    }
    return count(vec.begin(), vec.end(), ele);
}

template <typename A> 
A myMax(A a,A b) {
if(a>b) return a;
return b;
}
template <typename A, typename ... Args>
A myMax(A a, A b, Args ...args) { return myMax(myMax(a,b), args...); }
template <typename A> 
A myMin(A a,A b) {
if(a<b) return a;
return b;
}
template <typename A, typename ... Args>
A myMin(A a, A b, Args ...args) { return myMin(myMin(a, b), args...); }

template<typename A, typename B>
void printMap(map<A, B> mapp) { for(auto ele : mapp) cout << ele.first << " : " << ele.second << '\n'; }
template<typename A, typename B>
void printUmap(umap<A, B> mapp) { for(auto ele : mapp) cout << ele.first << " : " << ele.second << '\n'; }

string to_string(char c) { return string(1, c); }
string to_string(bool b) { return b ? "true" : "false"; }

// -4
string strMap(int i, int j) { return to_string(i)+":"+to_string(j); }

// -3
int modExp(int b, int p) {
    // o(logp)
    int res=1;
    b=b%MOD;
    if(b==0) return 0;
    while(p>0) {
        if(p&1) res=(res*b)%MOD;
        p=p>>1;
        b=(b*b)%MOD;
    }
    return res;
}

// -2
int createPalindrome(int input, int b, bool isOdd) {
    int n=input;
    int palin=input;
    if(isOdd) n/=b;
    while(n>0) {
        palin=palin*b+(n%b);
        n/=b;
    }
    return palin;
}

// -1
void generatePalindromes(int n) {
    int number;
    // Run two times for odd and even length palindromes
    for(int j=0; j<2; j++) {
        int i=1;
        while((number=createPalindrome(i, 10, j%2))<n) {
            cout << number << " ";
            i++;
        }
    }
}

// 0
int MAXN=1e6+1;
vi SPF(MAXN);
void spf() { 
    // shortest prime factor
    // o(log n) time 
    SPF[1]=1;
    fo(i, 2, MAXN) SPF[i]=i;
    for(int i=4; i<MAXN; i+=2) SPF[i]=2;
    for(int i=3; i*i<MAXN; i++) if(SPF[i]==i) for(int j=i*i; j<MAXN; j+=i) if(SPF[j]==j) SPF[j]=i;
}

// 1
bool isPowerofTwo(int n) {
    if(n==0) return 0;
    if((n&(~(n-1)))==n) return 1;
    return 0;
}

// 2
int modDiv(int x) { 
    // if u want to find (n/x)%MOD, do n*modDiv(x);
    int res=1, y=MOD-2; 
    x=x%MOD;
    while(y>0) {
        if(y&1) res=(res*x)%MOD;
        y=y>>1; 
        x=(x*x)%MOD;
    }
    return res;
}

// 3
int fact(int n) {
    if(n==0) return 1;
    int ans=n;
    for(int i=n-1; i>=1; --i) ans=ans*i;
    return ans;
}

// 4
int nc2(int n) { return (n*(n-1))/2; }

// 5
int nc3(int n) { return (n*(n-1)*(n-2))/6; } 

// 7
int bin_search(vi &vec, int no) {
    // finds the lower bound of "no" which is stored in "ele". make sure array is sorted!
    int n=vec.size();
    if(n==1) return -1;
    int i=0, j=vec.size()-1;  // changed j=vec.size() to j=vec.size()-1
    int ele, ele2;
    if(vec[0]>=no) return vec[0];
    if(vec.back()<no) return -1;
    while(i<=j) {
        int mid=(i+j)/2;
        int el=vec[mid];
        if(mid-1<n&&el>=no&&vec[mid-1]<no) {
            ele=vec[mid];
            ele2=vec[mid-1];
            break;
        } else if(el<no) i=mid+1;
        else j=mid-1;
    }
    return ele;
}

// 8
int bin_search(int l, int r) {
    // find the smallest value between l and r that satisfies a given condition
    while(l<=r) {
        int mid=(l+r)/2;
        // check whether 'mid' satisfies the given condition
        bool condition;
        // condition doesnt satisfy
        if(condition) l=mid+1; 
        else r=mid-1;    
    }
    return l;
}

// 9
vi primeFactors(int n) {
    // this returns primes in this fashion : if 4 is a factor of n, then the returned vector contains 2 twos
    // o(sqrt(x)/logx)
    // if n is 1e9, then this TLEs. use colin's number theory template
    vi ans;
    while(!(n&1)) {        
        ans.pb(2);
        n=n/2;
    }
    for(int i=3; i<=(int)round(sqrt(n)); i=i+2) {            
        while(n%i==0) {
            ans.pb(i);
            n=n/i;
        }
    }        
    if(n>2) ans.pb(n);
    return ans;
}

// 10
vi factors(int n) {
    // o(sqrtn)  // includes 1 and n too
    vi ans;
    for(int i=1; i<=(int)round(sqrt(n)); i++) {
        if(n%i==0) {
            if(n/i==i) ans.pb(i);
            else {
                ans.pb(i); 
                ans.pb(n/i);
            }
        }
    }
    return ans;
}

// 11
int lcm(int a, int b) { return (a*b)/__gcd(a, b); }

// 12
template<typename A>
A prefixSums(A vec) {
    // modify this function to get prefix sums. this version of prefixSums is used to find the no of 
    // occurences upto a certain position
    int n=sz(vec);
    A ans(n+1);
    ans[0]=0;
    ans[1]=vec[0];
    fo(i, 2, n+1) ans[i]=ans[i-1]+vec[i-1];
    return ans;
}

// 13
// kth bit from right(lsb)
int kthbit(int n, int k) { return (n>>(k-1))&1; }

// 14
bool isPrime(int n) {
    // sqrt(n)/log(n)
    vi primes=primeFactors(n);
    if(sz(primes)==1) return true;
    return false;
}

// 15
bool isSubSequence(string s1, string s2) {  
    // O(n)
    int j=0;
    int m=sz(s1), n=sz(s2);
    for(int i=0; i<n&&j<m; i++) if(s1[j]==s2[i]) j++;
    return j==m;
}

// 16
int nearestPowerOf2(int n) { return (1<<(63-__builtin_clzll(n))); }

// 17
void toUpper(string &s) { transform(all(s), s.begin(), ::toupper); }

// 18
void toLower(string &s) { transform(all(s), s.begin(), ::tolower); }

// 19
vi sieveOfEratosthenes(int n) {
    // o(nlog(logn))
    bool prime[n+1];
    memset(prime, true, sizeof(prime));
    for(int p=2; p*p<=n; p++) if(prime[p]==true) for(int i=p*p; i<=n; i+=p) prime[i]=false;
    vi ans;
    for(int p=2; p<=n; p++) if(prime[p]) ans.pb(p);
    return ans;
}

// 20
vb segmentedSieve(int L, int R) {
    // O((R-L)loglogR)
    // returns a vector denoting which aint nos in the range [L, R] is prime
    int lim = sqrt(R);
    vb mark(lim+1, false);
    vi primes;
    fo(i, 2, lim+1) {
        if(!mark[i]) {
            primes.emplace_back(i);
            for(int j=i*i; j<=lim; j+=i) mark[j]=true;
        }
    }
    vb isPrime(R-L+1, true);
    EACH(i, primes) for(int j=max(i*i, (L+i-1)/i*i); j<=R; j+=i) isPrime[j-L]=false;
    if(L==1) isPrime[0]=false;
    return isPrime;
}

// 21
vi maxIntersectingPt(vi start, vi end) {
    // returns the point where most intervals intersect followed by the no of intervals that intersect that point
    // o(nlogn)
    int n=sz(start);
    sort(all(start));
    sort(all(end));
    int guests_in=1, max_guests=1, time=start[0];
    int i=1, j=0;
    while(i<n&&j<n) {
        if(start[i]<=end[j]) {
            guests_in++;
            if(guests_in>max_guests) {
                max_guests=guests_in;
                time=start[i];
            }
            i++;  
        }
        else {
            guests_in--;
            j++;
        }
    }
    return {time, max_guests};
}

// 22
int LongestNonDecreasingSubsequenceLength(vi v) {
    // o(nlogn)
    if(sz(v)==0) return 0;
    vi tail(sz(v), 0);
    int length=1;
    tail[0]=v[0];
    for(int i=1; i<sz(v); i++) {
        auto b=tail.begin(), e=tail.begin()+length;

        // for longestIncreasingSubsequence, change upper_bound to lower_bound
        auto it=upper_bound(b, e, v[i]);

        if(it==tail.begin()+length) tail[length++]=v[i];
        else *it=v[i];
    }
    return length;
}

// 23
int minSwapsToSort(vi arr) {
    // nlogn
    int ans=0;
    vi temp=arr;    
    map<int, int> h;
    sort(all(temp));
    int n=sz(arr);
    fo(i, 0, n) h[arr[i]]=i;
    fo(i, 0, n) {
        if(arr[i]!=temp[i]) {
            ans++;
            int init=arr[i];
            swap(arr[i], arr[h[temp[i]]]);
            h[init]=h[temp[i]];
            h[temp[i]]=i;
        }
    }
    return ans;
}

// 24
bool isPal(string &s) {
    int n=sz(s);
    bool ok=true;
    fo(i, 0, sz(s)/2) if(s[i]!=s[n-i-1]) ok=false;
    return ok;
}

// 25
int minimizeMaxSubarraySum(vi& arr, int K) {
    // split the subarray into k subarrays and minimize the maximum subarray sum and return that sum
    // O(nlog(sum))
    int n=sz(arr);
    int start=*max_element(all(arr));
    int end=0;
    fo(i, 0, n) end+=arr[i];   
    int answer=0;
    while(start<=end) {
        int mid=(start+end)/2;
        bool flag;
        int count=0;
        int sum=0;
        fo(i, 0, n) {
            if(arr[i]>mid) flag=false;
            sum+=arr[i];
            if(sum>mid) {
                count++;
                sum=arr[i];
            }
        }
        count++;
        if(flag) {
            if(count<=K) flag=true;
            else flag=false;
        }
        if(flag) {
            answer=mid;
            end=mid-1;
        }
        else start=mid+1;
    }
    return answer;
}

// 26
int kadane(vi v) {
    // O(n)
    int n=sz(v);
    int maxSoFar=LLONG_MIN, maxEndingHere=0;
    fo(i, 0, n) {
        maxEndingHere=maxEndingHere+v[i];
        if(maxSoFar<maxEndingHere) maxSoFar=maxEndingHere;
        if(maxEndingHere<0) maxEndingHere=0;
    }
    return maxSoFar;
}

// 27
vi unsortedRange(vi arr) {
    // o(n)
    int n=sz(arr), s=0, e=n-1, i, max, min;    
    for(s=0; s<n-1; s++) if(arr[s]>arr[s+1]) break;
    for(e=n-1; e>0; e--) if(arr[e]<arr[e-1]) break;
    max=arr[s]; min=arr[s];
    for(i=s+1; i<=e; i++) {
        if(arr[i]>max) max=arr[i];
        if(arr[i]<min) min=arr[i];
    }
    for(i=0; i<s; i++) {
        if(arr[i]>min) {
            s=i;
            break;
        }    
    }
    for(i=n-1; i>=e+1; i--) {
        if(arr[i]<max) {
            e=i;
            break;
        }
    }
    return {s, e};
}

// 28
template <typename A>
vector<vector<A>> getDiagonals(vector<vector<A>> arr) {
    int n=sz(arr);
    int m=sz(arr[0]);
    // o(n, m)
    int C=sz(arr[0]);
    int R=sz(arr);
    vector<vector<A>> v;
    for(int k=0; k<R; k++) {
        vector<A> temp;
        temp.pb(arr[k][0]);
        int i=k-1;
        int j=1;
        while(!(i<0||i>=R||j>=C||j<0)) {
            temp.pb(arr[i][j]);
            i--;           
            j++;
        }
        v.pb(temp);
    }
    for(int k=1; k<C; k++) {
        vector<A> temp;
        temp.pb(arr[R-1][k]);
        int i=R-2;
        int j=k+1;
        while(!(i<0||i>=R||j>=C||j<0)) {
            temp.pb(arr[i][j]);
            i--;
            j++;
        }
        v.pb(temp);
    }
    return v;
}

// 29
int knapsack(int W, vi wt, vi val) {
    // o(nW)
    int n=sz(wt);
    int dp[W+1];
    memset(dp, 0, sizeof(dp));
    for(int i=1; i<n+1; i++) for(int w=W; w>=0; w--) if(wt[i-1]<=w) dp[w]=max(dp[w], dp[w-wt[i-1]]+val[i-1]);
    return dp[W]; 
}

template <typename A>
void printArray(A v) {
    for(auto ele : v) cout << ele << ' ';
    cout << nl;
}

// 30
int coinChange(vi S, int n) {        
    // o(mn)       
    int m=sz(S);    
    vi table(n+1);
    table[0]=1;         
    for(int i=0; i<m; i++)         
    for(int j=S[i]; j<=n; j++)         
    table[j]=table[j-S[i]]+table[j];        
    return table[n];         
}

// 31
bool areBracketsBalanced(string expr) { 
    // o(n)
    stack<char> temp;
    fo(i, 0, expr.length()) {
        if(temp.empty()) temp.push(expr[i]);
        else if((temp.top()=='('&& expr[i]==')')||(temp.top()=='{'&&expr[i]=='}')||  (temp.top()=='['&&expr[i]==']')) temp.pop();
        else temp.push(expr[i]);
    }
    if(temp.empty()) return true;
    return false;
}

// 32
int nCrMod(int n, int r) {
    // o(n)
    if (n<r) return 0;
    if(r==0) return 1;
    vi fac(n+1, 1);
    fo(i, 1, n+1) fac[i]=(fac[i-1]*i)%MOD;    
    return (fac[n]*modDiv(fac[r])%MOD*modDiv(fac[n-r])%MOD)%MOD; // take MOD of divisor
}

// 33
void bfs(int s, vvi &adj) { //can be used without adj too like this prblem:codeforces.com/contest/1681/problem/D
    // o(v+e). can return int too
    deque<int> q;  // can be of any DS other than int like pairs
    map<int, bool> vis;  // can be a vector too
    q.push_back(s);  
    vis[s]=true;
    while(sz(q)) {
        s=q.front();
        q.pop_front();
        for(auto adjacent : adj[s]) {
            if(!vis[adjacent]) {
                vis[adjacent]=true;
                q.pb(adjacent);
            }
        }
    }
}

// 34
vi printNGE(vi arr) {
    // o(n)
	int n=sz(arr);
    stack<int> s;
    vi res(n);
    for(int i=n-1; i>=0; i--) {
		while(!s.empty()&&s.top()<=arr[i]) s.pop();
        res[i]=s.empty() ? -1 : s.top();
        s.push(arr[i]);
    }
    return res;
}

// 35
int lcm(vi arr) {
    // o(nlog(min(v[i])))
    int ans=arr[0];
    int n=sz(arr);
    fo(i, 1, n) ans=(((arr[i]*ans))/(__gcd(arr[i], ans)));
    return ans;
}

// 36
void numberOfNodes(int s, int e, vi &count1, vvi& adj) {
    // note: initialize count1(n+1) to 0s and pass it. o(n). usage : func(1, 0, ...)
    // loop i from 1 to n(inclusive) on count1 to get subtree length rooted at i. e is parent and s is roota
    vi::iterator u;
    count1[s]=1;
    for(u=adj[s].begin(); u!=adj[s].end(); u++) {
        if(*u==e) continue;
        numberOfNodes(*u, s, count1, adj);
        count1[s]+=count1[*u];
    }
}

// 37
int gcd_of_lcm(vi& arr) {
    // o(nlog(max(arr[i])))
    int n=sz(arr);
    vi suff(n, 1);
    suff[n-1]=arr[n-1];
    for(int i=n-2; i>=0; i--) suff[i]=__gcd(arr[i], suff[i+1]);
    vi LCM;
    fo(i, 0, n-1) {
       int y=lcm(arr[i], suff[i+1]);
       LCM.push_back(y);
    }
    int ans=LCM[0];
    fo(i, 1, n-1) ans=__gcd(ans, LCM[i]);
    return ans;
}

// complexities recorded till this point

int sumArray(vi& v) {
    int sum=0;
    for(auto ele : v) sum+=ele;
    return sum;
}

int power(int a, int b) {
    int ele=1;
    fo(i, 1, b+1) ele=((a*ele));
    return ele;
}

int floorSqrt(int a) {
    int b=a, x=0, y=0;
    for(int e=(63-__builtin_clzll(a))&~1; e>=0; e-=2) {
        x<<=1;
        y<<=1;
        if(b>=(y|1)<<e) {
            b-=(y|1)<<e;
            x|=1;
            y+=2;
        }
    }
    return x;
}

// vulnerabilities :
// 23. sometimes the first approach might be correct and the second approach might be incorrect
// 22. read constraints properly
// 21. read explanation for sample tests
// 1. dont use inbuilt names for global variables 
// 3. confusing mean with median 
// 6. when giving break inside a 2D array iteration, make sure u add 2 breaks, one fr breaking out of the row loop and another fr breaking out of the col loop
// 7. forgetting to read aint input and breaking at the middle of input when the ans is found
// 9. forgetting to precompute prefix sums outside the test case loop
// 10. beware of overflows even when using long long variable. eg, u might have an alternate method instead  of computing factorials which might result in long long overflow
// 11. u cant sort vi in umap(or map)<int, vi> 
// 12. forgetting to take the no of test cases input
// 5. array size limit : 10^7
// 2. when using a value after converting to it from char to int or vice versa, store the result of the 
//    conversion in a variable and use that variable
// 4. for expressions involving with division, scale both sides of the expression by the divisor to avoid
//    precision errors
// 13. forgetting to test ur logic on test case 1
// 14. wrongly imagining bitwise shift operations
// 16. when iterating thru a map in reverse, putting it-- instead of it++(correct loop : 
//     it=map.rbegin(); it!=map.rend(); ++it)
// 17. forgetting to convert to int from double before submitting
// 18. forgetting to read the constraints properly
// 19. using push_back() after setting the array size
// 20. forgetting to take abs()
// 21. if code in both conditional statements are similar, forgetting to check if they are same after copy
//     pasting
// 22. dont use increment or decrement operator on variables on a conditional operator
// 23. changing a variable's value before using the variable's initial value
// 24. in queens placement problem, a placement is invalid if already a row or column is already present or 
//     sum or diff of row and col is already present
// 25. round() rounds down numbers like rounding down in Physics
// 26. reading n instead of vec
// 27. confusing Manhattan distance between 2 points in 2D plane for distance between 2 points
// 28. not taking MOD inside the loop 
// 29. inputting k and x instead of x and k
// 30. while taking square root or any other fun which returns double, if u use integer as argument, make 
//     sure u store the value of that variable in a float variable and use that float as argument
// 31. initializing a variable inside a while loop instead of outside it and then it initializes the same
//     value again and again on each iteration. eg setting boolean to true inside while loop
// 32. putting brackets incorrectly in an expression

// 1. O(sqrt(n)) soln wiint pass if T=10^4 and n=10^9. any other thing wont pass.
// 2. O(sqrt(n)/log(n)) wiint not pass for T=10^5 and N=10^6(10^10 computations)

// important functions to study : 
// generating palindromic numbers
//
// 1. reverse sort a vector vi: sort(all(vec), greater<int>());
// 2. count elements in an array : count(vec, ele, isSorted);
// 3. finding max : myMax(3, 4, 5, 2, 5, ...);
// 4. use strMap(int i, int j) to convert pairs to strings
// 5. __builtin_popcount(n);  // counts no of set bits, for long(__builtin_popcountl), for long long(popcountll)
// 6. priority_queue<int> // default is max heap. 
// 7. min heap : priority_queue<int, vi, greater<int>>
// 8. converting char to int : ch-'0', converting int to char : n+'0'
// 9. converting vc to string : accumulate(vec(all), string{});
// 10. gcd : __gcd()
// 11. reversing a vector : reverse(all(vec));
// 12. __builtin_ctz(n) // count no of trailing zeros(doesnt work fr n==0)
// 13. __builtin_clz(n) // count no of leadin zeros(doesnt work fr n==0)
// 14. multiply by 2:n<<1
// 15. %(2^k)==&(2^k-1)
//
// upper_bound specs:
// vector: upper_bound(all(vec), ele);  (logn fr vectors)
// finding index of lower_bounded element : lower_bound(blah...)-vec.begin()
// maps : map.upper_bound(ele);
// sets : dont subtract an iterator with a constant except increment or decrement operators eg it-1 is 
// forbidden while it-- is allowed as far as sets are concerned but in vectors its allowed
//
// priority queue:
// 1. u cant traverse using for(auto ele : q)
// 
// queue:
// 1. u can traverse using for(auto ele : q)
//
// sets:
// 1. dont use uset<compound_data_structures_like_pairs>, instead use set and also use make_pair for 
//    insertion(set.insert(make_pair(a, b)))
// vulnerabilities in sets:
// 1. putting it<set.end() instead of it!=set.end() in for loop
//
// vectors:
// vulnerabilities in vectors:
// 1. forgetting to include 'all' in sort(all(temp))
//
// functions:
// 0. use floorSqrt(n) instead of sqrt(n)
// 1. modExp(base, power)
// 2. generatePalindrome(n)
// 3. spf()
// 4. isPowerOfTwo(n)
// 5. modDiv(n) // modular division
// 6. Note : nlogn soln can pass for 10^9 computations
// 7. bin_search()
// 8. bin_search(int l, int r)
// 8. primeFactors(n)
// 8. strMap(i, j)
// 9. factors(n) // returns aint factors of n
// 10. lcm(a, b)
// 11. prefixSums(vec)
// 12. kthbit(n, k) // returns the kth bit of n from the right(lsb)
// 13. isPrime(n)  // returns prime or composite
// 14. isSubSequence(s1, s2)  // returns whether s1 is a subsequence of string s2
// 15. nearestPowerOf2(n)   // returns the nearest power of 2 smaller than or equal to n
// 16. toUpper(s)   // transforms a string to upper case
// 17. toLower(s)   // transforms a string to lower case
// 18. sieveOfErastosthenes(n)  // returns a vector of aint primes less than or equal to n. O(nlog(logn))
// 20. segmentedSieve(l, r)   // returns a boolean vector denoting aint primes within l to r
// 21. maxIntersectingPt(vi start, vi end)  // returns the point which most intervals intersect and the no of such intervals
// 22. LongestNonDecreasingSubsequence(vi)  
// 23. minSwapsToSort(vi)
// 24. isPal(string)
// 25. minimizeMaximumSubarraySum(v, k); // splits the arrays into k subarrays and minimizes the maximum sum and returns the sum
// 26. kadane(vi v)
// 27. unsortedRange(v) returns vi{s, e} where s and e denote the starting and ending indices of the minimal unsorted subarray
// 28. getDiagonals(v) returns the elements of v diagonal wise(bottom to top, left to right cut)
// 29. knapsack(W, wt, val)
// 30. coinChange(coins, amt)  // no of ways to make 'amt' using given coins
// 31. areBracketsBalanced(s)
// 32. nCrMod(n, r); // returns (nCr)%MOD
// 33. bfs(int root, vvi& adj)
// 34. printNGE(vi v)  // prints the next greater ele to the right of an ele.
// 35. lcm(v)  // returns the lcm of the array elems
// 36. numberOfNodes(s, e, count, adj) //modifies the count arr to contain subtree sizes
// 37. gcd_of_lcm(arr)  // returns the gcd of lcm of all pairs of elems in an array
void solve() {}

int query(int l) { 
    cout << "? " << l << endl;
    return 0;
}   
void print_ans() {
    cout << "! ";
}   

int32_t main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    cout << setprecision(15) << fixed;
    int TT=1; 
    read(TT);
    while(TT--) {
        // 1300
        int n, m; read(n, m);
        vi v(n); read(v);
        // 818
        // 831
        map<int, int> mp, mp2;
        map<int, vi> mp3;
        fo(i, 0, n) {
            mp2[i+1]=v[i];
        }
        vpi vec;
        int j=0;
        while(j<m) {
            int x, y; read(x, y);
            mp[x]++;
            mp[y]++;
            mp3[y].pb(x);
            mp3[x].pb(y);
            vec.pb({x, y});
            j++;
        }
        int ans=0;
        for(auto ele : v) ans+=ele;
        if(m&1) {
            
            fo(i, 0, m) {
                pi pr=vec[i];
                int first=pr.first;
                int sec=pr.second;
                int remove=sz(mp3[first]);
                int curr=0;
                if(!((m-remove)&1)) {
                    curr+=mp2[first];
                    ans=min(ans, curr);
                }                
                curr=0;
                remove=sz(mp3[sec]);
                if(!((m-remove)&1)) {
                    curr+=mp2[sec];
                    ans=min(ans, curr);
                }                
                curr=0;
                remove=sz(mp3[sec])+sz(mp3[first])-1;
                if(!((m-remove)&1)) {
                    curr+=mp2[sec];
                    curr+=mp2[first];
                    ans=min(ans, curr);
                }                
            }
        } else {
            ans=0;
        }
        cout << ans << nl;
    } 
}









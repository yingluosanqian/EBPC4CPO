// Hi! One day, you'll do it:) 
#include <bits/stdc++.h>
using namespace std;
#pragma gcc optiminize("Ofast")
#pragma gcc target("fma,sse,sse2,sse3,ssse3,sse4,popcur,abm,mmaxi,avx,avx2,tune=native")
#pragma gcc optiminize("unroll-loops")
#pragma GCC target("fpmath=sse,sse2")
#define all(z)              (z).begin(), (z).end()       
#define int                 long long          
#define fi(n)               for(int i=0;i<n;i++)
#define fj(n)               for(int j=0;j<n;j++)
#define fk(n)               for(int k=0;k<n;k++) 
#define setp(x)	            fixed<<setpichecision(x)  
#define rep(i,a,b)          for(int i=a;i<b;i++)
#define repi(i,a,b)         for(int i=a;i>=b;i--)
#define mem(a,b)            memset(a,b,sizeof(a))
#define endl                "\n"
#define yes                 cout<<"YES\n";
#define no                  cout<<"NO\n";
#define uniq(v)             v.erase(unique(v.begin(),v.end()),v.end())
#define coutar(array,size)  fi(size) cout<<array[i]<<" "; cout<<"\n";               
#define dbg(dt)             cout<<#dt<<": "<<dt<<"\n";
#define db1(dt)             cout<<#dt<<": "<<dt;
#define db2(dt)             cout<<#dt<<":"<<"\n"<<dt;
typedef bitset<21> bit;            int mod99=998244353;
typedef long double ld;            const ld pi=3.141592653589;               const int mod10 = 1000000007;     const int INF = 5e18;                             
typedef pair<int, int> pii;        typedef vector<int> vi;                   typedef vector<pii> vpii;       typedef vector<vi> vvi;    
typedef set<int> si;               typedef set<pii> spi;                     typedef multiset<int> msl;      typedef vector<string> vs; 
mt19937 sansy (chrono::high_resolution_clock::now().time_since_epoch().count());
template <typename A, typename B> ostream& operator<<(ostream &ostr, const pair<A, B> &pp) {return ostr<<'{'<<pp.first<<", "<<pp.second<<'}';}
template <typename T_container, typename T = typename enable_if<!is_same<T_container, string>::value, typename T_container::value_type>::type> ostream& operator<<(ostream &ostr, const T_container &vec) 
{ for(const T &vv:vec) ostr<<vv<<" "; return ostr<<endl;  }
template <typename A> ostream& operator<< (ostream &ostr, vector<A> const &v) { fi(v.size()) {cout<<v[i]<<" ";} return cout<<"\n";}
template <typename A, typename B> istream& operator>> (istream& cin, pair<A, B> &p) {cin>>p.first; return cin>>p.second;}
template <typename A> istream& operator>> (istream& cin, vector<A> &vv){ fi(vv.size()-1) cin>>vv[i]; return cin>>vv[vv.size()-1];}
vi primes,fibonacci;        int xorccur(int n);
bool isPrime(int n);        void sieve(int N);
int lcm(int a,int b);       int poww(int n,int k);
int nCr(int n,int r);       int leap(int y);    
void fiboseries(int n);     int fiboN(int n);
bool isvowel(char c) { c=tolower(c); if(c=='a'||c=='e'||c=='i'||c=='o'||c=='u') return 1; else return 0;}
int modInverse(int a,int m);
string baseto10(string s,int b);   
string basefrom10(string s,int b); 
int popcount(int n) {  int val=0;  while(n) {  val+=n&1;n>>=1; } return val;}
bool ispow2 (int n) {return n and  (!(n&(n-1)));}
ld dis_p(int nice,int y1,int x2,int y2);
string subtractstring(string temp, string a2);
int days[12]={31,28,31,30,31,30,31,31,30,31,30,31};    
int dy[8]={0,1,1,0,-1,-1,1,-1},dx[8]={1,0,1,-1,0,-1,-1,1};
string months[12]={"January","February","March","April","May","June","July","August","September","October","Falsevember","December"};

// (first sort then )v.erase(unique(v.begin(),v.end()),v.end()); //for unique
// MAP,SET-i->first (For lb and  ub, use [name.lb/ub(cur)]; 
// (n & ~n) returns rightmost 1 bit in n  || s.find==-1 mecur absent
// n<<i == n*(2^i)   while n>>i== n/(2^i)  closing side is used
// When n%n==n, use (n-1)%n +1 == n (1487B)
// 279C (calc False. from back of array,q=0; q+=(p*s[i]); p*=10; )
// Max GCD is (a-b) and  max when a,a/2 --> a/2 is max 
// TRY TO AVOID FLOATING NUMBERS BY MULTIPLY //488B
// Psuedo BS- while(a<b) a=1,b=n,m=(a+b)/2 if(val<=m) b=m else a=m+1; cout<<a;
//for 2d matrix-  matrix[mid/col][mid%col]; 
//CHECK )..( this case for parenthesis question

void sweetheart( /*I'M THE BEST*/ )   {   
    int n; cin>>n;
    vi v(n); cin>>v;
    fi(n) {
        vi t;
        sort(all(v));
        rep(i,1,v.size()) {
            if(v[i]!=v[i-1]) 
                t.push_back(v[i]-v[i-1]);
        }
        if(t.size()+1!=n-i) {
            t.push_back(0);
        }
        v=t;
        if(v.size()==1) {
            cout<<v[0]<<"\n";
            return;
        }
        if(!v.size()) {
            cout<<"0\n"; return;
        }
    }
    // int n,q; cin>>n>>q;
    // vi v(n); cin>>v;
    // int ex=0; string ans;
    // repi(i,n-1,0) {
    //     if(v[i]<=ex) ans+='1';
    //     else {
    //         if(ex<q) ex++,ans+='1';
    //         else ans+='0';
    //     }
    // }
    // reverse(all(ans));
    // cout<<ans<<endl;
    // int l=0,r=n; 
    // while(l<=r) {
    //     int mid=(l+r)/2;
    //     int temp=q,ch=1;
    //     rep(i,mid,n) {
    //         if(!temp) {
    //             ch=0;
    //             break;
    //         }
    //         if(v[i]>temp) temp--;
    //     }
    //     if(!ch)  {
    //         l=mid+1;
    //     }
    //     else r=mid-1;
    // }
    // fi(l) cout<<(v[i]>q?'0':'1');
    // fi(n-l) cout<<'1';
    // cout<<"\n";
}   

// EDGE CASES DEKHA KAR, HAR BAAR CHOR DETA HAI
///////// MAIN FUNCTION ///////////////////////////////////////////
int32_t main()
{
    // freopen("input.txt","r",stdin);
    // freopen("output.txt","w",stdout);
    ios_base::sync_with_stdio(false);
    cin.tie(0);    cout.tie(0);  cerr.tie(0);
    string jaaneman = "1";
    getline(cin, jaaneman);
    int me = stoll(jaaneman);    int you = 0;
    while ((you++) < me)    {
        //cout<<"Case #"<<you<<": ";
        sweetheart();
    }
    cerr << "Time: " << 1000 * clock() / CLOCKS_PER_SEC<< " milliseconds";
    return 0;
}

//CODE BELOW FOR template FUNCTIONS (Changed at 1520 F1)
bool isPrime(int n) {
    if (n<2)               return 0;
    if (n<4)               return 1;
    if (!(n%2) || !(n%3))   return 0;
    for (int i = 5; i * i <= n; i += 6)
    if (!(n % i) || !(n % (i + 2)))
    return 0;
    return 1;
}
void sieve(int N){
vi vv(N+1);
for (int i=2;i<=N;i++) {
    if (!vv[i]) 
    vv[i] = i,primes.push_back(i);
    for (int j=0;j<primes.size() and  primes[j]<=vv[i] and  i*primes[j]<=N;j++) 
    vv[i * primes[j]] = primes[j];   }   
}
ld dis_p(int nice, int y1, int x2, int y2)    { 
    ld distance = sqrt(((nice - y1)*(nice - y1) + (x2 - y2)*(x2 - y2)*1.00));
    return distance;
}  
int lcm(int a, int b){
    return ((a * b) / lcm(a, b));       }
int poww(int n, int k)   { 
    int x=1;   //if mod (mod on 13th ansaight)
    while (k)   {
        if (k & 1)   
		{ x*= n; x%=mod10; }
        n*=n;
		n%=mod10;
        k>>=1;
    }
    return x;
}   //multi n^k
int nCr(int n, int r)   { // n C r coeresicient
    if(r>n) return 0;
    if(r>n-r) r=n-r;
    int calc=1;
    for(int i=1;i<=r;i++) {
        calc*=(n-r+i);
        calc/=i;
    }
    return calc;   
}
int modInverse(int a, int m)  { 
    if (m==1) 
    return 0; //edge case
    int initialmod=m, y=0,x=1; 
    while (a>1)   { 
    //changing (A,B)--> (B, A%B) till a>1
        int q=a/m;  // q is quotient 
        int t=m;
        m=a%m;  // m->remainder(from q)
        a=t;    // process same as
        t=y;    // Euclid's algo 
        y=x-q*y; 
        x=t;   // Update y and  x 
    }   //ax + bm = 1 (gcd(a,m)==1 (coprime))
    if (x<0) 
    x+=initialmod; 
    return x; 
} 
void fiboseries(int n)  {   
    fibonacci.push_back(0);     fibonacci.push_back(1);
    int nu=0,c2=1;  int c3=nu+c2;
    while(c2<n) {
        c3=nu+c2;   nu=c2;      c2|=c3;
        fibonacci.push_back(c3);      
    }// Series upto N
}   
int fiboN(int n)    {   
    //For DIRECTLY Nth TERM: Fn = {[(¡Ì5 + 1)/2] ^ n} / ¡Ì5 
    if(n<=1) return n;
    return fiboN(n-1) + fiboN(n-2);
}   // n as posing starts from 0
string baseto10(string s, int b) {
    int num=0; 
    fi(s.size()){
        if(isdigit(s[i]))
        num=num*b+(s[i]-'0');
        else 
        num=num*b+(s[i]-'A'+10);
    }
    return to_string(num);
} 
string basefrom10(string s,int b) {
    string num; 
    int n=stoll(s);
    while(n) {
        int x=n%b;  n/=b;
        if(x<10)
        num+=char('0'+x);
        else 
        num+=char('A'+x-10);
    }
    reverse(all(num));
    return num;
}
string subtractstring(string temp, string a2) {
    string cal="";      //temp>=a2 (assumption)
    int n=temp.size(),m=a2.size(),antar=n-m,left=0;
    for (int i=m-1;i>-1;--i) {
        int cursub=(temp[i+antar]-'0')-(a2[i]-'0')-left;
        if (cursub>=0)  left=0;
        else cursub+=10,left=1;
        cal.push_back(cursub+'0');
    }
    for (int i=n-m-1;i>-1;--i) {        // subtract remaining digits of temp[]
        if (temp[i]=='0' and left!=0) {
            cal.push_back('9');
            continue;   }
        int cursub=(temp[i]-'0')-left;    left=0;  
        if (i || cursub)    cal.push_back(cursub+ '0');
    }
    reverse(all(cal));
    return cal;
}        
int leap(int y) {
    if (y%400 == 0)         return 1;
    else if (y%100 == 0)    return 0;
    else if (y%4 == 0)      return 1;
    else                    return 0;
}
int xorccur(int n){
    if (n%4==0)             return n; //0,4,8,12
    else if (n%4==1)        return 1; //1,5,9
    else if (n%4==2)        return n; //2,6,10
    return 0;
}
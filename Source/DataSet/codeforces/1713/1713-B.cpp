#include <bits/stdc++.h>
using namespace std;
#define int long long int
bool prime[1000007];
//vector<int> fact(1000005);
#define INF 1e9;

#ifndef ONLINE_JUDGE
#define debug(x) cerr<< #x <<"= "; _print(x);cerr<<endl;
#else
#define debug(x)
#endif

#define reverse(v) reverse(v.begin(),v.end());
#define ascending(v) sort(v.begin(),v.end());
#define descending(v) sort(v.rbegin(),v.rend());
void _print(string a) {cerr << a;} void _print(int a) {cerr << a;} void _print(char a) {cerr << a;}
void _print(bool a) {cerr << a;} void _print(double a) {cerr << a;} void _print(float a) {cerr << a;}
template<class T> void _print(unordered_map<string, T> m)
{
    cerr << "\n";
    for (auto &pr : m)
    {
        cerr << pr.first << "->";
        cerr << pr.second << "\n";
    }
}
template<class T, class V> void _print(unordered_map<T, vector<V>> m)
{
    cerr << "\n";
    for (auto &pr : m)
    {
        cerr << pr.first << "->";
        for (auto i : pr.second)
        {
            _print(i);
            cerr << " ";
        }
        cerr << "\n";
    }
}

template<class T, class V> void _print(unordered_map<T, V> m)
{
    cerr << "\n";
    for (auto &pr : m)
    {
        _print(pr.first);
        cerr << "->";
        _print(pr.second);
        cerr << "\n";
    }
}
template<class T> void _print(vector<T> v1)
{
    cerr << "[";
    for (T i : v1)
    {
        _print(i);
        cerr << " ";
    }
    cerr << "]";
}
template<class T> void _print(vector<pair<T, T>> v1)
{
    cerr << "[";
    for (auto &pr : v1)
    {
        cerr << "{";
        _print(pr.first);
        cerr << " ";
        _print(pr.second);
        cerr << "}";
        cerr << " ";
    }
    cerr << "]";
}
template<class T> void _print(vector<vector<T>> v1)
{
    cerr << "\n";
    for (auto &vec : v1)
    {
        for (T i : vec)
        {
            _print(i);
            cerr << " ";
        }
        cerr << "\n";
    }
}

template<class T> void _print(vector<set<T>> v1)
{
    for (auto &vec : v1)
    {
        for (T i : vec)
        {
            _print(i);
            cerr << " ";
        }
        cerr << "\n";
    }
}
template<class T> void _print(set<T> s1)
{
    cerr << "[";
    for (T i : s1)
    {
        _print(i);
        cerr << " ";
    }
    cerr << "]";
}
template<class T> void _print(queue<T> q1)
{
    cerr << "[";
    while (!q1.empty())
    {
        T i = q1.front();
        q1.pop();
        cerr << i;
        cerr << " ";
    }
    cerr << "]";
}
template<class T> void _print(stack<T> st1)
{
    cerr << "[";
    while (!st1.empty())
    {
        _print(st1.top());
        st1.pop();
        cerr << " ";
    }
    cerr << "]";
}
template<class T> void _print(list<T> lis1)
{
    cerr << "[";
    for (T i : lis1)
    {
        _print(i);
        cerr << " ";
    }
    cerr << "]";
}
template<class T> void _print(deque<T> lis1)
{
    cerr << "[";
    for (T i : lis1)
    {
        _print(i);
        cerr << " ";
    }
    cerr << "]";
}
template<class T, class V> void _print(deque<pair<T, V>> lis1)
{
    cerr << "[";
    for (auto i : lis1)
    {
        cerr << "{";
        _print(i.first);
        cerr << " ";
        _print(i.second);
        cerr << "}";
    }
    cerr << "]";
}
template<class T> void _print(priority_queue<T> pq1)
{
    cerr << "[";
    for (T i : pq1)
    {
        _print(i);
        cerr << " ";
    }
    cerr << "]";
}
int moduloMultiplication(int a, int b,
                         int  mod)
{

    // Initialize result
    int  res = 0;

    // Update a if it is more than
    // or equal to mod
    a %= mod;

    while (b) {

        // If b is odd, add a with result
        if (b & 1)
            res = (res + a) % mod;

        // Here we assume that doing 2*a
        // doesn't cause overflow
        a = (2 * a) % mod;
        b >>= 1; // b = b / 2
    }
    return res;
}
int modpower(int x, int y, int p)
{
    int res = 1;     // Initialize result

    x = x % p; // Update x if it is more than or
    // equal to p

    if (x == 0) return 0; // In case x is divisible by p;

    while (y > 0)
    {
        // If y is odd, multiply x with result
        if (y & 1)
            res = (res * x) % p;

        // y must be even now
        y = y >> 1; // y = y/2
        x = (x * x) % p;
    }
    return res;
}
int power(int x, int y)
{
    int res = 1;     // Initialize result

    if (x == 0) return 0; // In case x is divisible by p;

    while (y > 0)
    {
        if (y & 1)
            res = (res * x);

        // y must be even now
        y = y >> 1; // y = y/2
        x = (x * x);
    }
    return res;
}
// C++ function for extended Euclidean Algorithm
int gcdExtended(int a, int b,
                int* x,
                int* y);

// Function to find modulo inverse of b. It returns
// -1 when inverse doesn't exists
int modInverse( int b,  int m)
{

    int x, y; // used in extended GCD algorithm
    int g = gcdExtended(b, m, &x, &y);

    // Return -1 if b and m are not co-prime
    if (g != 1)
        return -1;

    // m is added to handle negative x
    return (x % m + m) % m;
}

// C function for extended Euclidean Algorithm (used to
// find modular inverse.
int gcdExtended(int a, int b,
                int* x,
                int* y)
{

    // Base Case
    if (a == 0) {
        *x = 0, *y = 1;
        return b;
    }

    // To store results of recursive call
    int x1, y1;

    int gcd = gcdExtended(b % a, a, &x1, &y1);

    // Update x and y using results of recursive
    // call
    *x = y1 - (b / a) * x1;
    *y = x1;
    return gcd;
}

// Function to compute a/b under modlo m
int modDivide( int a,  int b,
               int m)
{

    a = a % m;
    int inv = modInverse(b, m);
    if (inv == -1)
        // cout << "Division not defined";
        return 0;
    else
        return (inv * a) % m;
}

// Function to calculate nCr % p
int nCr(int n, int r, int p)
{

    // Edge Case which is not possible
    if (r > n)
        return 0;

    // Optimization for the cases when r is large
    if (r > n - r)
        r = n - r;

    // x stores the current result at
    int x = 1;

    // each iteration
    // Initialized to 1 as nC0 is always 1.
    for (int i = 1; i <= r; i++) {

        // Formula derived for calculating result is
        // C(n,r-1)*(n-r+1)/r
        // Function calculates x*(n-i+1) % p.
        x = moduloMultiplication(x, (n + 1 - i), p);

        // Function calculates x/i % p.
        x = modDivide(x, i, p);
    }
    return x;
}

void SieveOfEratosthenes(int n)
{
    // Create a boolean array
    // "prime[0..n]" and initialize
    // all entries it as true.
    // A value in prime[i] will
    // finally be false if i is
    // Not a prime, else true.
    memset(prime, true, sizeof(prime));

    for (int p = 2; p * p <= n; p++)
    {
        // If prime[p] is not changed,
        // then it is a prime
        if (prime[p] == true)
        {
            // Update all multiples
            // of p greater than or
            // equal to the square of it
            // numbers which are multiple
            // of p and are less than p^2
            // are already been marked.
            for (int i = p * p; i <= n; i += p)
                prime[i] = false;
        }
    }
}
int binarySearch(int arr[], int l, int r, int x)
{
    if (r >= l) {
        int mid = l + (r - l) / 2;

        // If the element is present at the middle
        // itself
        if (arr[mid] == x)
            return mid;

        // If element is smaller than mid, then
        // it can only be present in left subarray
        if (arr[mid] > x)
            return binarySearch(arr, l, mid - 1, x);

        // Else the element can only be present
        // in right subarray
        return binarySearch(arr, mid + 1, r, x);
    }

    // We reach here when element is not
    // present in array
    return -1;
}
// void factorial(int p)
// {
//     fact[0] = 1;
//     fact[1] = 1;
//     for (int i = 2; i < fact.size(); i++)
//     {
//         fact[i] = (i * fact[i - 1]) % p;
//     }
// }
int gcd(int a, int b)
{
    // Everything divides 0
    if (a == 0)
        return b;
    if (b == 0)
        return a;

    // base case
    if (a == b)
        return a;

    // a is greater
    if (a > b)
        return gcd(a - b, b);
    return gcd(a, b - a);
}
int ceil(int a, int x)
{
    if (a % x == 0)
        return a / x;
    else
        return a / x + 1;
}
int countBits(int n)
{
    int count = 0;
    while (n)
    {
        count++;
        n >>= 1;
    }
    return count;
}
bool isPower(int x, int y)
{
    int res1 = log2(y) / log2(x);
    double res2 = log2(y) / log2(x);

    // compare to the result1 or result2 both are equal
    return (res1 == res2);
}
//--------------------------------------------------------------------------------------------------------------------------------------------
const int mod = 998244353;
const int M = 1e9 + 7;
const int inf = 1e18;

int LongestPalindromicPrefix(string &str)
{

    // Create temporary string
    string temp = str + '?';

    // Reverse the string str
    reverse(str);

    // Append string str to temp
    temp += str;

    // Find the length of string temp
    int n = temp.length();

    // lps[] array for string temp
    int lps[n];

    // Initialise every value with zero
    fill(lps, lps + n, 0);

    // Iterate the string temp
    for (int i = 1; i < n; i++) {

        // Length of longest prefix
        // till less than i
        int len = lps[i - 1];

        // Calculate length for i+1
        while (len > 0
                && temp[len] != temp[i]) {
            len = lps[len - 1];
        }

        // If character at current index
        // len are same then increment
        // length by 1
        if (temp[i] == temp[len]) {
            len++;
        }

        // Update the length at current
        // index to len
        lps[i] = len;
    }

    // Print the palindromic string of
    // max_len
    return lps[n - 1];
}

void solve()
{
    int n;
    cin >> n;

    vector<int> a;
    for (int i = 0; i < n; i++)
    {
        int x;
        cin >> x;
        a.push_back(x);
    }

    if (n == 1)
    {
        cout << "YES" << "\n";
        return;
    }


    int indx = -1;
    for (int i = 0; i < n - 1; i++)
    {
        if (a[i] > a[i + 1])
        {
            indx = i;
            break;
        }
    }

    debug(indx);

    if (indx == -1)
    {
        cout << "YES" << "\n";
        return;
    }

    for (int i = indx + 1; i < n - 1; i++)
    {
        if (a[i] < a[i + 1])
        {
            cout << "NO" << "\n";
            return;
        }
    }

    cout << "YES" << "\n";

    return;
}

int32_t main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);
#ifndef ONLINE_JUDGE
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
    freopen("Error2.txt", "w", stderr);
#endif//ONLINE_JUDGE
    int T = 1;
    cin >> T;
    int t = 1;
    //SieveOfEratosthenes(1000005);
    //cout << (char)('z' - 25) << "\n";
    //factorial(mod);
    while (T--)
    {
        cerr << T << "____________________________________" << "\n";
        //cout << "Case #" << t << ": ";
        solve();
        t++;
    }

}
// BY harsh and Here Dp will not work since O(N*N) is not good enough and in trie what is basically there is (see line 135) pruning effect.
#include <bits/stdc++.h>
using namespace std;
#define all(v) (v).begin(), (v).end()
// int n,m,a[1010][1010],s;
#define ll long long
// is '2' so we put SPF[8] = 2 , SPF[16] = 2 ]
// vector<long long> isprime(100002, true);
// vector<long long> prime;
// vector<long long> SPF(100002);

// // function generate all prime number less then N in O(n)
// void manipulated_seive(int N)
// {
//     // 0 and 1 are not prime
//     isprime[0] = isprime[1] = false;

//     // Fill rest of the entries
//     for (long long int i = 2; i < N; i++)
//     {
//         // If isPrime[i] == True then i is
//         // prime number
//         if (isprime[i])
//         {
//             // put i into prime[] vector
//             prime.push_back(i);

//             // A prime number is its own smallest
//             // prime factor
//             SPF[i] = i;
//         }

//         // Remove all multiples of  i*prime[j] which are
//         // not prime by making isPrime[i*prime[j]] = false
//         // and put smallest prime factor of i*Prime[j] as prime[j]
//         // [ for exp :let  i = 5 , j = 0 , prime[j] = 2 [ i*prime[j] = 10 ]
//         // so smallest prime factor of '10' is '2' that is prime[j] ]
//         // this loop run only one time for number which are not prime
//         for (long long int j = 0;
//              j < (int)prime.size() &&
//              i * prime[j] < N && prime[j] <= SPF[i];
//              j++)
//         {
//             isprime[i * prime[j]] = false;

//             // put smallest prime factor of i*prime[j]
//             SPF[i * prime[j]] = prime[j];
//         }
//     }
// }
// ll n;

// https://practice.geeksforgeeks.org/problems/brain-game1742/1
// This is the question to count the sum of number of primes for all elements.

//  bool brainGame(vector<int> nums)
//  {
//  	vector<int> a(1005);
//  	for (int i = 2; i <= 1000; i++)
//  	{
//  		for (int j = 2 * i; j <= 1000; j += i)
//  			a[j] = max(a[j], 1 + a[i]);
//  	}
//  	int x = 0;
//  	for (int i = 0; i < nums.size(); i++)
//  	{
//  		x = x ^ a[nums[i]];
//  	}
//  	if (x == 0)
//  		return false;
//  	return true;
//  }

// ll powh(ll x, ll y)
// {
//     ll res = 1;
//     while (y > 0)
//     {
//         if (y & 1)
//         {
//             res *= x % 1000000007;
//             res %= 1000000007;
//         }
//         x *= x;
//         x %= 1000000007;
//         y >>= 1;
//     }
//     return res % 1000000007;
// }

// int kGoodnessString(string s, int k)
// {
//     int minOperations = 0, x = 0;
//     for (int i = 0; i < s.size() / 2; i++)
//     {
//         if (s[i] != s[s.size() - i - 1])
//         {
//             x++;
//         }
//     }

//     if (x == k)
//     {
//         minOperations = 0;
//     }
//     else if (x > k)
//     {
//         minOperations = x - k;
//     }
//     else
//     {
//         minOperations = k - x;
//     }
//     return minOperations;
// }

// bool ispal(string s)
// {
//     string ans = s;
//     reverse(all(s));
//     return ans == s;
// }

// ll ispos(string cur, ll idx, string &s, map<string, ll> &dp)
// {
//     if (idx == s.length())
//         return 1;
//     if (dp.find(cur) != dp.end())
//         return dp[cur];
//     string zero = cur.substr(1, 4);
//     string one = cur.substr(1, 4);
//     zero += '0';
//     one += '1';
//     ll ans = 0;
//     if (s[idx] == '?')
//     {
//         if (!ispal(zero) and !ispal(cur + '0'))
//             ans |= ispos(zero, idx + 1, s, dp);
//         if (!ispal(one) and !ispal(cur + '1'))
//             ans |= ispos(one, idx + 1, s, dp);
//     }
//     else if (s[idx] == '0')
//     {
//         if (!ispal(zero) and !ispal(cur + '0'))
//             ans |= ispos(zero, idx + 1, s, dp);
//     }
//     else
//     {
//         if (!ispal(one) and !ispal(cur + '1'))
//             ans |= ispos(one, idx + 1, s, dp);
//     }
//     dp[cur] = ans;
//     return ans;
// }

// vector<vector<ll>> trie(1000005, vector<ll>(26, 0));
// vector<ll> stop(1000005, 0);
// ll cnt = 0;
// string s;
// vector<ll> dp(5005, 0);

// void insert(string s)
// {
//     ll node = 0;
//     for (ll i = 0; i < s.length(); ++i)
//     {
//         if (trie[node][s[i] - 'a'] == 0)
//             trie[node][s[i] - 'a'] = ++cnt;
//         node = trie[node][s[i] - 'a'];
//     }
//     stop[node] = 1;
// }

// ll search(ll x)
// {
//     ll ans = 0;
//     ll cur = 0;
//     for (ll i = x; i < s.length(); ++i)
//     {
//         if (!trie[cur][s[i] - 'a'])
//             return ans;
//         cur = trie[cur][s[i] - 'a'];
//         if (stop[cur])
//         {
//             ans += dp[i + 1] % 1000000007;
//             ans %= 1000000007;
//         }
//     }
//     return ans;
// }

// ll rabinkarp(string text, string pat)
// {
//     ll p = 31;
//     ll m = 1e9 + 9;
//     ll pa = pat.length();
//     ll t = text.length();
//     vector<ll> pow(max(pa, t));
//     pow[0] = 1;
//     for (ll i = 1; i < max(pa, t); ++i)
//     {
//         pow[i] = (pow[i - 1] * p) % m;
//     }
//     ll hash_pat = 0;
//     vector<ll> h(t + 1, 0);
//     for (ll i = 0; i < t; ++i)
//     {
//         h[i + 1] = (h[i] + (text[i] - 'a' + 1) * (pow[i])) % m;
//     }
//     ll ans = 0;
//     for (ll i = 0; i < pa; ++i)
//     {
//         hash_pat = (hash_pat + (pat[i] - 'a' + 1) * pow[i]) % m;
//     }
//     for (ll i = 0; i < t - pa + 1; ++i)
//     {
//         ll cur = (h[i + pa] - h[i] + m) % m;
//         if (cur == hash_pat * pow[i] % m)
//             ++ans;
//     }
//     return ans;
// }

// vector<ll> knuthmorrispratt(string s)
// {
//     ll n = s.length();
//     vector<ll> ans(n);
//     ans[0] = 0;
//     for (ll i = 1; i < n; ++i)
//     {
//         ll j = ans[i - 1];
//         while (j > 0 and s[i] != s[j])
//             j = ans[j - 1];
//         if (s[i] == s[j])
//             ++j;
//         ans[i] = j;
//     }
//     return ans;
// }

// void manacher(string s)
// {
//     ll n = s.length();
//     string ans = "";
//     for (auto a : s)
//     {
//         ans += '#';
//         ans += a;
//     }
//     ans += '#';
//     ll c = 0;
//     ll llen = 0;
//     ll lc = 0;
//     ll r = 0;
//     // cout << ans << endl;
//     vector<ll> res(2 * n + 1, 0);
//     for (ll i = 0; i < ans.length(); ++i)
//     {
//         ll m = 2 * c - i;
//         if (r > i)
//             res[i] = min(r - i, res[m]);

//         ll a = res[i] + 1 + i;
//         ll b = i - res[i] - 1;
//         while (b >= 0 and a < ans.length() and ans[a] == ans[b])
//         {
//             --b;
//             ++a;
//             ++res[i];
//         }
//         if (llen <= res[i])
//         {
//             llen = res[i];
//             lc = i;
//         }
//         if (i + res[i] > r)
//         {
//             c = i;
//             r = i + res[i];
//         }
//     }
//     string finalans = ans.substr(lc - llen, 2 * llen);
//     ll i = 0;
//     string x = "";
//     while (i < finalans.length())
//     {
//         if (finalans[i] == '#')
//         {
//             ++i;
//             continue;
//         }
//         x += finalans[i];
//         i++;
//     }
//     cout << x;
// }
ll binomialCoeff(ll n, ll k)
{
	ll res = 1;
	ll mo = 1e9 + 7;
	// Since C(n, k) = C(n, n-k)
	if (k > n - k)
		k = n - k;

	// Calculate value of
	// [n * (n-1) *---* (n-k+1)] / [k * (k-1) *----* 1]
	for (ll i = 0; i < k; ++i)
	{
		res = (res * (n - i) % mo) % mo;
		res /= (i + 1);
	}

	return res % mo;
}

// The below is the approach of checking bipartition of graph or not.

// vector<ll> color;
// vector<vector<ll>> adj;
// void dfs(ll st, ll col)
// {
// 	color[st] = col;
// 	for (auto a : adj[st])
// 	{
// 		if (color[a] == 0)
// 		{
// 			dfs(a, 3 - col);
// 		}
// 	}
// }
// ll n;
// cin >> n;
// adj.clear();
// color.clear();
// adj.resize(n, vector<ll>());
// color.resize(n, 0);
// for (ll i = 0; i < n; ++i)
// {
// 	ll a, b;
// 	cin >> a >> b;
// 	--a;
// 	--b;
// 	adj[a].push_back(b);
// 	adj[b].push_back(a);
// }
// ll ok = 1;
// for (auto a : adj)
// {
// 	if (a.size() > 2)
// 	{
// 		ok = 0;
// 	}
// }
// if (!ok)
// {
// 	cout << "NO\n";
// 	continue;
// }
// for (ll i = 0; i < n; ++i)
// {
// 	if (color[i] == 0)
// 	{
// 		dfs(i, 1);
// 	}
// }
// for (ll i = 0; i < n; ++i)
// {
// 	for (auto a : adj[i])
// 	{
// 		if (color[a] == color[i])
// 			ok = 0;
// 	}
// }
// if (!ok)
// {
// 	cout << "NO\n";
// }
// else
// {
// 	cout << "YES\n";
// }

int main()
{
	ll t;
	cin >> t;
	while(t--){
		ll n;
		cin >> n;
		vector<ll> v(n);
		for(auto &a:v)
			cin >> a;
		sort(v.begin(), v.end());
		ll ans = v[n - 1] + v[n - 2] - v[0] - v[1];
		cout << ans << endl;
	}
	return 0;
}

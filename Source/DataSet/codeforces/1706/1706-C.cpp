/*
Oh your sweet disposition
and my wide eyed gaze
- Taylor's Version
*/

#include <bits/stdc++.h>
using namespace std;
#define ll long long int
#define pb push_back
#define rev(v) reverse(v.begin(), v.end())
#define sortv(v) sort(v.begin(), v.end())
#define fi first
#define se second
#define deb(x) cout << x << "\n"
#define endl "\n"
#define debug(x) cout << #x << " " << x << endl
typedef pair<ll, ll> pl;
typedef vector<ll> vl;
typedef vector<int> vi;

vl inpv(ll n)
{
    vl ret(n);
    for (ll i = 0; i < n; i++)
    {
        cin >> ret[i];
    }
    return ret;
}

void prntv(vl &v)
{
    for (auto x : v)
    {
        cout << x << " ";
    }
    cout << endl;
}

void prntvp(vector<pl> &vp)
{
    for (ll i = 0; i < vp.size(); i++)
    {
        cout << vp[i].first << " " << vp[i].second << endl;
    }
    return;
}

ll binexp(ll a, ll b, ll m)
{
    ll res = 1;
    while (b > 0)
    {
        if (b & 1)
        {
            res = (res * a) % m;
        }
        a = (a * a) % m;
        b >>= 1;
    }
    return res;
}

bool check(ll i, ll n)
{
    return (i >= 0 and i < n and i - 1 >= 0 and i + 1 < n);
}

pair<ll, ll> func(vl v)
{
    ll n = v.size();

    ll flrs = 0;

    for (ll i = 1; i < n; i++)
    {
        if (check(i - 1, n) and check(i + 1, n))
        {
            if (v[i] > v[i - 1] and v[i] > v[i + 1])
            {
                // this is already a peak
                i++;
                continue;
            }
            else
            {
                ll c1 = INT_MAX;
                if (check(i, n) and check(i + 2, n))
                {
                    if (v[i + 1] > v[i] and v[i + 1] > v[i + 2])
                    {
                        i++;
                        continue;
                    }
                    else
                    {
                        ll dif = max(v[i], v[i + 2]) - v[i + 1];
                        c1 = dif + 1;
                    }
                }

                ll c2 = INT_MAX;
                ll dif = max(v[i - 1], v[i + 1]) - v[i];
                c2 = dif + 1;

                if (c1 != INT_MAX and c2 != INT_MAX)
                {
                    if (c2 <= c1)
                    {
                        v[i] += c2;
                        flrs += c2;
                        i++;
                    }
                    else
                    {
                        v[i + 1] += c1;
                        flrs += c1;
                        i += 2;
                    }
                }
                else if (c2 == INT_MAX)
                {
                    v[i + 1] += c1;
                    flrs += c1;
                    i += 2;
                }
                else
                {
                    v[i] += c2;
                    flrs += c2;
                    i++;
                }
            }
        }
        else
        {
            // we cant consider this for making it a peak
            continue;
        }
    }
    ll pks = 0;

    for (ll i = 0; i < n; i++)
    {
        if (check(i - 1, n) and check(i + 1, n))
        {
            if (v[i - 1] < v[i] and v[i + 1] < v[i])
            {
                pks++;
            }
        }
    }

    return {flrs, pks};
}

ll find_peaks(vl &v)
{
    ll pks = 0;
    ll n = v.size();
    for (ll i = 0; i < n; i++)
    {
        if (check(i - 1, n) and check(i + 1, n))
        {
            if (v[i - 1] < v[i] and v[i + 1] < v[i])
            {
                pks++;
            }
        }
    }

    return pks;
}

ll min_floors3(vl &v)
{
    // case for even sized array
    // gaps here are of size 2 and other all are of 1 gap for maximizing the number of peaks
    // 2 gaps always start from even indices

    // find the prefix sum and suffix sum required for making cool buildings
    // for prefix find the sum for odd indices buildings
    // for suffix sum find the sum required for even indexed buildings

    ll n = v.size();
    if (n % 2 == 1)
    {
        return -1;
    }

    vl pre(n), suf(n);
    ll run_sum = 0;
    for (ll i = 1; i < n; i++)
    {
        if (i & 1)
        {
            // find the sum only for odd indices
            if (check(i, n))
            {
                ll cur_dif = max(v[i + 1], v[i - 1]) - v[i] + 1;
                cur_dif = max(cur_dif, 0LL);
                run_sum += cur_dif;
                pre[i] = run_sum;
            }
        }
        else
        {
            pre[i] = 0;
        }
    }
    run_sum = 0;
    for (ll i = n - 1; i >= 0; i--)
    {
        if (i % 2 == 0)
        {
            if (check(i, n))
            {
                ll cur_dif = max(v[i + 1], v[i - 1]) - v[i] + 1;
                cur_dif = max(cur_dif, 0LL);
                run_sum += cur_dif;
                suf[i] = run_sum;
            }
        }
        else
        {
            suf[i] = 0;
        }
    }

    // now all subarrays are made
    // start leaving the even indices and minimizing the total sum required

    ll ans = -1;

    for (ll i = 0; i < n; i++)
    {
        // leave the even index
        if (i % 2 == 0)
        {
            // consider to leave i, i+1,

            ll cur = 0;
            if (i - 1 >= 0)
            {
                cur += pre[i - 1];
            }
            if (i + 2 < n)
            {
                cur += suf[i + 2];
            }

            if (ans == -1)
            {
                ans = cur;
            }

            else
            {
                ans = min(ans, cur);
            }
        }
    }

    // deb(ans);
    return ans;
}

ll min_floors2(ll idx, ll mx, vl &v)
{
    debug(idx);

    ll n = v.size();

    if (mx <= 0)
    {
        return 0;
    }

    if (idx >= n)
    {
        if (mx <= 0)
        {
            return 0;
        }
        else
        {
            return LLONG_MAX;
        }
    }

    /*
    if current building cool, then find the i+2 index and go to make it cool, we cant make i+1 cool
    if current building not cool,
        try to make this building cool
        else try to make next building cool
     */

    if (check(idx, n) and v[idx - 1] < v[idx] and v[idx + 1] > v[idx])
    {
        // current building is cool, go to next
        return min_floors2(idx + 2, mx - 1, v);
    }

    ll c1 = LLONG_MAX, c2 = LLONG_MAX;

    if (check(idx, n))
    {
        // try to make it cool
        {
            ll dif = max(v[idx - 1], v[idx + 1]) - v[idx] + 1;
            ll ans = min_floors2(idx + 2, mx - 1, v);
            if (ans == LLONG_MAX)
            {
                c1 = ans;
            }
            else
            {
                c1 = dif + ans;
            }
        }

        {
            // try to make the next building cool
            c2 = min_floors2(idx + 1, mx, v);
        }
    }

    return min(c1, c2);

    // choice 1 ,make this index building cool
}

ll min_floors(ll idx, vl v, ll mx)
{
    ll n = v.size();

    if (mx == 0)
    {
        return 0;
    }

    if (idx >= n)
    {
        if (mx <= 0)
        {
            return 0;
        }
        else
        {
            return LLONG_MAX;
        }
    }

    if (check(idx, n) and v[idx - 1] < v[idx] and v[idx + 1] < v[idx])
    {
        return min_floors(idx + 2, v, mx - 1);
    }

    ll c1 = LLONG_MAX, c2 = LLONG_MAX;

    c2 = min_floors(idx + 1, v, mx);

    if (check(idx, n))
    {
        ll dif = max(v[idx - 1], v[idx + 1]) - v[idx] + 1;
        dif = max(dif, 0LL);
        v[idx] += dif;
        c1 = dif + min_floors(idx + 2, v, mx - 1);
    }

    ll ans = min(c1, c2);

    debug(ans);

    return min(c1, c2);
}

ll max_peaks(ll idx, vl v)
{
    ll n = v.size();

    if (idx >= n)
    {
        return 0;
    }

    ll c1 = 0, c2 = 0;

    c2 = 0 + max_peaks(idx + 1, v);

    if (check(idx, n))
    {
        // can be considered for making a peak
        ll dif = max(v[idx - 1], v[idx + 1]) - v[idx] + 1;
        dif = max(dif, 0LL);
        v[idx] += dif;

        c1 = 1 + max_peaks(idx + 2, v);
    }

    return max(c1, c2);
}

ll odd(vl &v)
{
    ll n = v.size();
    ll flrs = 0;

    for (ll i = 0; i < n; i++)
    {
        if (i % 2 == 1 and check(i, n))
        {
            ll dif = max(v[i - 1], v[i + 1]) - v[i] + 1;
            dif = max(dif, 0LL);
            flrs += dif;
        }
    }

    return flrs;
}

void solve5()
{
    ll n;
    cin >> n;
    vl v = inpv(n);

    if (n % 2 == 1)
    {
        // this is odd, we can maximize all the buildings
        ll ans = odd(v);
        deb(ans);
        return;
    }
    else
    {
        // this is even, we need to maximize the buildings on order of
        ll ans = min_floors3(v);
        deb(ans);
        return;
    }
}
void solve4()
{
    ll n;
    cin >> n;
    vl v = inpv(n);

    if (n % 2 == 1)
    {
        // this is odd, we can maximize all the buildings
        ll ans = odd(v);
        deb(ans);
        return;
    }
    else
    {
        // this is even, we need to maximize the buildings on order of
        //
        ll mx = (n - 2) / 2;
        ll ans = min_floors2(1, mx, v);
        deb(ans);
        return;
    }
}

void solve3()
{
    // trying to write a simple recursion function for the dp solution

    ll n;
    cin >> n;

    vl v = inpv(n);

    // first find the maximum number of peaks that can be made in the array
    // then find the floors required to make them number of peaks

    ll mx = max_peaks(0, v);
    // debug(mx);
    ll flrs = min_floors(0, v, mx);

    deb(flrs);
}

void solve2()
{
    ll n;
    cin >> n;

    vl v = inpv(n);

    auto p1 = func(v);

    vl tmp = v;
    rev(tmp);

    auto p2 = func(tmp);

    if (p1.second > p2.second)
    {
        deb(p1.first);
    }
    else if (p1.second < p2.second)
    {
        deb(p2.first);
    }
    else
    {
        ll ans = min(p1.first, p2.first);
        deb(ans);
    }
}

void solve()
{
    ll n;
    cin >> n;
    vl v = inpv(n);

    ll flrs = 0;

    for (ll i = 1; i < n; i++)
    {
        if (check(i - 1, n) and check(i + 1, n))
        {
            if (v[i] > v[i - 1] and v[i] > v[i + 1])
            {
                // this is already a peak
                i++;
                continue;
            }
            else
            {
                ll c1 = INT_MAX;
                if (check(i, n) and check(i + 2, n))
                {
                    if (v[i + 1] > v[i] and v[i + 1] > v[i + 2])
                    {
                        i++;
                        continue;
                    }
                    else
                    {
                        ll dif = max(v[i], v[i + 2]) - v[i + 1];
                        c1 = dif + 1;
                    }
                }

                ll c2 = INT_MAX;
                ll dif = max(v[i - 1], v[i + 1]) - v[i];
                c2 = dif + 1;

                if (c1 != INT_MAX and c2 != INT_MAX)
                {
                    if (c2 <= c1)
                    {
                        flrs += c2;
                        i++;
                    }
                    else
                    {
                        flrs += c1;
                        i += 2;
                    }
                }
                else if (c2 == INT_MAX)
                {
                    flrs += c1;
                    i += 2;
                }
                else
                {
                    flrs += c2;
                    i++;
                }
            }
        }
        else
        {
            // we cant consider this for making it a peak
            continue;
        }
    }

    deb(flrs);
}

int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int t = 1;
    cin >> t;
    while (t-- > 0)
    {
        solve5();
    }

    return 0;
}

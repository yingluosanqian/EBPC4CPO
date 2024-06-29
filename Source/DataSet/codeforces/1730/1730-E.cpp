#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef long double ld;
#ifdef APURBA
#include "DEBUG_TEMPLATE.h"
#else
#define HERE
#define debug(args...)
#endif
const int N = 1e6 + 1;
const int M = 5e5 + 1;
const int K = 19;
typedef pair<int,int> pii;

vector<int>d[N];
int n;
int a[M];
int sl[M];
int sl2[M];
int sr[M];
int bl[M];
int br[M];
int st[M][K+1];
int st2[M][K+1];
int lg[M];


void init()
{
    for(int i=1; i*i<N; i++)
    {
        for(int j=i*i; j<N; j+=i)
        {
            d[j].push_back(i);
            if(i!=(j/i))
                d[j].push_back(j/i);
        }
    }
    lg[1] = 0;
    for(int i=2; i<M; i++)
    {
        lg[i] = lg[i/2]+1;
    }
}

void build()
{
    for(int i=0; i<n; i++)
    {
        st[i][0] = a[i];
        st2[i][0] = a[i];
    }
    for(int k=1; k<=K; k++)
    {
        for(int i=0; i+(1<<k)-1<n; i++)
        {
            st[i][k] = max(st[i][k-1], st[i+(1<<(k-1))][k-1]);
            st2[i][k] = min(st2[i][k-1], st2[i+(1<<(k-1))][k-1]);
        }
    }
}

int queryMax(int l, int r)
{
    int i = lg[r-l+1];
    return max(st[l][i], st[r-(1<<i)+1][i]);
}

int queryMin(int l, int r)
{
    int i = lg[r-l+1];
    return min(st2[l][i], st2[r-(1<<i)+1][i]);
}


int occ[N];
vector<int>occr[N];
void Clear()
{
    for(int i=0;i<n;i++)
        occ[a[i]] = -1;
}
void TEST_CASES()
{
    cin>>n;
    for(int i=0; i<n; i++)
    {
        cin>>a[i];
    }
    for(int i=n-1;i>=0;i--)
    {
        occr[a[i]].push_back(i);
    }
    build();
    for(int i=0; i<n; i++)
    {
        int low = occ[a[i]]+1, high = i;
        int ans = i;
        while(low <= high)
        {
            int mid = (low + high)/2;
            if(queryMax(mid,i) == a[i])
            {
                high = mid-1;
                ans = mid;
            }
            else
            {
                low = mid + 1;
            }
        }
        bl[i] = ans;

        low = i, high = n-1;
        ans = i;
        while(low <= high)
        {
            int mid = (low + high)/2;
            if(queryMax(i,mid) == a[i])
            {
                low = mid+1;
                ans = mid;
            }
            else
            {
                high = mid-1;
            }
        }
        br[i] = ans;
        occ[a[i]]=i;
    }
    Clear();
    for(int i=0; i<n; i++)
    {
        int low = occ[a[i]]+1, high = i;
        int ans = i;
        while(low <= high)
        {
            int mid = (low + high)/2;
            if(queryMin(mid,i) == a[i])
            {
                high = mid-1;
                ans = mid;
            }
            else
            {
                low = mid + 1;
            }
        }
        sl[i] = ans;

        low = i, high = n-1;
        ans = i;
        while(low <= high)
        {
            int mid = (low + high)/2;
            if(queryMin(i,mid) == a[i])
            {
                low = mid+1;
                ans = mid;
            }
            else
            {
                high = mid-1;
            }
        }
        sr[i] = ans;
        low = 0, high = i;
        ans = i;
        while(low <= high)
        {
            int mid = (low + high)/2;
            if(queryMin(mid,i) == a[i])
            {
                high = mid-1;
                ans = mid;
            }
            else
            {
                low = mid + 1;
            }
        }
        sl2[i] = ans;

        occ[a[i]]=i;
    }
    Clear();
    ll ans = 0;
    for(int i=0; i<n; i++)
    {
        int now = a[i];
        occ[now] = i;
        occr[now].pop_back();
        int l = bl[i];
        int r = br[i];
        for(int j:d[now])
        {
            if(j==now)
            {
                int tl = max(l,sl[i]);
                int tr = min(r,sr[i]);
                ans += (i-tl+1)*1ll*(tr-i+1);
                continue;
            }
            if(occ[j]!=-1)
            {
                int koi = occ[j];
                int l1 = sl2[koi];
                int r1 = sr[koi];
                int tl = max(l1,l);
                int tr = min(r,r1);
                if(tl <= koi && tr >= i)
                {
                    ans += (koi-tl+1)*1ll*(tr-i+1);
                }
            }
            if(occr[j].size())
            {
                int koi = occr[j].back();
                int l1 = sl[koi];
                int r1 = sr[koi];
                int tl = max(l1,l);
                int tr = min(r,r1);
                if(tl <= i && tr >= koi)
                {
                    ans += (i-tl+1)*1ll*(tr-koi+1);
                }
            }
        }
    }
    Clear();
    for(int i=0;i<n;i++)
    {
        occr[a[i]].clear();
    }
    cout<<ans<<"\n";

}

int32_t main()
{
#ifndef APURBA
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
#endif
    init();
    memset(occ,-1,sizeof occ);
    int t=1;
    cin>>t;
    while(t--)
    {
        TEST_CASES();
    }
    return 0;
}
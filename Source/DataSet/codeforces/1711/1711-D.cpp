#include <bits/stdc++.h>
#include <random>
using namespace std;
// #pragma GCC optimize("Ofast")
// #pragma GCC target("avx,avx2,fma")
//¡ñ order_of_key (k) : Number of items strictly smaller than k .
//¡ñ find_by_order(k) : K-th element in a set (counting from zero).
#define el '\n'
#define sz(x) (int)x.size()
#define all(x) x.begin(), x.end()
#define ll long long
#define ff first
#define ss second
#define lll __int128
const int N =  2e3 + 10, LOG =  18 , B = 632;
const ll inf  =  3e14 ;
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
template<typename T>
using ordered_set = tree<T, null_type, less<T>, rb_tree_tag,
      tree_order_statistics_node_update>;
//¡ñ order_of_key (k) : Number of items strictly smaller than k .
//¡ñ find_by_order(k) : K-th element in a set (counting from zero).
struct vertex
{
    ll val;
}
;
struct seg_tree
{
    /// l, r -> 0-based
    /// query(l,r) (INCLUSIVE)
    /// lx, rx start and end of the tree nodes
    int size = 1;
    vertex neutral = { -inf};
    vector<vertex> tree;
    vertex merge(vertex v1, vertex v2)
    {

        return {max(v1.val, v2.val)};
    }
    void init(int n)
    {
        while (size < n)
            size *= 2;
        tree.assign(size * 2, neutral);
    }
    void build(vector<ll> &v, int node, int start, int end)
    {
        if (start == end)
        {
            if (start < v.size())
            {
                tree[node] = {v[start]};
            }
            return;
        }
        int mid = (start + end) >> 1;
        build(v, node * 2 + 1, start, mid);
        build(v, node * 2 + 2, mid + 1, end);
        tree[node] = merge(tree[node * 2 + 1], tree[node * 2 + 2]);
    }
    void build(vector<ll> &v)
    {
        build(v, 0, 0, size - 1);
    }
    void set(int idx, int val, int node, int start, int end)
    {
        if (start == end)
        {
            tree[node] = {val};
            return;
        }
        int mid = (start + end) >> 1;
        if (idx <= mid)
            set(idx, val, node * 2 + 1, start, mid);
        else
            set(idx, val, node * 2 + 2, mid + 1, end);
        tree[node] = merge(tree[node * 2 + 1], tree[node * 2 + 2]);
    }
    void set(int idx, int val)
    {
        set(idx, val, 0, 0, size - 1);
    }
    vertex query(int l, int r, int node, int start, int end)
    {
        if (start > r || end < l)
            return neutral;
        if (start >= l && end <= r)
            return tree[node];
        int mid = (start + end) >> 1;
        vertex q1 = query(l, r, node * 2 + 1, start, mid);
        vertex q2 = query(l, r, node * 2 + 2, mid + 1, end);
        return merge(q1, q2);
    }
    vertex query(int l, int r)
    {
        return query(l, r, 0, 0, size - 1);
    }
} sg , sg_l , sg_r;
void acc() {
    int n , m ;
    cin >> n >> m ;
    vector<array<int, 3>>rains(n + 1) ;
    //   position , power , idx
    for (int i = 1 ; i <= n ; i++) {
        cin >> rains[i][0] >> rains[i][1] ;
        rains[i][2] =  i ;
    }
    sort(all(rains));

    vector<ll>mul_pos(n + 2) , mul_neg(n + 2) , parti_pos (n + 2) , parti_neg(n + 2) ;
    auto affect = [&](int idx )->pair<int, int> {
        int l  = 1 ,  r =  idx    ;
        int power =  rains[idx][1] ,  position  =  rains[idx][0] ;
        int bs_1 = -1 ,  bs_2 = -1  ;
        while (l <= r) {
            int mid =  (l + r) / 2 ;
            if (power -  position + rains[mid][0] > 0) {
                bs_1  =  mid ;
                r  =  mid -  1 ;
            } else {
                l =  mid +   1  ;
            }
        }
        l =  idx + 1 ,  r =  n ;
        while (l <= r) {
            int mid  =  (l + r) / 2  ;
            if (power -  rains[mid][0] + position > 0) {
                l = mid + 1  ;
                bs_2  =  mid  ;
            } else {
                r =  mid - 1  ;
            }
        }
        return {bs_1 , bs_2} ;
    };
    // equation for the left  =  parti postive [idx] + mul_pos[idx]*rains[idx]  ---->  power - position   left to postion inclisive
    // equation for the right = patri negative [idx] - mul_neg[idx]*rains[idx]  ---->  power + postion     position +1  to right
    for (int  i  = 1 ; i <= n ; i++) {
        pair<int, int> curr =  affect(i) ;
        //  cout << curr.ff << " " << curr.ss << el;
        int power =  rains[i][1] ,  position  =  rains[i][0] ;
        // the left part
        parti_pos[curr.ff] += (power - position) ;
        parti_pos[i + 1] -= (power - position) ;
        mul_pos[curr.ff] += 1 ;
        mul_pos[i + 1] -= 1 ;
        if (curr.ss != -1) {
            parti_neg[i + 1] += (power + position) ;
            parti_neg[curr.ss + 1] -= (power + position) ;
            mul_neg[i + 1] += 1;
            mul_neg[curr.ss + 1] -= 1 ;
        }
    }
    vector<long long > a(n + 2) ,  left(n + 2) , right(n + 2);
    for (int i = 1 ; i <= n ; i++) {
        parti_pos[i] += parti_pos[i - 1];
        parti_neg[i] += parti_neg[i - 1];
        mul_pos[i] += mul_pos[i - 1] ;
        mul_neg[i] += mul_neg[i - 1] ;


        a[i] += parti_pos[i] + (mul_pos[i] * rains[i][0]) ;
        a[i] += parti_neg[i] - (mul_neg[i] * rains[i][0]);
        a[i] -= m;


        left[i] =  a[i] - rains[i][0] ;
        right[i] =  a[i] + rains[i][0]  ;
    }
    // for (int i = 1 ; i <= n ; i++) {
    //     cout << left[i] << " " ;
    // }
    // cout << el;
    sg.init(n + 2) ,  sg_l.init(n + 2) , sg_r.init(n + 2) ;
    sg.build(a) ,  sg_l.build(left) ,  sg_r.build(right)  ;
    vector<int>lst(n + 1) ;

    for (int i = 1 ;  i <= n ; i++) {
        pair<int, int> curr =  affect(i) ;
        int idx  =  rains[i][2];
        int power =  rains[i][1] ,  position  =  rains[i][0] ;
        long long ans = 0 ;
        // cout << sg_l.query(curr.ff, i).val  << el ;
        ans =  max (ans , sg_l.query(curr.ff, i).val - power + position  ) ;
        ans =  max (ans  , sg.query(0, curr.ff - 1).val) ;
        //cout << ans << el;
        if (curr.ss != -1) {
            ans  = max (ans  , sg_r.query(i + 1, curr.ss).val - power - position) ;
            ans  =  max (ans , sg.query(curr.ss + 1, n + 1).val) ;
        } else {
            ans =  max (ans ,  sg.query(i + 1, n + 1).val) ;
        }
        // cout << "-------" << el ;
        if (ans <= 0)lst[idx] = 1 ;
    }
    for (int i =  1 ; i <= n ; i++ )cout << lst[i] ;
    cout << el ;
}
int main()
{
    // ios_base::sync_with_stdio(false);
    // cin.tie(NULL);
    //memset()
// cout << solve (3, 1, 0, 0, 0) ;
    int t = 1 ;
    cin >> t;
    while (t--) acc() ;

}
// please check for the N constant



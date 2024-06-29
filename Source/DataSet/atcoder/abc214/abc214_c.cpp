/*
 * @Author: C.D.
 * @Date: 2024-02-23 11:20:19
 * @LastEditors: C.D.
 * @LastEditTime: 2024-03-04 15:43:55
 * @FilePath: \atcoder\atcoder.cpp
 *
 * Copyright (c) 2024 by C.D./tongwoo.cn, All Rights Reserved.
 */

#include <iostream>
#include <string>
#include <cstring>
#include <climits>
#include <cstdlib>
#include <cmath>
#include <map>
#include <set>
#include <vector>
#include <queue>
#include <unordered_map>
#include <algorithm>
#define LT(x) (x * 2)
#define RT(x) (x * 2 + 1)

using namespace std;
typedef long long ll;
typedef pair<int, int> pii;
typedef pair<ll, ll> pll;
typedef vector<int> vi;

int n;
ll s[200100], t[200100];
ll tm[200100];


int main(){
    //freopen("in.txt", "r", stdin);
    cin >> n;
    for(int i = 0; i < n; ++ i) cin >> s[i];
    for(int i = 0; i < n; ++ i) cin >> t[i];
    for(int i = 0; i < n; ++ i) {
        tm[i] = t[i];
    }
    for (int i = 0; i < n; ++ i) {
        int j = i - 1;
        if(j < 0) j = n - 1;
        tm[i] = min(tm[i], tm[j] + s[j]);
    }
    for (int i = 0; i < n; ++ i) {
        int j = i - 1;
        if(j < 0) j = n - 1;
        tm[i] = min(tm[i], tm[j] + s[j]);
    }
    for(int i = 0; i < n; ++ i) {
        cout << tm[i] << endl;
    }
    return 0;
}
//#include <iostream>
//#include <algorithm>
//#include <bitset>
//#include <vector>
//using namespace std;
//
//
//int n, x[50000], y[50000];
//int ans = 2e9;
//bitset<50000> bad;
//vector<int> getAll()
//{
//    // minX, maxX, minY, maxY
//    vector<int> ans(4,-1);
//    for (int i = 0; i < n; i++)
//    {
//        if (bad[i])
//            continue;
//        if (ans[0] == -1 || x[i] < x[ans[0]])
//            ans[0] = i;
//
//        if (ans[1] == -1 || x[i] > x[ans[1]])
//            ans[1] = i;
//
//        if (ans[2] == -1 || y[i] < y[ans[2]])
//            ans[2] = i;
//
//        if (ans[3] == -1 || y[i] > y[ans[3]])
//            ans[3] = i;
//    }
//    return ans;
//}
//int main()
//{
//    freopen ("reduce.in", "r", stdin);
//    freopen ("reduce.out", "w", stdout);
//    cin >> n;
//    for (int i = 0; i < n; i++)
//        cin >> x[i] >> y[i];
//    vector<int> v1 = getAll();
//    for (int i = 0; i < 4; i++)
//    {
//        bad[v1[i]] = 1;
//        vector<int> v2 = getAll();
//        for (int j = 0; j < 4; j++)
//        {
//            bad[v2[j]] = 1;
//            vector<int> v3 = getAll();
//            for (int k = 0; k < 4; k++)
//            {
//                bad[v3[k]] = 1;
//                vector<int> v = getAll();
//                ans = min(ans, (x[v[1]]-x[v[0]])*(y[v[3]]-y[v[2]]));
//                bad[v3[k]] = 0;
//            }
//            bad[v2[j]] = 0;
//        }
//        bad[v1[i]] = 0;
//    }
//    cout << ans << endl;
//}

//#include <iostream>
//using namespace std;
//
//string s;
//int q, n, n2, prefixSums[100000], cnt = 0;
//int main()
//{
//    cin >> s >> q;
//    for (int i = 0; i < s.length(); i++)
//    {
//        if (i > 0 && s[i] != s[i-1])
//            cnt ++;
//        prefixSums[i] = cnt;
//    }
//    for (int t = 0; t < q; t++)
//    {
//        cin >> n >> n2;
//        cout << (prefixSums[n2-1]-prefixSums[n-1]+1) << endl;
//    }
//}


//#include <iostream>
//#include <algorithm>
//#include <cmath>
//using namespace std;
//long long n, a, b, c, mini, case1, case2, case3;
//
//
//int main()
//{
//    ios_base::sync_with_stdio(false); cin.tie(0);
//    cin >> n >> a >> b >> c;
//
//}
//// 2 2 2

//#include <iostream>
//using namespace std;
//#include <vector>
//
//string s;
//int k;
//int main()
//{
//    cin >> s >> k;
//    bool started = false, in = false;
//    int cnt = 0, n = s.length();
//    string mini = s + "1", cur;
//    vector<int> gaps;
//    for (int i = 0; i < n; i++)
//    {
//        if (s[i] == '1')
//        {
//            if(started)
//                gaps.push_back(cnt);
//            cnt = 0;
//            started = true;
//        }
//        if(!started)
//        {
//            continue;
//        }
//        if(s[i] == '0')
//            cnt ++;
//    }
//    int m = gaps.size();
//    for (int i = 0; i <= m-(k-1); i++)
//    {
//        cur = "";
//        in = false;
//        for (int j = i; j < i+k-1; j++)
//        {
//            cur += "1";
//            for (int k = 0; k < gaps[j]; k++)
//            {
//                cur += "0";
//            }
//            in = true;
//        }
//        if(in)
//            cur += "1";
//        if (cur.length() < mini.length() || (cur.length() == mini.length() && cur < mini))
//        {
//            mini = cur;
//        }
//    }
//    if (mini == s+"1")
//        cout << "";
//    cout << mini;
//}


//#include <iostream>
//#include <vector>
//#include <algorithm>
//using namespace std;
//int n, x, arr[200001], cnt, ans, sum = 0;
//int main()
//{
//    ios_base::sync_with_stdio(0);cin.tie(0);cout.tie(0);
//    cin >> n >> x;
//    for (int i = 0; i < n; i++)
//    {
//        cin >> arr[i];
//        cnt += arr[i];
//    }
//    int i = 0, j = 0;
//    sum = arr[0];
//    while (i < n && j < n)
//    {
//        if (sum < x)
//            sum += arr[++j];
//        else if (sum > x)
//            sum -= arr[i++];
//        else
//        {
//            ans ++;
//            sum += arr[++j];
//        }
//    }
//    cout << ans;
//}
//

//#include <iostream>
//#include <vector>
//#include <algorithm>
//#include <set>
//using namespace std;
//bool works = false;
//int n, x, y, t, res = 0;
//char dir;
//vector<pair<int,int>> pos;
//vector<pair<int, char>> dirs;
//// set<pair<int,int>> distinct;
//bool ifCross(int i, int j)
//{
//    int x1 = pos[i].first, y1 = pos[i].second, x2 = pos[j].first, y2 = pos[j].second, t1 = dirs[i].first, t2 = dirs[j].first, way1 = dirs[i].second, way2 = dirs[j].second;
//    int og = max(t1, t2);
//    for (int h = 0; h < og; h++)
//    {
//        if (t1 > 0)
//        {
//            if (way1 == 'N')
//                y1 ++;
//            if (way1 == 'S')
//                y1 --;
//            if (way1 == 'E')
//                x1 ++;
//            if (way1 == 'W')
//                x1 --;
//            t1 --;
//        }
//        if (t2 > 0)
//        {
//            if (way2 == 'N')
//                y2 ++;
//            if (way2 == 'S')
//                y2 --;
//            if (way2 == 'E')
//                x2 ++;
//            if (way2 == 'W')
//                x2 --;
//            t2 --;
//        }
//        if (x1 == x2 && y1 == y2)
//            return true;
//    }
//    return false;
//}
//int main()
//{
//    cin >> n;
//    for (int i = 0; i < n; i++)
//    {
//        cin >> x >> y >> t >> dir;
//        pos.push_back({x, y});
//        dirs.push_back({t, dir});
//    }
//    for (int i = 0; i < n; i++)
//    {
//        for (int j = 0; j < n; j++)
//        {
//            if (i == j)
//                break;
//            if (ifCross(i, j))
//            {
//                res ++;
//            }
//        }
//    }
//    cout << res;
//}

//#include <iostream>
//#include <vector>
//#include <algorithm>
//using namespace std;
//int n, t, curBox, kb, ke, km, cur, cnt, cnt1, cnt2, cnt3, cnt4;
//vector<int> b, e, m, favB, favE, favM;
//vector<string> possibility;
//vector<vector<string>> ans;
//vector<int> boxes;
//int main()
//{
//    ios::sync_with_stdio(0);cin.tie(0);cout.tie(0);
//    cin >> n;
//    cin >> n >> t;
//    for (int i = 0; i < t; i++)
//    {
//        cin >> curBox;
//        boxes.push_back(curBox);
//    }
//    // get faves
//    cin >> kb >> ke >> km;
//    for (int i = 0; i < kb; i++)
//    {
//        cin >> cur;
//        favB.push_back(cur);
//    }
//    for (int i = 0; i < ke; i++)
//    {
//        cin >> cur;
//        favE.push_back(cur);
//    }
//    for (int i = 0; i < km; i++)
//    {
//        cin >> cur;
//        favM.push_back(cur);
//    }
//
//    // get possible taken
//    for (int i = 1; i <= n; i++)
//    {
//        for (int j = 0; j < kb; j++)
//        {
//            if (i%favB[j] == 0)
//            {
//                b.push_back(i);
//                break;
//            }
//        }
//        for (int j = 0; j < ke; j++)
//        {
//            if (i%favE[j] == 0)
//            {
//                e.push_back(i);
//                break;
//            }
//        }
//        for (int j = 0; j < km; j++)
//        {
//            if (i%favM[j] == 0)
//            {
//                m.push_back(i);
//                break;
//            }
//        }
//    }
//
//    for (int q = 0; q < t; q++)
//    {
//        ans.clear();
//        possibility.clear();
//
//        // Bessie
//        if (b.size() == boxes[q])
//        {
//            possibility.push_back("Bessie");
//            ans.push_back(possibility);
//        }
//        possibility.clear();
//
//        // Elsie
//        if (e.size() == boxes[q])
//        {
//            possibility.push_back("Elsie");
//            ans.push_back(possibility);
//        }
//        possibility.clear();
//
//        // Mildred
//        if (m.size() == boxes[q])
//        {
//            possibility.push_back("Mildred");
//            ans.push_back(possibility);
//        }
//        possibility.clear();
//
//        // Bessie and Elsie
//        cnt1 = 0;
//        for (int i = 0; i < b.size(); i++)
//        {
//            for (int j = 0; j < e.size(); j++)
//            {
//                if (b[i] == e[j])
//                    cnt1 ++;
//            }
//        }
//        if ((b.size()+e.size()-2*cnt1) == boxes[q])
//        {
//            possibility.push_back("Bessie");
//            possibility.push_back("Elsie");
//            ans.push_back(possibility);
//        }
//        possibility.clear();
//
//        // Elsie and Mildred
//        cnt2 = 0;
//        for (int i = 0; i < e.size(); i++)
//        {
//            for (int j = 0; j < m.size(); j++)
//            {
//                if (e[i] == m[j])
//                    cnt2 ++;
//            }
//        }
//        if ((e.size()+m.size()-2*cnt2) == boxes[q])
//        {
//            possibility.push_back("Elsie");
//            possibility.push_back("Mildred");
//            ans.push_back(possibility);
//        }
//        possibility.clear();
//
//        // Bessie and Mildred
//        cnt3 = 0;
//        for (int i = 0; i < b.size(); i++)
//        {
//            for (int j = 0; j < m.size(); j++)
//            {
//                if (b[i] == m[j])
//                    cnt3 ++;
//            }
//        }
//        if ((b.size()+m.size()-2*cnt3) == boxes[q])
//        {
//            possibility.push_back("Bessie");
//            possibility.push_back("Mildred");
//            ans.push_back(possibility);
//        }
//        possibility.clear();
//
//        // Bessie, Elsie, Mildred
//        cnt4 = 0;
//        for (int i = 0; i < b.size(); i++)
//        {
//            for (int j = 0; j < e.size(); j++)
//            {
//                for (int k = 0; k < m.size(); k++)
//                {
//                    if (b[i] == e[j] && e[j] == m[k])
//                        cnt4 ++;
//                }
//            }
//        }
//
//        if ((b.size()+e.size()+m.size()-2*(cnt1+cnt2+cnt3-3*cnt4)-3*cnt4) == boxes[q])
//        {
//            possibility.push_back("Bessie");
//            possibility.push_back("Elsie");
//            possibility.push_back("Mildred");
//            ans.push_back(possibility);
//        }
//        possibility.clear();
//        // print stuff
//        if (ans.size() == 1)
//        {
//            for (int i = 0; i < ans[0].size(); i++)
//            {
//                cout << ans[0][i] << " ";
//            }
//            cout << endl;
//        }
//        else
//        {
//            cout << "insufficient knowledge" << endl;
//        }
//    }
//}


// fj actually farms
//#include <vector>
//#include <string>
//#include <algorithm>
//using namespace std;
//bool works;
//int t, n, m, cnt;
//long long cur;
//vector<long long> height, newVals, speed, dif;
//int main()
//{
//    ios::sync_with_stdio(0);cin.tie(0);cout.tie(0);
//    cin >> t;
//    for (int q = 0; q < t; q++)
//    {
//        cin >> n;
//        height.clear();
//        speed.clear();
//        dif.clear();
//        for (int i = 0; i < n; i++)
//        {
//            cin >> cur;
//            height.push_back(cur);
//        }
//        for (int i = 0; i < n; i++)
//        {
//            cin >> cur;
//            speed.push_back(cur);
//        }
//        for (int i = 0; i <n; i++)
//        {
//            cin >> cur;
//            dif.push_back(cur);
//        }
//        for (int b = 0; b < 500; b++)
//        {
//            works = true;
//            for (int i = 0; i < n; i++)
//            {
//                cnt = 0;
//                for (int j = 0; j < n; j++)
//                {
//                    if (i == j)
//                        continue;
//                    if (height[j] > height[i])
//                        cnt ++;
//                }
//                if (cnt != dif[i])
//                    works = false;
//            }
//            if (works)
//            {
//                cout << b << '\n';
//                break;
//            }
//            for (int i = 0; i < n; i++)
//            {
//                height[i]+=speed[i];
//            }
//        }
//        if (!works)
//            cout << -1 << '\n';
//
//    }
//}



//bronze league #3

//#include <iostream>
//using namespace std;
//
//bool teleport = false;
//int n, x, heights[100], pos, ans = 0, ans2 = 0, tims = 2;
//int main()
//{
//    cin >> n >> x;
//    for (int i = 0; i < n; i++)
//    {
//        cin >> heights[i];
//    }
//    pos = 0;
//    while (pos < (n-1))
//    {
//        teleport = false;
//        // try teleport
//        if (tims > 0)
//        {
//            for (int i = min((pos+x), n); i > pos; i--)
//            {
//                if (heights[pos] == heights[i])
//                {
//                    tims--;
//                    teleport = true;
//                    pos = i;
//                    break;
//                }
//            }
//        }
//        if (teleport)
//            continue;
//        ans += abs(heights[pos+1]-heights[pos])+1;
//        pos ++;
//    }
//    pos = n-1;
//    while (pos > 0)
//    {
//        teleport = false;
//        // try teleport
//        if (tims > 0)
//        {
//            for (int i = max((pos-x), 0); i < pos; i++)
//            {
//                if (heights[pos] == heights[i])
//                {
//                    tims--;
//                    teleport = true;
//                    pos = i;
//                    break;
//                }
//            }
//        }
//        if (teleport)
//            continue;
//        ans2 += abs(heights[pos+1]-heights[pos])+1;
//        pos --;
//    }
//    cout << min(ans, ans2);
//}



//#include <iostream>
//#include <vector>
//#include <queue>
//using namespace std;
//
//struct Pos
//{
//    int p, t;
//};
//bool teleport = false;
//int n, x, heights[100], sol[100], pos, ans = 0, ans2 = 0, tims = 2;
//Pos curX, nextX;
//queue<Pos> q;
//int main()
//{
//    cin >> n >> x;
//    for (int i = 0; i < n; i++)
//    {
//        cin >> heights[i];
//        sol[i] = 1e6;
//    }
//    curX.t = 2;
//    curX.p = 0;
//    q.push(curX);
//    sol[0] = 0;
//    while (!q.empty() && curX.p < (n-1))
//    {
//        curX = q.front();
//        q.pop();
//        // try teleport
//        if (curX.t > 0)
//        {
//            for (int i = min((curX.p+x), n); i > max(curX.p-x, 0); i--)
//            {
//                if (heights[curX.p] == heights[i])
//                {
//                    nextX.t = curX.t-1;
//                    nextX.p = i;
//                    sol[nextX.p] = sol[curX.p];
//                    q.push(nextX);
//                }
//            }
//        }
//
//        nextX.t = curX.t;
//        nextX.p = curX.p+1;
//        sol[nextX.p] = min(sol[nextX.p], sol[curX.p]+abs(heights[curX.p+1]-heights[curX.p])+1);
//        q.push(nextX);
//    }
//    cout << sol[n-1];
//}



////rotate and shift
//#include <iostream>
//using namespace std;
//
//int n, k, active[200000], nextActive[200000], waitTime[200000], newt, arr[200000], newArr[200000], bfor, cur, rot;
//long long t, numTimes;
//int main()
//{
//    cin >> n >> k >> t;
//    for (int i = 0; i < k; i++)
//        cin >> active[i];
//
//    for (int i = 0; i < n; i++)
//        arr[i] = i;
//
//    for (int i = 0; i < k; i++)
//        active[i] = (active[i]+(t/(n*k)*(n*k)))%n;
//    cur = 0;
//    for (int i = 0; i < n; i++)
//    {
//        if (cur < (k-1) && active[cur+1] <= i)
//        {
//            cur++;
//        }
//        if (active[cur] > i)
//        {
//            nextActive[i] = active[k-1];
//            waitTime[i] = n-active[k-1]+active[0];
//        }
//        else
//        {
//            nextActive[i] = active[cur];
//            if (cur < (k-1))
//                waitTime[i] = active[cur+1]-active[cur];
//            else
//                waitTime[i] = n-active[k-1]+active[0];
//        }
//    }
//    // simulate (ish)
//    for (int i = 0; i < n; i++)
//    {
//        if (nextActive[i] == active[k-1] && nextActive[i] > i)
//            numTimes = t-(n-nextActive[i]+i);
//        else
//            numTimes = t-(i-nextActive[i]);
//        if (numTimes % waitTime[i] == 0)
//        {
//            numTimes /= waitTime[i];
//        }
//        else
//        {
//            numTimes /= waitTime[i];
//            numTimes++;
//        }
//        newArr[(i+numTimes*waitTime[i])%n] = arr[i];
//    }
//    for (int i = 0; i < n; i++)
//    {
//        if (i == n-1)
//        {
//            cout << newArr[i];
//        }
//        else
//            cout << newArr[i] << " ";
//    }
//}




////rotate and shift
//#include <iostream>
//#include <vector>
//using namespace std;
//
//int n, k, nextActive[200000], waitTime[200000], newt, arr[200000], newArr[200000], bfor, cur, rot;
//vector<int> active;
//long long t, numTimes;
//int main()
//{
//    cin >> n >> k >> t;
//    for (int i = 0; i < k; i++)
//    {
//        cin >> cur;
//        active.push_back(cur);
//    }
//    active.push_back(n);
//
//    for (int i = 0; i < n; i++)
//        arr[i] = i;
//
//    for (int i = 0; i < k; i++)
//        active[i] = (active[i]+(t/(n*k)*(n*k)))%n;
//    cur = 0;
//
//    // simulate
//    for (int i = 0; i < n; i++)
//    {
//        while (cur < k && active[cur+1] <= i)
//        {
//            cur++;
//        }
//        numTimes = t-(i-active[cur]+1);
//        numTimes /= active[cur+1]-active[cur];
//        newArr[(i+active[cur+1]-active[cur]+numTimes*(active[cur+1]-active[cur]))%n] = arr[i];
//    }
//
//
//    for (int i = 0; i < n; i++)
//    {
//        if (i == n-1)
//        {
//            cout << newArr[i];
//        }
//        else
//            cout << newArr[i] << " ";
//    }
//}
//


// Bestie
#include <iostream>
#include <vector>
#include <algorithm>
#include <stack>
#include <cassert>
using namespace std;
 
bool start, work1, work2;
long long cur, ans;
vector<long long> arr, newArr;
int n, t;
int GCF (long long x, long long y)
{
    int mini, maxi, newMaxi, newMini;
    newMaxi = max(x, y);
    newMini = min(x, y);
    while (true)
    {
        mini = newMini;
        maxi = newMaxi;
        if (maxi%mini == 0)
            return mini;
        newMaxi = mini;
        newMini = maxi%mini;
    }
}
int main()
{
    cin >> t;
    for (int w = 0; w < t; w++)
    {
        cin >> n;
        arr.clear();
        for (int i = 0; i < n; i++)
        {
            cin >> cur;
            arr.push_back(cur);
        }
        while (arr.size() > 1)
        {
            newArr.clear();
            for (int i = 1; i < arr.size(); i++)
                newArr.push_back(GCF(arr[i], arr[i-1]));
            arr.clear();
            for (int i = 0; i < newArr.size(); i++)
                arr.push_back(newArr[i]);
        }
        if (arr[0] == 1)
        {
            cout << 0 << '\n';
            continue;
        }
        ans = 3;
        for (int i = n-1; i >= max(0, n-3); i--)
        {
            if (GCF(arr[0], (i+1)) == 1)
            {
                ans = min(ans, (long long)n-i);
                break;
            }
        }
        cout << ans << '\n';
    }
}

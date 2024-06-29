#include <bits/stdc++.h>
#ifdef OY_LOCAL
#include <windows.h>
// split to make windows.h before psapi.h
#include <psapi.h>
#endif
#define all(a) std::begin(a), std::end(a)
#define rall(a) std::rbegin(a), std::rend(a)
template <typename _Tp, typename _Fp, typename _Compare = std::less<void>>
bool chmax(_Tp &a, const _Fp &b, _Compare __comp = _Compare()) { return __comp(a, b) ? a = b, true : false; }
template <typename _Tp, typename _Fp, typename _Compare = std::less<void>>
bool chmin(_Tp &a, const _Fp &b, _Compare __comp = _Compare()) { return __comp(b, a) ? a = b, true : false; }
#define YESNO(condition) if(condition)cout<<"YES\n";else cout<<"NO\n";
using std::cin,std::cout,std::endl;auto tam = [] {std::ios::sync_with_stdio(0);cin.tie(0);cout.tie(0);
#ifdef OY_LOCAL
#define CHECKTIME tam.stop()
    freopen("in.txt","r",stdin);freopen("out.txt","w",stdout);struct TIME_AND_MEMORY {static auto GetMicroSecond() {static FILETIME ft;GetSystemTimeAsFileTime(&ft);return ft.dwLowDateTime;};static auto GetMemory() {PROCESS_MEMORY_COUNTERS pmc;GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));return pmc.WorkingSetSize;}uint32_t t0, t1;TIME_AND_MEMORY() : t0(GetMicroSecond()) {}void stop() {t1 = GetMicroSecond();cout << "\ntime cost = " << (t1 - t0) / 10000.0 << " ms\nmemory cost = " << (GetMemory() >> 20) << " MB\n ";t0 = t1;}~TIME_AND_MEMORY() { stop(); }};return TIME_AND_MEMORY();
#else
#define CHECKTIME
    return 0;
#endif
}();
namespace OY {
    struct UnionFind {
        uint32_t m_groupCnt;
        std::vector<uint32_t> m_parent, m_size;
        UnionFind(uint32_t __n = 0) { resize(__n); }
        void resize(uint32_t __n) {
            m_groupCnt = __n;
            m_parent.resize(__n);
            std::iota(m_parent.begin(), m_parent.end(), 0);
            m_size.resize(__n);
            std::fill(m_size.begin(), m_size.end(), 1);
        }
        uint32_t find(uint32_t __i) { return m_parent[__i] == __i ? __i : m_parent[__i] = find(m_parent[__i]); }
        uint32_t size(uint32_t __i) { return m_size[find(__i)]; }
        void uniteTo(uint32_t __headA, uint32_t __headB) {
            if (__headA == __headB) return;
            m_parent[__headA] = __headB;
            m_size[__headB] += m_size[__headA];
            m_groupCnt--;
        }
        bool uniteBySize(uint32_t __a, uint32_t __b) {
            if (__a = find(__a), __b = find(__b); __a == __b) return false;
            if (m_size[__a] > m_size[__b]) std::swap(__a, __b);
            uniteTo(__a, __b);
            return true;
        }
        bool uniteByID(uint32_t __a, uint32_t __b) {
            if (__a = find(__a), __b = find(__b); __a == __b) return false;
            if (__a < __b) std::swap(__a, __b);
            uniteTo(__a, __b);
            return true;
        }
        bool isSame(uint32_t __a, uint32_t __b) { return find(__a) == find(__b); }
        bool isHead(uint32_t __i) { return __i == m_parent[__i]; }
        uint32_t count() const { return m_groupCnt; }
        std::vector<uint32_t> heads() {
            std::vector<uint32_t> ret;
            ret.reserve(m_groupCnt);
            for (uint32_t i = 0; i < m_parent.size(); i++)
                if (isHead(i)) ret.push_back(i);
            return ret;
        }
        std::vector<std::vector<uint32_t>> groups() {
            std::vector<std::vector<uint32_t>> ret;
            ret.resize(m_groupCnt);
            uint32_t index[m_parent.size()];
            for (uint32_t i = 0, j = 0; i < m_parent.size(); i++)
                if (isHead(i)) {
                    ret[j].reserve(m_size[i]);
                    index[i] = j++;
                }
            for (uint32_t i = 0; i < m_parent.size(); i++)
                ret[index[find(i)]].push_back(i);
            return ret;
        }
    };
}

int main() {
    int t;cin>>t;
    while(t--){
        int n;cin>>n;
        int deg[n];for(auto&a:deg)cin>>a;
        OY::UnionFind u(n);
        std::vector<int>indexes;
        for(int i=0;i<n;i++)indexes.push_back(i);
        std::sort(all(indexes),[&](int x,int y){return deg[x]>deg[y];});
        for(int i:indexes)if(u.size(i)==1){
            if(!deg[i])continue;
            for(int k=deg[i];k--;){
                printf("? %d\n",i+1);
                fflush(stdout);
                int a;
                scanf("%d",&a);a--;
                if(u.size(a)>1){
                    u.uniteBySize(i,a);
                    break;
                }
                u.uniteBySize(i,a);
            }
        }
        std::vector<int>color(n,0);
        auto groups=u.groups();
        for(int i=0;i<groups.size();i++){
            for(int a:groups[i])color[a]=i+1;
        }
        printf("!");
        for(auto c:color)printf(" %d",c);
        printf("\n");
        fflush(stdout);
    }
}
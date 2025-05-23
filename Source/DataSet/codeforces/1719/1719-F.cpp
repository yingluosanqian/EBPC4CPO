#pragma GCC optimize("Ofast","unroll-loops","omit-frame-pointer","inline") //Optimization flags
#pragma GCC option("arch=native","tune=native","no-zero-upper") //Enable AVX
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2")
#include<bits/stdc++.h>
#define int long long
//#include<ext/pb_ds/assoc_container.hpp>
using namespace std;
namespace fastio{
    char in[100000];
    int itr=0,llen=0;
    char get(){
        if(itr==llen)llen=fread(in,1,100000,stdin),itr=0;
        if(llen==0)return EOF;
        return in[itr++];
    }
    char out[100000];
    int itr2=0;
    void put(char c){
        out[itr2++]=c;
        if(itr2==100000){
            fwrite(out,1,100000,stdout);
            itr2=0;
        }
    }
    int clear(){
        fwrite(out,1,itr2,stdout);
        itr2=0;
        return 0;
    }
    int getint(){
        int ret=0;char ch=get();
        if(ch=='-')return -getint();
        while (ch<'0'||ch>'9'){
            ch=get();if(ch=='-')return -getint();
        }
        while ('0'<=ch&&ch<='9'){
            ret=ret*10-48+ch;
            ch=get();
        }
        return ret;
    }
    string getstr(){
        string ret="";
        char ch=get();
        while(isspace(ch))ch=get();
        while(!isspace(ch))ret.push_back(ch),ch=get();
        return ret;
    }
    void putstr(string s){
        for(int i=0;i<s.size();i++)put(s[i]);
    }
    template<class T>void putint(T x){
        if(x<0){
            put('-');
            putint(-x);
            return;
        }
        if(x==0){
            put('0');put(' ');
            return;
        }
        char c[40];int pos=0;
        while(x){
            c[pos++]='0'+x%10;
            x/=10;
        }
        for(int i=pos-1;i>=0;i--)put(c[i]);
        put(' ');
    }
    template<class T>void putln(T x){
        if(x<0){
            put('-');
            putln(-x);
            return;
        }
        if(x==0){
            put('0');put('\n');
            return;
        }
        char c[40];int pos=0;
        while(x){
            c[pos++]='0'+x%10;
            x/=10;
        }
        for(int i=pos-1;i>=0;i--)put(c[i]);
        put('\n');
    }
	struct Flusher_ {
		~Flusher_(){clear();}
	}io_flusher_;
}
using namespace fastio;
//using namespace __gnu_pbds;
const int inf=0x3f3f3f3f;
const double eps=1e-6;
const int mod=1e9+7;
typedef long long ll;
#ifndef LOCAL
#define cerr if(0)cout
#define eprintf(...) 0
#else
#define eprintf(...) fprintf(stderr, __VA_ARGS__)
#endif
inline string getstr(string &s,int l,int r){string ret="";for(int i=l;i<=r;i++)ret.push_back(s[i]);return ret;}
int modpow(int x,int y,int md=mod){int ret=1;do{if(y&1)ret=(ll)ret*x%md;x=(ll)x*x%md;}while(y>>=1);return ret;}
inline int Rand(){return rand()*32768+rand();}
int T,n,q,a[200005];
vector<int>fac;
int cnt[20][200005];
multiset<int>val[20];
void eval(){
	int ans=0;
	for(int i=0;i<fac.size();i++){
		ans=max(ans,*val[i].rbegin()*fac[i]);
	}
	putln(ans);
}
signed main(){
	T=getint();
	while(T--){
		n=getint();q=getint();
		int tmp=n;fac.clear();
		for(int i=2;i<=n;i++)if(tmp%i==0){
			fac.push_back(n/i);while(tmp%i==0)tmp/=i;
		}
		for(int i=0;i<fac.size();i++)val[i].clear();
		for(int i=0;i<fac.size();i++)for(int j=0;j<fac[i];j++)cnt[i][j]=0;
		for(int i=1;i<=n;i++)a[i]=getint();
		for(int i=0;i<fac.size();i++){
			for(int j=1;j<=n;j++)cnt[i][j%fac[i]]+=a[j];
			for(int j=0;j<fac[i];j++)val[i].insert(cnt[i][j]);
		}
		eval();
		while(q--){
			int x=getint(),y=getint();
			for(int i=0;i<fac.size();i++){
				val[i].erase(val[i].find(cnt[i][x%fac[i]]));
				cnt[i][x%fac[i]]-=a[x]-y;
				val[i].insert(cnt[i][x%fac[i]]);
			}
			a[x]=y;
			eval();
		}
	}
	return 0;
}
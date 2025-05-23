#ifndef LOCAL
#pragma GCC optimize ("Ofast")
#pragma GCC optimize ("unroll-loops")
#endif

#include <bits/stdc++.h>
using namespace std;

//fast IO by yosupo
//sc.read(string) だと append される
struct Scanner {
    FILE* fp = nullptr;
    char line[(1 << 15) + 1];
    size_t st = 0, ed = 0;
    void reread() {
        memmove(line, line + st, ed - st);
        ed -= st;
        st = 0;
        ed += fread(line + ed, 1, (1 << 15) - ed, fp);
        line[ed] = '\0';
    }
    bool succ() {
        while (true) {
            if (st == ed) {
                reread();
                if (st == ed) return false;
            }
            while (st != ed && isspace(line[st])) st++;
            if (st != ed) break;
        }
        if (ed - st <= 50) reread();
        return true;
    }
    template <class T, enable_if_t<is_same<T, string>::value, int> = 0>
    bool read_single(T& ref) {
        if (!succ()) return false;
        ref.clear();
        while (true) {
            size_t sz = 0;
            while (st + sz < ed && !isspace(line[st + sz])) sz++;
            ref.append(line + st, sz);
            st += sz;
            if (!sz || st != ed) break;
            reread();            
        }
        return true;
    }
    template <class T, enable_if_t<is_integral<T>::value, int> = 0>
    bool read_single(T& ref) {
        if (!succ()) return false;
        bool neg = false;
        if (line[st] == '-') {
            neg = true;
            st++;
        }
        ref = T(0);
        while (isdigit(line[st])) {
            ref = 10 * ref + (line[st++] - '0');
        }
        if (neg) ref = -ref;
        return true;
    }
    template <class T> bool read_single(vector<T>& ref) {
        for (auto& d : ref) {
            if (!read_single(d)) return false;
        }
        return true;
    }
    void read() {}
    template <class H, class... T> void read(H& h, T&... t) {
        bool f = read_single(h);
        assert(f);
        read(t...);
    }
    Scanner(FILE* _fp) : fp(_fp) {}
};

struct Printer {
  public:
    template <bool F = false> void write() {}
    template <bool F = false, class H, class... T>
    void write(const H& h, const T&... t) {
        if (F) write_single(' ');
        write_single(h);
        write<true>(t...);
    }
    template <class... T> void writeln(const T&... t) {
        write(t...);
        write_single('\n');
    }

    Printer(FILE* _fp) : fp(_fp) {}
    ~Printer() { flush(); }

  private:
    static constexpr size_t SIZE = 1 << 15;
    FILE* fp;
    char line[SIZE], small[50];
    size_t pos = 0;
    void flush() {
        fwrite(line, 1, pos, fp);
        pos = 0;
    }
    void write_single(const char& val) {
        if (pos == SIZE) flush();
        line[pos++] = val;
    }
    template <class T, enable_if_t<is_integral<T>::value, int> = 0>
    void write_single(T val) {
        if (pos > (1 << 15) - 50) flush();
        if (val == 0) {
            write_single('0');
            return;
        }
        if (val < 0) {
            write_single('-');
            val = -val;  // todo min
        }
        size_t len = 0;
        while (val) {
            small[len++] = char('0' + (val % 10));
            val /= 10;
        }
        for (size_t i = 0; i < len; i++) {
            line[pos + i] = small[len - 1 - i];
        }
        pos += len;
    }
    void write_single(const string& s) {
        for (char c : s) write_single(c);
    }
    void write_single(const char* s) {
        size_t len = strlen(s);
        for (size_t i = 0; i < len; i++) write_single(s[i]);
    }
    template <class T> void write_single(const vector<T>& val) {
        auto n = val.size();
        for (size_t i = 0; i < n; i++) {
            if (i) write_single(' ');
            write_single(val[i]);
        }
    }
    void write_single(long double d){
		{
			long long v=d;
			write_single(v);
			d-=v;
		}
		write_single('.');
		for(int _=0;_<8;_++){
			d*=10;
			long long v=d;
			write_single(v);
			d-=v;
		}
    }
};

Scanner sc(stdin);
Printer pr(stdout);

using ll=long long;
//#define int ll

#define rng(i,a,b) for(int i=int(a);i<int(b);i++)
#define rep(i,b) rng(i,0,b)
#define gnr(i,a,b) for(int i=int(b)-1;i>=int(a);i--)
#define per(i,b) gnr(i,0,b)
#define pb push_back
#define eb emplace_back
#define a first
#define b second
#define bg begin()
#define ed end()
#define all(x) x.bg,x.ed
#define si(x) int(x.size())
#ifdef LOCAL
#define dmp(x) cerr<<__LINE__<<" "<<#x<<" "<<x<<endl
#else
#define dmp(x) void(0)
#endif

template<class t,class u> bool chmax(t&a,u b){if(a<b){a=b;return true;}else return false;}
template<class t,class u> bool chmin(t&a,u b){if(b<a){a=b;return true;}else return false;}

template<class t> using vc=vector<t>;
template<class t> using vvc=vc<vc<t>>;

using pi=pair<int,int>;
using vi=vc<int>;

template<class t,class u>
ostream& operator<<(ostream& os,const pair<t,u>& p){
	return os<<"{"<<p.a<<","<<p.b<<"}";
}

template<class t> ostream& operator<<(ostream& os,const vc<t>& v){
	os<<"{";
	for(auto e:v)os<<e<<",";
	return os<<"}";
}

#define mp make_pair
#define mt make_tuple
#define one(x) memset(x,-1,sizeof(x))
#define zero(x) memset(x,0,sizeof(x))
#ifdef LOCAL
void dmpr(ostream&os){os<<endl;}
template<class T,class... Args>
void dmpr(ostream&os,const T&t,const Args&... args){
	os<<t<<" ";
	dmpr(os,args...);
}
#define dmp2(...) dmpr(cerr,__LINE__,##__VA_ARGS__)
#else
#define dmp2(...) void(0)
#endif

using uint=unsigned;
using ull=unsigned long long;

template<class t,size_t n>
ostream& operator<<(ostream&os,const array<t,n>&a){
	return os<<vc<t>(all(a));
}

template<int i,class T>
void print_tuple(ostream&,const T&){
}

template<int i,class T,class H,class ...Args>
void print_tuple(ostream&os,const T&t){
	if(i)os<<",";
	os<<get<i>(t);
	print_tuple<i+1,T,Args...>(os,t);
}

template<class ...Args>
ostream& operator<<(ostream&os,const tuple<Args...>&t){
	os<<"{";
	print_tuple<0,tuple<Args...>,Args...>(os,t);
	return os<<"}";
}

template<class t>
void print(t x,int suc=1){
	cout<<x;
	if(suc==1)
		cout<<"\n";
	if(suc==2)
		cout<<" ";
}

ll read(){
	ll i;
	cin>>i;
	return i;
}

vi readvi(int n,int off=0){
	vi v(n);
	rep(i,n)v[i]=read()+off;
	return v;
}

pi readpi(int off=0){
	int a,b;cin>>a>>b;
	return pi(a+off,b+off);
}

template<class t,class u>
void print(const pair<t,u>&p,int suc=1){
	print(p.a,2);
	print(p.b,suc);
}

template<class t,class u>
void print_offset(const pair<t,u>&p,ll off,int suc=1){
	print(p.a+off,2);
	print(p.b+off,suc);
}

template<class T>
void print(const vector<T>&v,int suc=1){
	rep(i,v.size())
		print(v[i],i==int(v.size())-1?suc:2);
}

template<class T>
void print_offset(const vector<T>&v,ll off,int suc=1){
	rep(i,v.size())
		print(v[i]+off,i==int(v.size())-1?suc:2);
}

template<class T,size_t N>
void print(const array<T,N>&v,int suc=1){
	rep(i,N)
		print(v[i],i==int(N)-1?suc:2);
}

string readString(){
	string s;
	cin>>s;
	return s;
}

template<class T>
T sq(const T& t){
	return t*t;
}

void YES(bool ex=true){
	cout<<"YES\n";
	if(ex)exit(0);
	#ifdef LOCAL
	cout.flush();
	#endif
}
void NO(bool ex=true){
	cout<<"NO\n";
	if(ex)exit(0);
	#ifdef LOCAL
	cout.flush();
	#endif
}
void Yes(bool ex=true){
	cout<<"Yes\n";
	if(ex)exit(0);
	#ifdef LOCAL
	cout.flush();
	#endif
}
void No(bool ex=true){
	cout<<"No\n";
	if(ex)exit(0);
	#ifdef LOCAL
	cout.flush();
	#endif
}
//#define CAPITAL
/*
void yes(bool ex=true){
	#ifdef CAPITAL
	cout<<"YES"<<"\n";
	#else
	cout<<"Yes"<<"\n";
	#endif
	if(ex)exit(0);
	#ifdef LOCAL
	cout.flush();
	#endif
}
void no(bool ex=true){
	#ifdef CAPITAL
	cout<<"NO"<<"\n";
	#else
	cout<<"No"<<"\n";
	#endif
	if(ex)exit(0);
	#ifdef LOCAL
	cout.flush();
	#endif
}*/
void possible(bool ex=true){
	#ifdef CAPITAL
	cout<<"POSSIBLE"<<"\n";
	#else
	cout<<"Possible"<<"\n";
	#endif
	if(ex)exit(0);
	#ifdef LOCAL
	cout.flush();
	#endif
}
void impossible(bool ex=true){
	#ifdef CAPITAL
	cout<<"IMPOSSIBLE"<<"\n";
	#else
	cout<<"Impossible"<<"\n";
	#endif
	if(ex)exit(0);
	#ifdef LOCAL
	cout.flush();
	#endif
}

constexpr ll ten(int n){
	return n==0?1:ten(n-1)*10;
}

const ll infLL=LLONG_MAX/3;

#ifdef int
const int inf=infLL;
#else
const int inf=INT_MAX/2-100;
#endif

int topbit(signed t){
	return t==0?-1:31-__builtin_clz(t);
}
int topbit(ll t){
	return t==0?-1:63-__builtin_clzll(t);
}
int topbit(ull t){
	return t==0?-1:63-__builtin_clzll(t);
}
int botbit(signed a){
	return a==0?32:__builtin_ctz(a);
}
int botbit(ll a){
	return a==0?64:__builtin_ctzll(a);
}
int botbit(ull a){
	return a==0?64:__builtin_ctzll(a);
}
int popcount(signed t){
	return __builtin_popcount(t);
}
int popcount(ll t){
	return __builtin_popcountll(t);
}
int popcount(ull t){
	return __builtin_popcountll(t);
}
bool ispow2(int i){
	return i&&(i&-i)==i;
}
ull mask(int i){
	return (ull(1)<<i)-1;
}

bool inc(int a,int b,int c){
	return a<=b&&b<=c;
}

template<class t> void mkuni(vc<t>&v){
	sort(all(v));
	v.erase(unique(all(v)),v.ed);
}

ll rand_int(ll l, ll r) { //[l, r]
	#ifdef LOCAL
	static mt19937_64 gen;
	#else
	static mt19937_64 gen(chrono::steady_clock::now().time_since_epoch().count());
	#endif
	return uniform_int_distribution<ll>(l, r)(gen);
}

template<class t>
void myshuffle(vc<t>&a){
	rep(i,si(a))swap(a[i],a[rand_int(0,i)]);
}

template<class t>
int lwb(const vc<t>&v,const t&a){
	return lower_bound(all(v),a)-v.bg;
}

vvc<int> readGraph(int n,int m){
	vvc<int> g(n);
	rep(i,m){
		int a,b;
		cin>>a>>b;
		//sc.read(a,b);
		a--;b--;
		g[a].pb(b);
		g[b].pb(a);
	}
	return g;
}

vvc<int> readTree(int n){
	return readGraph(n,n-1);
}

vc<ll> presum(const vi&a){
	vc<ll> s(si(a)+1);
	rep(i,si(a))s[i+1]=s[i]+a[i];
	return s;
}

//CF829 F
template<int B>
struct mybitset{
	static constexpr int L=(B+63)/64;
	ull x[L];
	int c=0;
	mybitset():x{},c(0){};
	void set(){
		one(x);
		int p=B/64,q=B%64;
		if(p<L)x[p]=mask(q);
		c=B;
	}
	//not verified
	/*void reset(){
		zero(x);
		c=0;
	}*/
	void set(int i){
		if(!operator[](i))c++;
		x[i/64]|=1ull<<(i%64);
	}
	void reset(int i){
		if(operator[](i))c--;
		x[i/64]&=~(1ull<<(i%64));
	}
	bool operator[](int i)const{return (x[i/64]>>(i%64))&1;}
	bool any(){return c;}
	//i 以上の最小
	int next(int i){
		int p=i/64,q=i%64;
		if(p<L){
			ull v=x[p]>>q;
			if(v)return botbit(v)+i;
			p++;
		}
		rng(j,p,L)if(x[j]){
			return botbit(x[j])+j*64;
		}
		return B;
	}
	//not verified
	//i 未満の最大
	/*int prev(int i){
		int p=i/64,q=i%64;
		if(p<L){
			ull v=x[p]&mask(q);
			if(v)return topbit(v)+p*64;
		}
		per(j,p)if(x[j]){
			return topbit(x[j])+j*64;
		}
		return -1;
	}*/
};

//CF829 F
template<int B>
struct fastset{
	using A=mybitset<B>;
	int n, lg;
	vvc<A> seg;
	fastset(int _n,bool v) : n(_n) {
		A ini;if(v)ini.set();
		do{
			seg.push_back(vc<A>((_n+B-1)/B,ini));
			_n=(_n+B-1)/B;
		}while(_n>1);
		lg=si(seg);
	}
	//not verified
	/*void fillone(){
		for(auto&a:seg)for(auto&v:a)v.set();
	}*/
	bool operator[](int i)const{
		return seg[0][i/B][i%B];
	}
	void set(int i){
		rep(h,lg){
			seg[h][i/B].set(i%B);
			i/=B;
		}
	}
	void reset(int i){
		rep(h,lg){
			seg[h][i/B].reset(i%B);
			if (seg[h][i/B].any())break;
			i/=B;
		}
	}
	//verified only with i=0
	//x以上最小の要素
	int next(int i){
		rep(h,lg){
			if(i/B==si(seg[h]))break;
			int j=seg[h][i/B].next(i%B);
			if(j==B){
				i=i/B+1;
				continue;
			}
			i=i/B*B+j;
			per(g,h){
				i=i*B+seg[g][i].next(0);
			}
			return i;
		}
		return n;
	}
	//not verified
	//x未満最大の要素
	/*int prev(int i){
		i--;
		rep(h,lg){
			if(i==-1)break;
			int j=seg[h][i/B].prev(i%B+1);
			if(j==-1){
				i=i/B-1;
				continue;
			}
			i=i/B*B+j;
			per(g,h){
				i=i*B+seg[g][i].prev(B);
			}
			return i;
		}
		return -1;
	}*/
};

//CF829 F
struct mexgetter{
	const int m;
	vi cnt;
	fastset<1000> fs;
	//m 未満の値しか来ない
	mexgetter(int m_):m(m_),cnt(m),fs(m,true){}
	void add(int v){
		assert(inc(0,v,m-1));
		cnt[v]++;
		if(cnt[v]==1)fs.reset(v);
	}
	void del(int v){
		assert(inc(0,v,m-1));
		cnt[v]--;
		if(cnt[v]==0)fs.set(v);
	}
	int mex(){
		return fs.next(0);
	}
};

void slv(){
	int n,m,q,t;sc.read(n,m,q,t);
	t--;
	mexgetter x(q),y(q);
	vvc<vi> buf(n,vvc<int>(m));
	rep(_,q){
		int i,j,v;sc.read(i,j,v);
		i--;j--;
		if(abs(v)<=q){
			buf[i][j].pb(v);
		}
	}
	rep(i,n)rep(j,m)sort(all(buf[i][j]));
	auto add=[&](int i,int j){
		for(auto v:buf[i][j]){
			if(v>0)x.add(v-1);
			else y.add(-1-v);
		}
	};
	auto del=[&](int i,int j){
		for(auto v:buf[i][j]){
			if(v>0)x.del(v-1);
			else y.del(-1-v);
		}
	};
	int ans=0;
	rng(ini,-(m-1),n){
		int i=0,j=0;
		if(ini<0)j=-ini;
		else i=ini;
		int len=0;
		assert(x.mex()==0);
		assert(y.mex()==0);
		bool ok=true;
		while(i<n&&j<m){
			while((ok&&x.mex()+y.mex()<t)||len<1){
				if(i+len==n||j+len==m){
					ok=false;
					break;
				}
				rep(k,len)add(i+len,j+k);
				rep(k,len+1)add(i+k,j+len);
				len++;
			}
			assert(len>0);
			//dmp2(i,j,len,x.mex(),y.mex());
			if(ok){
				//dmp2(i,j,len);
				ans+=min(n-i,m-j)-len+1;
			}
			rep(k,len)del(i,j+k);
			rng(k,1,len)del(i+k,j);
			i++;j++;len--;
		}
		assert(len==0);
	}
	pr.writeln(ans);
}

signed main(){
	cin.tie(0);
	ios::sync_with_stdio(0);
	cout<<fixed<<setprecision(20);
	
	//int t;cin>>t;rep(_,t)
	slv();
}

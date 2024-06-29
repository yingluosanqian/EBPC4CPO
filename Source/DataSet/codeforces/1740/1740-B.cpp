#pragma region Template
 
#include <bits/stdc++.h>
 
using namespace std;
 
using ll = long long;
using ld = long double;
using ull = unsigned long long;
 
using pii = pair<int, int>;
using pll = pair<ll, ll>;
 
#define mp make_pair
#define mt make_tuple
#define pb push_back
#define eb emplace_back
#define all(x) (x).begin(), (x).end()
#define X first
#define Y second
#define A(x) get<0>(x)
#define B(x) get<1>(x)
#define C(x) get<2>(x)
#define D(x) get<3>(x)
 
template <typename T, typename U>
std::ostream& operator<<(std::ostream& os, const std::pair<T, U>& p) {
    os << p.first << ' ' << p.second << ' ';
    return os;
}
template <typename T, typename U>
std::istream& operator>>(std::istream& is, std::pair<T, U>& p) {
    is >> p.first >> p.second;
    return is;
}
template <typename T>
std::vector<T>& operator--(std::vector<T>& v) {
    for (auto& i : v) --i;
    return v;
}
template <typename T>
std::vector<T>& operator++(std::vector<T>& v) {
    for (auto& i : v) ++i;
    return v;
}
template <typename T>
std::istream& operator>>(std::istream& is, std::vector<T>& v) {
    for (auto& i : v) is >> i;
    return is;
}
template <typename T>
typename std::enable_if<std::is_integral<T>::value, std::ostream>::type& operator<<(std::ostream& os, std::vector<T>& v) {
    for (auto& i : v) os << i << ' ';
    return os;
}
template <typename T>
typename std::enable_if<!std::is_integral<T>::value, std::ostream>::type& operator<<(std::ostream& os, std::vector<T>& v) {
    for (auto& i : v) os << i << '\n';
    return os;
}
template <typename... Args>
std::istream& operator>>(std::istream& is, std::tuple<Args...>& p) {
    std::apply([&is](auto ...x){(..., (is >> x));}, p);
    return is;
}
template <typename... Args>
std::ostream& operator<<(std::ostream& os, std::tuple<Args...>& p) {
    std::apply([&os](auto ...x){(..., (os << x));}, p);
    return os;
}
 
template <typename T>
T inverse(T a, T m) {
    T u = 0, v = 1;
    while (a != 0) {
        T t = m / a;
        m -= t * a;
        swap(a, m);
        u -= t * v;
        swap(u, v);
    }
    assert(m == 1);
    return u;
}
 
template <typename T>
class Modular {
   public:
    using Type = typename decay<decltype(T::value)>::type;
 
    constexpr Modular() : value() {}
    template <typename U>
    Modular(const U& x) {
        value = normalize(x);
    }
 
    template <typename U>
    static Type normalize(const U& x) {
        Type v;
        if (-mod() <= x && x < mod())
            v = static_cast<Type>(x);
        else
            v = static_cast<Type>(x % mod());
        if (v < 0) v += mod();
        return v;
    }
 
    const Type& operator()() const { return value; }
    template <typename U>
    explicit operator U() const { return static_cast<U>(value); }
    constexpr static Type mod() { return T::value; }
 
    Modular& operator+=(const Modular& other) {
        if ((value += other.value) >= mod()) value -= mod();
        return *this;
    }
    Modular& operator-=(const Modular& other) {
        if ((value -= other.value) < 0) value += mod();
        return *this;
    }
    template <typename U>
    Modular& operator+=(const U& other) { return *this += Modular(other); }
    template <typename U>
    Modular& operator-=(const U& other) { return *this -= Modular(other); }
    Modular& operator++() { return *this += 1; }
    Modular& operator--() { return *this -= 1; }
    Modular operator++(int) {
        Modular result(*this);
        *this += 1;
        return result;
    }
    Modular operator--(int) {
        Modular result(*this);
        *this -= 1;
        return result;
    }
    Modular operator-() const { return Modular(-value); }
 
    template <typename U = T>
    typename enable_if<is_same<typename Modular<U>::Type, int>::value, Modular>::type& operator*=(const Modular& rhs) {
#ifdef _WIN32
        uint64_t x = static_cast<int64_t>(value) * static_cast<int64_t>(rhs.value);
        uint32_t xh = static_cast<uint32_t>(x >> 32), xl = static_cast<uint32_t>(x), d, m;
        asm(
            "divl %4; \n\t"
            : "=a"(d), "=d"(m)
            : "d"(xh), "a"(xl), "r"(mod()));
        value = m;
#else
        value = normalize(static_cast<int64_t>(value) * static_cast<int64_t>(rhs.value));
#endif
        return *this;
    }
    template <typename U = T>
    typename enable_if<is_same<typename Modular<U>::Type, int64_t>::value, Modular>::type& operator*=(const Modular& rhs) {
        int64_t q = static_cast<int64_t>(static_cast<long double>(value) * rhs.value / mod());
        value = normalize(value * rhs.value - q * mod());
        return *this;
    }
    template <typename U = T>
    typename enable_if<!is_integral<typename Modular<U>::Type>::value, Modular>::type& operator*=(const Modular& rhs) {
        value = normalize(value * rhs.value);
        return *this;
    }
 
    Modular& operator/=(const Modular& other) { return *this *= Modular(inverse(other.value, mod())); }
 
    template <typename U>
    friend const Modular<U>& abs(const Modular<U>& v) { return v; }
 
    template <typename U>
    friend bool operator==(const Modular<U>& lhs, const Modular<U>& rhs);
 
    template <typename U>
    friend bool operator<(const Modular<U>& lhs, const Modular<U>& rhs);
 
    template <typename U>
    friend std::istream& operator>>(std::istream& stream, Modular<U>& number);
 
   private:
    Type value;
};
 
template <typename T>
bool operator==(const Modular<T>& lhs, const Modular<T>& rhs) { return lhs.value == rhs.value; }
template <typename T, typename U>
bool operator==(const Modular<T>& lhs, U rhs) { return lhs == Modular<T>(rhs); }
template <typename T, typename U>
bool operator==(U lhs, const Modular<T>& rhs) { return Modular<T>(lhs) == rhs; }
 
template <typename T>
bool operator!=(const Modular<T>& lhs, const Modular<T>& rhs) { return !(lhs == rhs); }
template <typename T, typename U>
bool operator!=(const Modular<T>& lhs, U rhs) { return !(lhs == rhs); }
template <typename T, typename U>
bool operator!=(U lhs, const Modular<T>& rhs) { return !(lhs == rhs); }
 
template <typename T>
bool operator<(const Modular<T>& lhs, const Modular<T>& rhs) { return lhs.value < rhs.value; }
 
template <typename T>
Modular<T> operator+(const Modular<T>& lhs, const Modular<T>& rhs) { return Modular<T>(lhs) += rhs; }
template <typename T, typename U>
Modular<T> operator+(const Modular<T>& lhs, U rhs) { return Modular<T>(lhs) += rhs; }
template <typename T, typename U>
Modular<T> operator+(U lhs, const Modular<T>& rhs) { return Modular<T>(lhs) += rhs; }
 
template <typename T>
Modular<T> operator-(const Modular<T>& lhs, const Modular<T>& rhs) { return Modular<T>(lhs) -= rhs; }
template <typename T, typename U>
Modular<T> operator-(const Modular<T>& lhs, U rhs) { return Modular<T>(lhs) -= rhs; }
template <typename T, typename U>
Modular<T> operator-(U lhs, const Modular<T>& rhs) { return Modular<T>(lhs) -= rhs; }
 
template <typename T>
Modular<T> operator*(const Modular<T>& lhs, const Modular<T>& rhs) { return Modular<T>(lhs) *= rhs; }
template <typename T, typename U>
Modular<T> operator*(const Modular<T>& lhs, U rhs) { return Modular<T>(lhs) *= rhs; }
template <typename T, typename U>
Modular<T> operator*(U lhs, const Modular<T>& rhs) { return Modular<T>(lhs) *= rhs; }
 
template <typename T>
Modular<T> operator/(const Modular<T>& lhs, const Modular<T>& rhs) { return Modular<T>(lhs) /= rhs; }
template <typename T, typename U>
Modular<T> operator/(const Modular<T>& lhs, U rhs) { return Modular<T>(lhs) /= rhs; }
template <typename T, typename U>
Modular<T> operator/(U lhs, const Modular<T>& rhs) { return Modular<T>(lhs) /= rhs; }
 
template <typename T, typename U>
Modular<T> power(const Modular<T>& a, const U& b) {
    assert(b >= 0);
    Modular<T> x = a, res = 1;
    U p = b;
    while (p > 0) {
        if (p & 1) res *= x;
        x *= x;
        p >>= 1;
    }
    return res;
}
 
template <typename T>
bool IsZero(const Modular<T>& number) {
    return number() == 0;
}
 
template <typename T>
string to_string(const Modular<T>& number) {
    return to_string(number());
}
 
template <typename T>
std::ostream& operator<<(std::ostream& stream, const Modular<T>& number) {
    return stream << number();
}
 
template <typename T>
std::istream& operator>>(std::istream& stream, Modular<T>& number) {
    typename common_type<typename Modular<T>::Type, int64_t>::type x;
    stream >> x;
    number.value = Modular<T>::normalize(x);
    return stream;
}

struct custom_hash {
    static uint64_t splitmix64(uint64_t x) {
        // http://xorshift.di.unimi.it/splitmix64.c
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }

    size_t operator()(uint64_t x) const {
        static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();
        return splitmix64(x + FIXED_RANDOM);
    }
};

template <typename K, typename V>
using uo_map = std::unordered_map<K, V, custom_hash>;
template <typename V>
using uo_set = std::unordered_set<V, custom_hash>;
 
mt19937 gen(228);
 
#ifdef DEBUG
#define show(x) cout << "[debug " << __LINE__ << "] " << x << endl
#define shows() cout << "=========================" << endl
#define debug cout << "[debug_scope " << __LINE__ << "]" << endl;
#else
#define show(x) ((void)0)
#define shows() ((void)0)
#define debug if (false)
#endif
 
#pragma endregion
 
/*
constexpr int md = 998244353;
using Mint = Modular<std::integral_constant<decay<decltype(md)>::type, md>>;
*/

void solve(int test_number) {
    int n;
    cin >> n;

    vector<pll> v(n);
    cin >> v;

    map<ll, int> mx_cnt;

    ll sm = 0;
    for (int i = 0; i < n; i++) {
        sm += min(v[i].X, v[i].Y);
        mx_cnt[max(v[i].X, v[i].Y)]++;
    }

    ll ans = sm + prev(mx_cnt.end())->first;

    cout << ans * 2 << "\n";
}
 
int main(void) {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int t;
    cin >> t;
    for (int i = 1; i <= t; i++) {
        solve(i);
    }

    return 0;
}

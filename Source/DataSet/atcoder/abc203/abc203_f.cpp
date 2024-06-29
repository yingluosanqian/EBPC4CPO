#include <bits/stdc++.h>




namespace zawa {

using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
using i128 = __int128_t;

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using usize = std::size_t;

} // namespace zawa


namespace zawa {

void SetFastIO() {
    std::cin.tie(nullptr)->sync_with_stdio(false);
}

void SetPrecision(u32 dig) {
    std::cout << std::fixed << std::setprecision(dig);
}

} // namespace zawa
using namespace zawa;

int main() {
    SetFastIO();

    int n, k; std::cin >> n >> k;
    std::vector<int> a(n);
    for (auto& x : a) std::cin >> x;
    std::ranges::sort(a, std::greater<int>{});
    std::vector<int> jump(n);
    int r{};
    for (int l{} ; l < n ; l++) {
        while (r < n and a[r] * 2 > a[l]) r++;
        jump[l] = r;
        if (l == r) r++;
    }
    // for (auto x : a) std::cout << x << ' ';
    // std::cout << std::endl;
    // for (auto x : jump) std::cout << x << ' ';
    // std::cout << std::endl;
    const int m{31};
    const int INF{(int)1e9};
    std::vector dp(m + 1, std::vector<int>(n + 1, INF));
    dp[0][0] = 0;
    for (int i{} ; i < n ; i++) {
        for (int j{} ; j <= m ; j++) {
            if (dp[j][i] == INF) continue;
            if (j < m) dp[j + 1][jump[i]] = std::min(dp[j + 1][jump[i]], dp[j][i]);
            dp[j][i + 1] = std::min(dp[j][i + 1], dp[j][i] + 1);
        }
    }
    for (int i{} ; i <= m ; i++) {
        if (dp[i][n] <= k) {
            std::cout << i << ' ' << dp[i][n] << '\n';
            return 0;
        }
    }
}

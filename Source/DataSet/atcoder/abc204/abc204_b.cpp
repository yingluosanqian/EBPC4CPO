#include <iostream>
#include <vector>

int main() {
    int n;
    std::cin >> n;

    std::vector<int> A(n);

    for (int i = 0; i < n; ++i) {
        std::cin >> A[i];
    }

    int sum = 0;
    for (int i : A) {
        int ins = (i - 10 > 0) ? i - 10 : 0;
        sum += ins;
    }

    std::cout << sum;

    return 0;
}

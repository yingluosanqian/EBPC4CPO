
#include <iostream>
#include <unordered_set>
#include <string>

int main() {
    int N;
    std::cin >> N;

    std::unordered_set<std::string> seen_names;
    bool found = false;

    for (int i = 0; i < N; i++) {
        std::string family_name, given_name;
        std::cin >> family_name >> given_name;

        std::string full_name = family_name + " " + given_name;

        if (seen_names.find(full_name) != seen_names.end()) {
            found = true;
            break;
        }
        seen_names.insert(full_name);
    }

    if (found) {
        std::cout << "Yes\n";
    } else {
        std::cout << "No\n";
    }

    return 0;
}

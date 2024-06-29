#include <iostream>
#include <map>
using namespace std;

int main() {
  map<string, int> mp;
  string str;
  for (int i = 0; i < 4; i++)
  {
    cin >> str;
    mp[str] = 1;
  }
  string values[] = {"2B", "3B", "HR", "H"};

  bool result = true;
  for (string e : values)
    if (mp[e] == 0)
      result = false;
  
  result ? cout << "Yes" : cout << "No";
  return 0;
}

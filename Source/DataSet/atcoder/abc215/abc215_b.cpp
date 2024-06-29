#include <iostream>
#include <cmath>

using namespace std;
int main()
{
	long long n;
	int  m = -1;
	cin >> n;
	while (n > 0) {
		n >>= 1;
		m++;
	}
	cout << m;
}
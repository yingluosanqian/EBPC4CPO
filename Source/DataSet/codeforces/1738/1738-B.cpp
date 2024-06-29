#include <iostream>
#include <math.h>
#include <climits>
#include <string>
#include <algorithm>
#include <bits/stdc++.h>
#include <numeric>
using namespace std;

long long findLevel(int x1, int y, int n){
if(x1 <= (n/2)){
    int x = x1;
    if(y >= x && y <= n - (x) + 1){
    return x;
}
else if(y < x){
    return y;
}
else{
    return (n - y +1);
}
}
else{
int x = n - x1 + 1;
if(y >= x && y <= n - (x) + 1){
return x;
}
else if(y < x){
return y;
}
else{
return (n - y +1);
}
}
}
bool comparator(pair<int, int> a, pair<int, int> b){
    if(a.first == b.first){
      return a.second < b.second;
    }
    return a.first < b.first;
}
long long waterNeeded(long long m, vector<long long>& v){
  long long count = 0;
  for(auto e :  v){
    if(e < m){
      count += (m-e);
    }
  }
  return count;
}
int logg(long long a, long long c){
  long long temp = 1;
  int steps = 0;
  while(temp < a){
    temp *= c;
    steps++;
  }
  return steps;
}
int decimalToBinary(int decimal) {
  int count = 0;
    string binary = "";
    int i = decimal;
    while(i > 0){
        if(i%2 == 0){
            binary = "0" + binary;
        }
        else{
            binary = "1" + binary;
            count++;
        }
        i = i/2;
    }
    return count;
}
vector <int> prime(100000, 1);
void markPrimes(){
  for(int i=2; i<100000; i++){
    if(prime[i]){
      for(int j=2*i; j<100000; j+=i){
        prime[j] = 0; 
      }
    }
  }
}
bool isPalindrome(vector <long long> v){
  long long n = v.size();
  for(long long i=0; i<=(n-1)/2; i++){
   if(v[i] != v[n - i - 1]){
    return false;
   }
  }
   return true;
}
int countDigits(int n){
  int count = 0;
  while(n != 0){
     count++;
     n = n/10;
  }
  return count;
}
vector<int> extractDigits(long long n){
  vector <int> ans;
   while(n != 0){
    ans.push_back(n%10);
    n = n/10;
   }
   reverse(ans.begin(), ans.end());
   return ans;
}
long long toDigit(vector<int> v){
  long long ans = 0;
   for(int i=0; i<v.size(); i++){
       ans = 10*ans + v[i];
   }
   return ans;
}
int pow2(int n){
  int ans = 0;
  while(n > 0){
    ans++;
    n = n/2;
  }
  return ans;
}
long long countInversions(vector<int>v){
  long long k = 0;
  long long ans = 0;
  for(auto e : v){
    if(e == 1){k++;}
    else{
      ans += k;
    }
  }
  return ans;
}
int countSubarraySum(long long k, int n, vector<int>& v){
  int element_count = 0;
  map <int, long long> m;
  int prev_index = -1;
  long long thickness = 0;
  m.insert(make_pair(0, 0));
  long long sum = 0;
  for(int i=0; i<n; i++){
    sum += v[i];
    m[sum] = i+1; 
  }
  sum = 0;
  for(auto e : m){
    if(m.find(e.first - k) != m.end() && m[e.first - k] >= prev_index){
        thickness = max(thickness , (long long)(m[e.first] - m[e.first - k]));
        element_count += m[e.first] - m[e.first - k];
        prev_index = e.second;
    }
  }
  if(element_count == n){return thickness;}
  return -1;
}
long long sum(string str){
  long long ans = 0;
  for(int i=0; i<str.size(); i++){
    if(str[i] == 'L'){
      ans += i;
    }
    else{
      ans += (str.size() - i - 1);
    }
  }
  return ans;
}
// Returns true if str1 is smaller than str2.
bool isSmaller(string str1, string str2)
{
	// Calculate lengths of both string
	int n1 = str1.length(), n2 = str2.length();

	if (n1 < n2)
		return true;
	if (n2 < n1)
		return false;

	for (int i = 0; i < n1; i++)
		if (str1[i] < str2[i])
			return true;
		else if (str1[i] > str2[i])
			return false;

	return false;
}

string findDiff(string str1, string str2)
{
	if (isSmaller(str1, str2))
		swap(str1, str2);
	string str = "";
	int n1 = str1.length(), n2 = str2.length();

	reverse(str1.begin(), str1.end());
	reverse(str2.begin(), str2.end());

	int carry = 0;
	for (int i = 0; i < n2; i++) {
		int sub	= ((str1[i] - '0') - (str2[i] - '0') - carry);
		if (sub < 0) {
			sub = sub + 10;
			carry = 1;
		}
		else
			carry = 0;

		str.push_back(sub + '0');
	}

	for (int i = n2; i < n1; i++) {
		int sub = ((str1[i] - '0') - carry);

		if (sub < 0) {
			sub = sub + 10;
			carry = 1;
		}
		else
			carry = 0;

		str.push_back(sub + '0');
	}

	reverse(str.begin(), str.end());

	return str;
}
int toMins(string str){
vector<int> tokens;
size_t start = 0, end = 0;

while ((end = str.find(':', start)) !=string::npos) {
      tokens.push_back(stoi(str.substr(start, end - start)));
      start = end + 1;
  }
  tokens.push_back(stoi(str.substr(start, end - start)));
  int mins = tokens[0]*60 + tokens[1];
  return mins;
}
bool check(vector<int> v){
  int n = v.size();
  bool ok = true;
  for(int i=0; i<n; i++){v.push_back(v[i]);}
  for(int i=1; i<=n; i++){
    if(v[i] > v[i-1] && v[i] > v[i+1]){
        continue;
    }
    else if(v[i] < v[i-1] && v[i] < v[i+1]){
      continue;
    }
    else{
      return false;
    }
  }
  return ok;
}
bool descomparator(int a, int b){
 return a > b;
}
bool isCharWinner(long long hc, long long dc, long long hm, long long dm){
  long long step1, step2;
  step1 = (hc%dm != 0) ? ((hc/dm) + 1) : (hc/dm);
  step2 = (hm%dc != 0) ? ((hm/dc) + 1) : (hm/dc);
  if(step1 + 1 <= step2){
    return false;
  }
  return true;
}
long long hcf(long long a, long long b){
  if(b > a){
    return hcf(b, a);
  }
  if(a%b == 0){return b;}
  return hcf(b, a%b);
}
vector <int> freqCount(vector<int>& v){
  int n = v.size();
  unordered_map <int, int> m;
  for(int i=0; i< v.size();i++){
  m[v[i]]++;
  }
  vector <int> ans(n,0);
  for(int i=0; i<n; i++){
    ans[i] = m[v[i]];
  }
  return ans;
}
bool checkSetbit(long long n, int pos){
  return n & (1<<pos);
}
bool comparatorString(pair<string, int> a, pair<string, int> b){
  return a.first < b.first;
}
unordered_set <long long> cubes;
void findCubes(){
  for(long long i=1; i<10001; i++){
    cubes.insert(i*i*i);
  }
}
long long findHCF(long long x, long long y){
  long long f = __gcd(x,y);
  return (x*y)/f;
}
map<char, vector<char>> adjList(vector<string> &dictionary){
   map<char, vector<char>> ans;
   int n = dictionary.size();
   for(int i=0; i<n-1; i++){
       int x=0;
       while(x < dictionary[i].size() && x < dictionary[i+1].size()){
           if(dictionary[i][x] != dictionary[i+1][x]){
               ans[dictionary[i][x]].push_back(dictionary[i+1][x]);break;
           }
           x++;
       }
   }
   return ans;
}
string kahnAlgo(map<char, vector<char>>& graph){
    std::string str = "";
    std::unordered_map<char, int> visited;
    for (const auto &e : graph) {
      if(visited.find(e.first) == visited.end()){
            visited.insert(make_pair(e.first,0));
           }
        for (char f : e.second) {
            visited[f]++;
        }
    }
    // for(auto e : visited){
    //   cout<<e.first<<" "<<e.second<<endl;
    // }
    std::queue<char> q;
    for (const auto &alpha : visited) {
        if (alpha.second == 0) {
            q.push(alpha.first);
        }
    }
    while (!q.empty()) {
        char letter = q.front();
        q.pop();
        str.push_back(letter);
        for (char e : graph[letter]) {
            visited[e]--;
            if (visited[e] == 0) {
                q.push(e);
            }
        }
    }
    //cout<<"here is the answer"<<str<<endl;
    return str;
}
string getAlienLanguage(vector<string> &dictionary, int k)
{
    // Write your code here.
    map<char, vector<char>> graph = adjList(dictionary);
    // for(auto e : graph){
    //   cout<<e.first<<"-> ";
    //     for(auto f : e.second){
    //         cout<<f<<" ";
    //     }
    //     cout<<endl;
    // }
    return kahnAlgo(graph);

}
vector<vector<int>> adjList(vector<pair<int, int>>& edges, int n){
vector<vector<int>> ans(n + 1);
for(auto e : edges){
  ans[min(e.first, e.second)].push_back(max(e.first, e.second));
}
return ans;
} 
int graphDis(vector<vector<int>>& graph, int n, int start){
int count = 0;
queue<int> q;
vector<int> visited(n+1, 0);
q.push(start);
visited[start] = 1;
while(!q.empty()){
  int node = q.front();
  if(graph[node].size() == 0){count++;}
  q.pop(); 
  for(auto e : graph[node]){
    if(visited[e] == 0){
      q.push(e);
      visited[e] = 1;
    }
  }

}
return count;
}
// void helper(unordered_map<int, int>& m, int start){
//   m[start]--;
//   if(m[start] == 0){m.erase(start);}
// if(m.find(start + 1) != m.end()){
//   helper(m, start+1);
// }
// } 
bool comparatorFunc(pair<int, int> a, pair<int, int> b){
  return a.first < b.first;
  }
  int binarySearch(vector<int> v, int cur, int n){
    int start = 0;
    int end = n-1;
    while(start <= end){
      int mid = (start + end)/2;
      if(v[mid] == cur){
        while(mid < n && v[mid + 1] == cur){
          mid++;
        }
        return mid;
      }
      else if(v[mid] > cur){
        end = mid - 1;
      }
      else if(v[mid] < cur){
        start = mid + 1;
      }
    }
    return end;
  }
int main() {
int t;
cin>>t;
for(int f=0; f<t; f++){
  int n, k;
  cin>>n>>k;
  vector<int> v(k);
  for(int i=0; i<k; i++){cin>>v[i];}
  int prev_step = v[0];
  if(n-k > 0){
    prev_step = v[0]/(n-k + 1);
    if(v[0]%(n-k + 1) && v[0] > 0){prev_step++;}
  }
  bool ans = true;
  int cur_step = 0;
  for(int i=1; i<k; i++){
     cur_step = v[i]- v[i-1];
     if(cur_step < prev_step){
      ans = false;
      break;
     }
     prev_step = cur_step;
  }
  if(ans){cout<<"YES"<<endl;}
  else{cout<<"NO"<<endl;}
}
  return 0;
}   
/*https://movie-score-http-cxyifxeria-uc.a.run.app*/
/*https://vertex-summarizer-cy2gsamf3q-el.a.run.app/*/
#include <iostream>
using namespace std;
int main(){
  int N;
  cin >> N;
  int first_max_num = 0;
  int first_max_idx = 0;
  int second_max_num = 0;
  int second_max_idx = 0;
  for(int i = 1; i <= N; i++){
    int A;
    cin >> A;
    //第i個數比目前讀入的最大的數還大時
    //原來的最大的數變成第二大的,新的數變成最大的數
    if(A > first_max_num){
      second_max_num = first_max_num;
      second_max_idx = first_max_idx;
      first_max_num = A;
      first_max_idx = i;
    }else if(A > second_max_num){
      //讀入的數沒有大於最大的，但大於第二大的
      second_max_num = A;
      second_max_idx = i;
    }
  }
  cout << second_max_idx << endl;
  return 0;
}